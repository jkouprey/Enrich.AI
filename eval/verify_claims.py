"""
eval/verify_claims.py — database verification of judged claims.

For each sampled claim: resolve the gene symbol, pull the authoritative record from
UniProt (FUNCTION comment - a real quotable sentence), MyGene/NCBI (RefSeq summary)
and GO (annotated terms), and emit everything needed to adjudicate the claim by hand.

Honesty rules baked in:
  * quoted text is copied verbatim from the source - never paraphrased
  * GO annotations are gene->term links with no sentence, so the TERM DEFINITION is
    quoted and the evidence code recorded in notes
  * a claim with no clean source is left for an "uncertain" verdict, not forced

Usage:
    python eval\\verify_claims.py --label GROUNDED --n 120 --out eval/verify_grounded.csv
"""
from __future__ import annotations
import argparse, csv, json, glob, os, random, re, sys, time
from collections import defaultdict
from pathlib import Path

import httpx

BASE = Path(__file__).resolve().parent
STOP = {"DNA", "RNA", "ATP", "GO", "KEGG", "NADPH", "TCA", "ROS", "MHC", "TNF", "NF", "IL",
        "UV", "ER", "GTP", "II", "I", "THE", "AND", "OR", "PPAR", "MAPK", "AMPK", "EMT",
        "ECM", "TGF", "JAK", "STAT", "PI3K", "AKT", "MTOR", "APC", "COPI", "COPII", "ATPase"}
GENE_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,7}(?:-[A-Z0-9]{1,4})?\b")

client = httpx.Client(timeout=30, headers={"User-Agent": "EnrichAI-eval/1.0"})
_cache: dict = {}


def genes_in(text: str):
    return [g for g in dict.fromkeys(GENE_RE.findall(text)) if g not in STOP and len(g) > 2]


def mygene(sym: str):
    key = f"mg:{sym}"
    if key in _cache: return _cache[key]
    out = {}
    try:
        r = client.get("https://mygene.info/v3/query",
                       params={"q": f"symbol:{sym}", "species": "human",
                               "fields": "entrezgene,uniprot,summary,name,go"})
        hits = r.json().get("hits") or []
        if hits: out = hits[0]
    except Exception as e:
        out = {"_error": str(e)[:100]}
    _cache[key] = out
    return out


def uniprot_function(acc: str):
    """Verbatim FUNCTION comment from UniProt - a genuinely quotable sentence."""
    key = f"up:{acc}"
    if key in _cache: return _cache[key]
    txt = ""
    try:
        r = client.get(f"https://rest.uniprot.org/uniprotkb/{acc}.json",
                       params={"fields": "cc_function"})
        for c in r.json().get("comments", []):
            if c.get("commentType") == "FUNCTION":
                for t in c.get("texts", []):
                    if t.get("value"):
                        txt = t["value"]; break
            if txt: break
    except Exception as e:
        txt = ""
    _cache[key] = txt
    return txt


def go_terms(mg: dict):
    """Annotated GO terms with evidence codes (no sentence exists for an annotation)."""
    out = []
    go = mg.get("go") or {}
    for aspect in ("BP", "MF", "CC"):
        entries = go.get(aspect) or []
        if isinstance(entries, dict): entries = [entries]
        for e in entries:
            if isinstance(e, dict) and e.get("term"):
                out.append(f"{e['term']} [{aspect}/{e.get('evidence','?')}]")
    return out


OLD = {}
try:
    OLD = json.load(open(Path(__file__).resolve().parent.parent / 'old' / 'eval' / 'reverted_strict_criterion_experiment' / '_old_labels_5sets.json', encoding='utf-8'))
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="GROUNDED", help="label to sample, or ALL")
    ap.add_argument("--sets", nargs="*", default=None, help="restrict to these gene sets")
    ap.add_argument("--all", action="store_true", help="take every claim, no sampling")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--dirs", nargs="*", default=["eval/runs/full_run"])
    ap.add_argument("--out", default="eval/results/verify_grounded.csv")
    args = ap.parse_args()

    random.seed(17)
    by_set = defaultdict(list)
    for d in args.dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            if os.path.basename(f).startswith("_"): continue
            rec = json.load(open(f, encoding="utf-8"))
            for c in rec.get("judge_claims") or []:
                if args.sets and rec["gene_set"] not in args.sets: continue
                if args.label == "ALL" or c["label"] == args.label:
                    by_set[rec["gene_set"]].append((rec, c))

    # stratify: proportional-ish across sets, at least 1 each, capped so no set dominates
    sets = sorted(by_set)
    if args.all:
        sample = [x for s in sets for x in by_set[s]]
    else:
        per = max(1, args.n // max(len(sets), 1))
        sample = []
        for s in sets:
            pool = by_set[s]
            sample += random.sample(pool, min(per, len(pool)))
        random.shuffle(sample)
        sample = sample[:args.n]
    print(f"{args.label}: {sum(len(v) for v in by_set.values())} total across {len(sets)} sets "
          f"-> sampling {len(sample)}\n")

    rows = []
    for i, (rec, c) in enumerate(sample, 1):
        claim = c["claim"]
        gs = genes_in(claim)
        gene = uni_txt = summary = ""
        gos = []
        for g in gs[:2]:
            mg = mygene(g)
            if not mg or mg.get("_error"): continue
            gene = g
            summary = (mg.get("summary") or "")
            up = mg.get("uniprot") or {}
            acc = up.get("Swiss-Prot") if isinstance(up, dict) else None
            if isinstance(acc, list): acc = acc[0]
            if acc: uni_txt = uniprot_function(acc)
            gos = go_terms(mg)
            if uni_txt or summary or gos: break
        rows.append({
            "gene_set": rec["gene_set"], "rep": rec.get("rep", 1), "judge_label": c["label"],
            "old_label": OLD.get(rec["gene_set"], {}).get(claim, ""),
            "judge_why": c["why"], "claim": claim, "gene": gene,
            "uniprot_function": uni_txt, "ncbi_summary": summary[:1200],
            "go_terms": " | ".join(gos[:14]),
        })
        if i % 20 == 0: print(f"  fetched {i}/{len(sample)}")
        time.sleep(0.12)

    out = Path(args.out)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {out}")
    have_up = sum(1 for r in rows if r["uniprot_function"])
    have_go = sum(1 for r in rows if r["go_terms"])
    have_nc = sum(1 for r in rows if r["ncbi_summary"])
    no_gene = sum(1 for r in rows if not r["gene"])
    print(f"  with UniProt FUNCTION text : {have_up}")
    print(f"  with GO annotations        : {have_go}")
    print(f"  with NCBI summary          : {have_nc}")
    print(f"  no resolvable gene symbol  : {no_gene}  (interpretive claims -> likely 'uncertain')")


if __name__ == "__main__":
    main()
