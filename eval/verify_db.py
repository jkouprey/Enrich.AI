"""
eval/verify_db.py — HEADLINE metric: claims vs DATABASE ground truth.

Ground truth is the fetched UniProt / NCBI Gene / GO record. Sonnet performs ONLY the
comparison between a claim and that fetched text - it is explicitly forbidden from using
its own knowledge, and must quote the source sentence verbatim so every verdict is
spot-checkable. This is deliberately not the model's memory: the tool under test is an
LLM, so an LLM's recall shares its blind spots.

Batched BY GENE: each gene's record is sent once with all claims about it (2.8 claims
per gene on average), which roughly halves the token cost versus per-claim requests.

Flow:
    python eval\\verify_db.py --fetch        # free: build the gene record cache
    python eval\\verify_db.py --estimate     # free: price the comparison batch
    python eval\\verify_db.py --run          # submit + collect + CSV + stats
    python eval\\verify_db.py --collect BATCHID   # re-collect an existing batch (free)
"""
from __future__ import annotations
import argparse, csv, json, glob, math, os, sys, time
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE)); sys.path.insert(0, str(BASE.parent))

from verify_claims import genes_in, mygene, uniprot_function, go_terms
from evalkit import parse_json_array

CACHE = BASE / "data" / "_gene_records.json"
OUT_CSV = BASE / "results" / "verify_db_headline.csv"
MODEL = "claude-sonnet-4-5-20250929"
PRICE_IN, PRICE_OUT, DISC = 3.00, 15.00, 0.50

PROMPT = """You are verifying biological claims against an authoritative database record.

DATABASE RECORD for {gene}
UniProt FUNCTION: {uniprot}
NCBI Gene summary: {ncbi}
GO annotations (term [aspect/evidence code]): {go}

CLAIMS about {gene}:
{claims}

Judge each claim USING ONLY THE RECORD ABOVE. Do not use your own knowledge of biology -
if the record does not settle the question, the answer is "uncertain".
- "correct"   : the record supports what the claim asserts
- "incorrect" : the record contradicts it (different function, localisation, direction of
                effect, pathway, or disease association than the claim states)
- "uncertain" : the record neither supports nor contradicts it

"quote" MUST be a sentence or clause copied VERBATIM from the record above. Never
paraphrase, never invent. If no part of the record is relevant, use an empty string.

Return ONLY a JSON array, one object per claim in order:
[{{"i":0,"label":"correct","quote":"..."}}, ...]
"""


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0.0, c-h), min(1.0, c+h))


def collect_claims(dirs):
    by_gene, interpretive = defaultdict(list), []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            if os.path.basename(f).startswith("_"): continue
            rec = json.load(open(f, encoding="utf-8"))
            for c in rec.get("judge_claims") or []:
                g = genes_in(c["claim"])
                if g: by_gene[g[0]].append({"gene_set": rec["gene_set"], "claim": c["claim"],
                                            "judge_label": c["label"]})
                else: interpretive.append({"gene_set": rec["gene_set"], "claim": c["claim"],
                                           "judge_label": c["label"]})
    return by_gene, interpretive


def fetch_records(genes):
    cache = json.load(open(CACHE, encoding="utf-8")) if CACHE.exists() else {}
    todo = [g for g in genes if g not in cache]
    print(f"gene records: {len(cache)} cached, {len(todo)} to fetch")
    for i, g in enumerate(todo, 1):
        mg = mygene(g)
        rec = {"uniprot": "", "ncbi": "", "go": [], "entrez": ""}
        if mg and not mg.get("_error"):
            rec["ncbi"] = (mg.get("summary") or "")[:1500]
            rec["entrez"] = str(mg.get("entrezgene") or "")
            up = mg.get("uniprot") or {}
            acc = up.get("Swiss-Prot") if isinstance(up, dict) else None
            if isinstance(acc, list): acc = acc[0]
            if acc:
                rec["uniprot"] = uniprot_function(acc)
                rec["acc"] = acc
            rec["go"] = go_terms(mg)[:20]
        cache[g] = rec
        if i % 50 == 0:
            print(f"  {i}/{len(todo)}"); CACHE.write_text(json.dumps(cache), encoding="utf-8")
        time.sleep(0.1)
    CACHE.write_text(json.dumps(cache), encoding="utf-8")
    usable = sum(1 for g in genes if cache.get(g, {}).get("uniprot") or cache.get(g, {}).get("ncbi"))
    print(f"  cached {len(cache)} records; {usable}/{len(genes)} genes have UniProt/NCBI text")
    return cache


def build_requests(by_gene, cache):
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    reqs, index = [], {}
    for gene, items in sorted(by_gene.items()):
        rec = cache.get(gene) or {}
        if not (rec.get("uniprot") or rec.get("ncbi") or rec.get("go")):
            continue                      # no source -> uncertain, no API call needed
        numbered = "\n".join(f"{i}. {it['claim']}" for i, it in enumerate(items))
        prompt = PROMPT.format(gene=gene, uniprot=rec.get("uniprot") or "(none)",
                               ncbi=rec.get("ncbi") or "(none)",
                               go=" | ".join(rec.get("go") or []) or "(none)",
                               claims=numbered)
        cid = gene[:64]
        index[cid] = (gene, items)
        reqs.append(Request(custom_id=cid, params=MessageCreateParamsNonStreaming(
            model=MODEL, max_tokens=8000, temperature=0,
            messages=[{"role": "user", "content": prompt}])))
    return reqs, index


def write_csv(rows):
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["gene_set", "claim", "gene", "label",
                                          "quoted_source_sentence", "source_db",
                                          "source_id", "source_url", "judge_label"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows):,} rows -> {OUT_CSV}")


def report(rows):
    from collections import Counter
    c = Counter(r["label"] for r in rows)
    correct, incorrect, unc = c["correct"], c["incorrect"], c["uncertain"]
    n = correct + incorrect
    rate = incorrect / n if n else 0.0
    lo, hi = wilson(incorrect, n)
    print("\n=== DATABASE-VERIFIED HALLUCINATION RATE ===")
    print(f"  correct   : {correct:,}")
    print(f"  incorrect : {incorrect:,}")
    print(f"  uncertain : {unc:,}  (excluded from the rate)")
    print(f"  rate = incorrect/(correct+incorrect) = {incorrect:,}/{n:,} = {rate:.2%}")
    print(f"  95% Wilson CI: [{lo:.2%}, {hi:.2%}]")
    return {"correct": correct, "incorrect": incorrect, "uncertain": unc,
            "rate": rate, "ci_low": lo, "ci_high": hi, "n_scored": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", default=["eval/runs/full_run"])
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--collect", default=None, help="batch id to re-collect (free)")
    args = ap.parse_args()

    by_gene, interpretive = collect_claims(args.dirs)
    print(f"claims naming a gene: {sum(len(v) for v in by_gene.values()):,} across "
          f"{len(by_gene):,} genes | interpretive: {len(interpretive):,}")

    if args.fetch:
        fetch_records(list(by_gene)); return

    cache = json.load(open(CACHE, encoding="utf-8")) if CACHE.exists() else {}
    if not cache: sys.exit("No gene cache. Run --fetch first.")

    import anthropic
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key: sys.exit("ERROR: set ANTHROPIC_API_KEY (.\\load_env.ps1)")
    client = anthropic.Anthropic(api_key=key)

    reqs, index = build_requests(by_gene, cache)
    print(f"comparison requests (one per gene with a record): {len(reqs):,}")

    if args.estimate:
        tot = 0
        for r in reqs[:40]:
            tot += client.messages.count_tokens(
                model=MODEL, messages=r["params"]["messages"]).input_tokens
        avg = tot / min(40, len(reqs))
        est_in = avg * len(reqs)
        est_out = sum(len(v) * 55 + 80 for g, v in index.values())
        print(f"  measured avg input/request : {avg:,.0f} tok (40 sampled)")
        print(f"  est. total input           : {est_in:,.0f}")
        print(f"  est. total output          : {est_out:,.0f}")
        print(f"  EST. COST                  : ${(est_in/1e6*PRICE_IN + est_out/1e6*PRICE_OUT)*(1-DISC):.2f}")
        return

    bid = args.collect
    if not bid:
        b = client.messages.batches.create(requests=reqs)
        bid = b.id
        print(f"submitted batch {bid}")
    while True:
        st = client.messages.batches.retrieve(bid)
        if st.processing_status == "ended": break
        rc = st.request_counts
        print(f"  {st.processing_status}: processing={rc.processing} ok={rc.succeeded} err={rc.errored}")
        time.sleep(30)

    rows, in_tok, out_tok = [], 0, 0
    for res in client.messages.batches.results(bid):
        ent = index.get(res.custom_id)
        if ent is None: continue
        gene, items = ent
        if res.result.type != "succeeded":
            for it in items:
                rows.append({**it, "gene": gene, "label": "uncertain",
                             "quoted_source_sentence": "", "source_db": "", "source_id": gene,
                             "source_url": "", "judge_label": it["judge_label"]})
            continue
        msg = res.result.message
        in_tok += msg.usage.input_tokens; out_tok += msg.usage.output_tokens
        txt = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        verdicts = {int(o["i"]): o for o in parse_json_array(txt)
                    if isinstance(o, dict) and "i" in o}
        rec = cache.get(gene, {})
        acc = rec.get("acc", "")
        for i, it in enumerate(items):
            v = verdicts.get(i, {})
            lab = str(v.get("label", "uncertain")).lower()
            if lab not in ("correct", "incorrect", "uncertain"): lab = "uncertain"
            rows.append({"gene_set": it["gene_set"], "claim": it["claim"], "gene": gene,
                         "label": lab, "quoted_source_sentence": v.get("quote", ""),
                         "source_db": "UniProt" if rec.get("uniprot") else ("NCBI Gene" if rec.get("ncbi") else "GO"),
                         "source_id": acc or rec.get("entrez", ""),
                         "source_url": (f"https://www.uniprot.org/uniprotkb/{acc}" if acc else
                                        f"https://www.ncbi.nlm.nih.gov/gene/{rec.get('entrez','')}"),
                         "judge_label": it["judge_label"]})

    # genes with no record + interpretive claims -> uncertain, no API call was made
    for gene, items in by_gene.items():
        if gene[:64] in index: continue
        for it in items:
            rows.append({**it, "gene": gene, "label": "uncertain", "quoted_source_sentence": "",
                         "source_db": "", "source_id": "", "source_url": "",
                         "judge_label": it["judge_label"]})
    for it in interpretive:
        rows.append({**it, "gene": "", "label": "uncertain", "quoted_source_sentence": "",
                     "source_db": "", "source_id": "", "source_url": "",
                     "judge_label": it["judge_label"]})

    write_csv(rows)
    stats = report(rows)
    stats["batch_id"] = bid
    stats["cost"] = (in_tok/1e6*PRICE_IN + out_tok/1e6*PRICE_OUT)*(1-DISC)
    print(f"  spend: ${stats['cost']:.2f}  ({in_tok:,} in / {out_tok:,} out)")
    json.dump(stats, open(BASE/"results"/"verify_db_stats.json", "w"), indent=2)


if __name__ == "__main__":
    main()
