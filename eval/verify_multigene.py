"""
eval/verify_multigene.py — FIX for the first-gene-only limitation in verify_db.py.

verify_db.py assigned each claim to genes_in(claim)[0], so a claim naming several genes
was checked against ONE record. This re-verifies every multi-gene claim against ALL the
genes it names, one record per gene, and aggregates per-gene verdicts in CODE (the model
never applies the aggregation rule - it only compares one gene to one record at a time).

AGGREGATION RULE (primary, "any-contradiction"):
    incorrect  if ANY named gene's record contradicts the claim
    correct    if >=1 gene supports it and NONE contradicts
    uncertain  if no gene supports and none contradicts
A silent record is uninformative, not a failure - requiring positive support from every
gene would re-import the absence-as-contradiction bias this fix exists to remove.

A stricter variant ("all-must-support": correct only if EVERY gene supports) is reported
alongside it for transparency.

Flow:
    python eval\\verify_multigene.py --fetch      # free: top up the gene record cache
    python eval\\verify_multigene.py --estimate   # free: price the batch
    python eval\\verify_multigene.py --run        # submit + collect + patch CSV
    python eval\\verify_multigene.py --collect ID # re-collect an existing batch (free)
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE)); sys.path.insert(0, str(BASE.parent))

from verify_claims import genes_in, mygene, uniprot_function, go_terms
from verify_db import CACHE, MODEL, PRICE_IN, PRICE_OUT, DISC, wilson
from evalkit import parse_json_array

IN_CSV = BASE / "results" / "verify_db_headline.csv"
OUT_CSV = BASE / "results" / "verify_db_headline_v2.csv"
DETAIL_CSV = BASE / "results" / "verify_multigene_detail.csv"

UP_CHARS, NC_CHARS, GO_N = 700, 500, 10

PROMPT = """You are verifying one biological claim against authoritative database records.

CLAIM: {claim}

The claim names {n} genes. Below is the database record for each. Judge the claim
SEPARATELY against each gene's record, one verdict per gene.

{records}

For EACH gene, judge the claim USING ONLY THAT GENE'S RECORD. Do not use your own
knowledge of biology - if the record does not settle the question, the answer is
"uncertain". Do not let one gene's record influence another gene's verdict.
- "correct"   : that gene's record supports what the claim asserts about it
- "incorrect" : that gene's record contradicts it (different function, localisation,
                direction of effect, pathway, or disease association than the claim states)
- "uncertain" : that gene's record neither supports nor contradicts it

"quote" MUST be copied VERBATIM from that gene's record above. Never paraphrase, never
invent. If no part of that record is relevant, use an empty string.

Return ONLY a JSON array, one object per gene in the order listed:
[{{"gene":"...","label":"correct","quote":"..."}}, ...]
"""


def load_multi():
    rows = list(csv.DictReader(open(IN_CSV, encoding="utf-8")))
    multi = [(i, r, genes_in(r["claim"])) for i, r in enumerate(rows)
             if r["gene"] and len(genes_in(r["claim"])) > 1]
    return rows, multi


def fetch_missing(genes):
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
                rec["uniprot"] = uniprot_function(acc); rec["acc"] = acc
            rec["go"] = go_terms(mg)[:20]
        cache[g] = rec
        if i % 25 == 0:
            print(f"  {i}/{len(todo)}"); CACHE.write_text(json.dumps(cache), encoding="utf-8")
        time.sleep(0.1)
    CACHE.write_text(json.dumps(cache), encoding="utf-8")
    have = sum(1 for g in genes if cache.get(g, {}).get("uniprot") or cache.get(g, {}).get("ncbi"))
    print(f"  cached {len(cache)} total; {have}/{len(genes)} of the multi-gene symbols have text")
    return cache


def render(gene, rec):
    return (f"--- {gene} ---\n"
            f"UniProt FUNCTION: {(rec.get('uniprot') or '(none)')[:UP_CHARS]}\n"
            f"NCBI Gene summary: {(rec.get('ncbi') or '(none)')[:NC_CHARS]}\n"
            f"GO annotations: {' | '.join((rec.get('go') or [])[:GO_N]) or '(none)'}")


def build_requests(multi, cache):
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    reqs, index = [], {}
    for i, r, genes in multi:
        usable = [g for g in genes
                  if (cache.get(g) or {}).get("uniprot") or (cache.get(g) or {}).get("ncbi")
                  or (cache.get(g) or {}).get("go")]
        if len(usable) < 2:
            continue                      # nothing gained over the original single-gene check
        body = "\n\n".join(render(g, cache.get(g) or {}) for g in usable)
        prompt = PROMPT.format(claim=r["claim"], n=len(usable), records=body)
        cid = f"row{i}"
        index[cid] = (i, r, usable)
        reqs.append(Request(custom_id=cid, params=MessageCreateParamsNonStreaming(
            model=MODEL, max_tokens=4000, temperature=0,
            messages=[{"role": "user", "content": prompt}])))
    return reqs, index


def aggregate(per_gene):
    """per_gene: list of (gene, label, quote) -> (primary_label, strict_label)"""
    labs = [l for _, l, _ in per_gene]
    if "incorrect" in labs:
        primary = "incorrect"
    elif "correct" in labs:
        primary = "correct"
    else:
        primary = "uncertain"
    if labs and all(l == "correct" for l in labs):
        strict = "correct"
    elif "incorrect" in labs:
        strict = "incorrect"
    else:
        strict = "uncertain"
    return primary, strict


def report(rows, key="label"):
    c = Counter(r[key] for r in rows)
    n = c["correct"] + c["incorrect"]
    lo, hi = wilson(c["incorrect"], n)
    return {"correct": c["correct"], "incorrect": c["incorrect"], "uncertain": c["uncertain"],
            "n_scored": n, "rate": c["incorrect"]/n if n else 0.0, "ci_low": lo, "ci_high": hi}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--collect", default=None)
    args = ap.parse_args()

    rows, multi = load_multi()
    allg = sorted({g for _, _, gs in multi for g in gs})
    print(f"multi-gene claims: {len(multi):,} | distinct genes: {len(allg):,}")

    if args.fetch:
        fetch_missing(allg); return

    cache = json.load(open(CACHE, encoding="utf-8")) if CACHE.exists() else {}
    if not cache: sys.exit("No gene cache. Run --fetch first.")

    import anthropic
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key: sys.exit("ERROR: set ANTHROPIC_API_KEY (.\\load_env.ps1)")
    client = anthropic.Anthropic(api_key=key)

    reqs, index = build_requests(multi, cache)
    print(f"requests (claims with >=2 usable records): {len(reqs):,}")

    if args.estimate:
        tot = 0
        n = min(30, len(reqs))
        for r in reqs[:n]:
            tot += client.messages.count_tokens(model=MODEL, messages=r["params"]["messages"]).input_tokens
        avg = tot / n
        est_in = avg * len(reqs)
        est_out = sum(len(u) * 60 + 60 for _, _, u in index.values())
        print(f"  measured avg input/request : {avg:,.0f} tok ({n} sampled)")
        print(f"  est. total input           : {est_in:,.0f}")
        print(f"  est. total output          : {est_out:,.0f}")
        print(f"  EST. COST                  : ${(est_in/1e6*PRICE_IN + est_out/1e6*PRICE_OUT)*(1-DISC):.2f}")
        return

    bid = args.collect
    if not bid:
        b = client.messages.batches.create(requests=reqs); bid = b.id
        print(f"submitted batch {bid}")
    while True:
        st = client.messages.batches.retrieve(bid)
        if st.processing_status == "ended": break
        rc = st.request_counts
        print(f"  {st.processing_status}: processing={rc.processing} ok={rc.succeeded} err={rc.errored}")
        time.sleep(30)

    detail, patched, in_tok, out_tok = [], {}, 0, 0
    for res in client.messages.batches.results(bid):
        ent = index.get(res.custom_id)
        if ent is None: continue
        i, row, usable = ent
        if res.result.type != "succeeded": continue
        msg = res.result.message
        in_tok += msg.usage.input_tokens; out_tok += msg.usage.output_tokens
        txt = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        objs = [o for o in parse_json_array(txt) if isinstance(o, dict)]
        per_gene = []
        for j, g in enumerate(usable):
            o = next((x for x in objs if str(x.get("gene", "")).upper() == g.upper()), None)
            if o is None and j < len(objs): o = objs[j]
            lab = str((o or {}).get("label", "uncertain")).lower()
            if lab not in ("correct", "incorrect", "uncertain"): lab = "uncertain"
            per_gene.append((g, lab, (o or {}).get("quote", "")))
        primary, strict = aggregate(per_gene)
        patched[i] = (primary, strict, per_gene)
        for g, lab, q in per_gene:
            rec = cache.get(g, {})
            detail.append({"gene_set": row["gene_set"], "claim": row["claim"], "gene": g,
                           "gene_label": lab, "quote": q,
                           "aggregated_label": primary, "strict_label": strict,
                           "old_label_first_gene_only": row["label"],
                           "source_url": (f"https://www.uniprot.org/uniprotkb/{rec['acc']}"
                                          if rec.get("acc") else
                                          f"https://www.ncbi.nlm.nih.gov/gene/{rec.get('entrez','')}")})

    with open(DETAIL_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(detail[0])); w.writeheader(); w.writerows(detail)
    print(f"wrote {len(detail):,} per-gene verdicts -> {DETAIL_CSV}")

    before = report(rows)
    strict_rows = [dict(r) for r in rows]
    changed = Counter()
    for i, (primary, strict, per_gene) in patched.items():
        changed[(rows[i]["label"], primary)] += 1
        rows[i]["label"] = primary
        rows[i]["gene"] = ",".join(g for g, _, _ in per_gene)
        rows[i]["quoted_source_sentence"] = " || ".join(
            f"[{g}] {q}" for g, l, q in per_gene if q and l != "uncertain")
        strict_rows[i]["label"] = strict
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote patched headline -> {OUT_CSV}")

    after = report(rows)
    strict = report(strict_rows)
    print("\n=== LABEL CHANGES (old first-gene-only -> new all-gene) ===")
    for (a, b), n in sorted(changed.items(), key=lambda kv: -kv[1]):
        print(f"  {a:>9} -> {b:<9} : {n}")
    for name, s in (("BEFORE (first-gene-only)", before), ("AFTER (any-contradiction)", after),
                    ("AFTER (all-must-support)", strict)):
        print(f"\n{name}")
        print(f"  correct {s['correct']:,} | incorrect {s['incorrect']:,} | uncertain {s['uncertain']:,}")
        print(f"  rate = {s['incorrect']:,}/{s['n_scored']:,} = {s['rate']:.2%}  "
              f"CI [{s['ci_low']:.2%}, {s['ci_high']:.2%}]")
    cost = (in_tok/1e6*PRICE_IN + out_tok/1e6*PRICE_OUT)*(1-DISC)
    print(f"\n  spend: ${cost:.2f}  ({in_tok:,} in / {out_tok:,} out)")
    json.dump({"batch_id": bid, "cost": cost, "before": before, "after_any_contradiction": after,
               "after_all_must_support": strict,
               "changes": {f"{a}->{b}": n for (a, b), n in changed.items()}},
              open(BASE/"results"/"verify_multigene_stats.json", "w"), indent=2)


if __name__ == "__main__":
    main()
