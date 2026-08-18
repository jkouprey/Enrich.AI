"""
eval/recheck_incorrect.py — adjudicate every "incorrect" verdict as a GENUINE biological
error or a strictness artifact.

The headline verdicts were produced under a deliberately literal instruction ("if the
record does not settle it, answer uncertain"), and inspection showed two recurring ways a
claim gets flagged incorrect without actually being false:

  ABSENCE    - the record simply does not mention what the claim asserts, and silence was
               read as contradiction (e.g. PAX6/beta-cell identity: UniProt mentions only
               alpha cells, so the beta-cell role reads as "contradicted" though it is
               well established and merely unstated).
  TERMINOLOGY- the biology is right but the wording is imprecise (e.g. MAML1 described as
               "a transcription factor in the Notch pathway" when the record says
               "transcriptional coactivator for NOTCH").

Only GENUINE survives into the corrected rate. Each adjudication must quote the exact span
of the record that contradicts (GENUINE) or must state what the record does NOT say
(ABSENCE), so every reclassification is checkable by hand.

This is a SEPARATE pass with its own prompt - it does not re-run the original comparison,
it audits it.

Flow:
    python eval\\recheck_incorrect.py --estimate
    python eval\\recheck_incorrect.py --run
    python eval\\recheck_incorrect.py --collect ID
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE)); sys.path.insert(0, str(BASE.parent))

from verify_db import CACHE, MODEL, PRICE_IN, PRICE_OUT, DISC, wilson
from evalkit import parse_json_array

OUT_CSV = BASE / "results" / "recheck_incorrect.csv"
STATS = BASE / "results" / "recheck_incorrect_stats.json"

PROMPT = """A claim about the gene {gene} was previously judged INCORRECT against the
database record below. Audit that judgement.

DATABASE RECORD for {gene}
UniProt FUNCTION: {uniprot}
NCBI Gene summary: {ncbi}
GO annotations (term [aspect/evidence code]): {go}

CLAIM: {claim}
QUOTE the earlier judgement relied on: {quote}

Classify the earlier INCORRECT judgement into exactly one category:

- "GENUINE": the record affirmatively contradicts the claim. The claim asserts a function,
  localisation, direction of effect, pathway, or disease association that the record states
  differently. A reader of the record would conclude the claim is false, not merely unmentioned.

- "ABSENCE": the record does not mention what the claim asserts, and the earlier judgement
  treated that silence as contradiction. The record is incomplete on this point rather than
  in conflict with it. Choose this whenever the record neither states nor implies the opposite.

- "TERMINOLOGY": the record and the claim describe the same underlying biology, and the
  disagreement is one of naming, precision, or granularity rather than of fact.

Judge ONLY from the record above. Do not use outside knowledge to decide whether the claim
is true in general - decide only what the record does and does not establish.

"evidence" MUST be copied VERBATIM from the record for GENUINE and TERMINOLOGY (the span
that conflicts). For ABSENCE, "evidence" must instead name what the record is silent about,
beginning with "record does not mention".

Return ONLY a JSON array with one object:
[{{"category":"GENUINE","evidence":"...","reason":"one sentence"}}]
"""


def load_incorrect():
    src = BASE / "results" / "verify_db_headline_v2.csv"
    if not src.exists(): src = BASE / "results" / "verify_db_headline.csv"
    rows = list(csv.DictReader(open(src, encoding="utf-8")))
    inc = [(i, r) for i, r in enumerate(rows) if r["label"] == "incorrect"]
    return src, rows, inc


def build_requests(inc, cache):
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    reqs, index = [], {}
    for i, r in inc:
        g = r["gene"].split(",")[0]
        rec = cache.get(g) or {}
        prompt = PROMPT.format(gene=g, uniprot=rec.get("uniprot") or "(none)",
                               ncbi=rec.get("ncbi") or "(none)",
                               go=" | ".join(rec.get("go") or []) or "(none)",
                               claim=r["claim"], quote=r["quoted_source_sentence"] or "(none)")
        cid = f"row{i}"
        index[cid] = (i, r, g)
        reqs.append(Request(custom_id=cid, params=MessageCreateParamsNonStreaming(
            model=MODEL, max_tokens=1200, temperature=0,
            messages=[{"role": "user", "content": prompt}])))
    return reqs, index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--collect", default=None)
    args = ap.parse_args()

    src, rows, inc = load_incorrect()
    print(f"source: {src.name} | incorrect verdicts to audit: {len(inc)}")
    cache = json.load(open(CACHE, encoding="utf-8"))

    import anthropic
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key: sys.exit("ERROR: set ANTHROPIC_API_KEY (.\\load_env.ps1)")
    client = anthropic.Anthropic(api_key=key)

    reqs, index = build_requests(inc, cache)
    if args.estimate:
        tot = sum(client.messages.count_tokens(model=MODEL, messages=r["params"]["messages"]).input_tokens
                  for r in reqs[:20])
        avg = tot / min(20, len(reqs))
        est_in, est_out = avg * len(reqs), len(reqs) * 160
        print(f"  avg input {avg:,.0f} tok | est in {est_in:,.0f} out {est_out:,.0f}")
        print(f"  EST. COST ${(est_in/1e6*PRICE_IN + est_out/1e6*PRICE_OUT)*(1-DISC):.2f}")
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
        time.sleep(20)

    out, in_tok, out_tok = [], 0, 0
    cats = {}
    for res in client.messages.batches.results(bid):
        ent = index.get(res.custom_id)
        if ent is None: continue
        i, r, g = ent
        if res.result.type != "succeeded": continue
        msg = res.result.message
        in_tok += msg.usage.input_tokens; out_tok += msg.usage.output_tokens
        txt = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        objs = [o for o in parse_json_array(txt) if isinstance(o, dict)]
        o = objs[0] if objs else {}
        cat = str(o.get("category", "GENUINE")).upper()
        if cat not in ("GENUINE", "ABSENCE", "TERMINOLOGY"): cat = "GENUINE"
        cats[i] = cat
        out.append({"gene_set": r["gene_set"], "gene": r["gene"], "claim": r["claim"],
                    "category": cat, "audit_evidence": o.get("evidence", ""),
                    "audit_reason": o.get("reason", ""),
                    "original_quote": r["quoted_source_sentence"],
                    "source_url": r["source_url"], "judge_label": r["judge_label"]})

    out.sort(key=lambda d: (d["category"], d["gene_set"]))
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(f"wrote {len(out)} adjudications -> {OUT_CSV}")

    c = Counter(d["category"] for d in out)
    correct = sum(1 for r in rows if r["label"] == "correct")
    strict_inc = len(inc)
    genuine = c["GENUINE"]
    # reclassified rows become "correct" only if the claim is right; ABSENCE/TERMINOLOGY are
    # not contradicted by the record, so they move OUT of the numerator but stay in the
    # denominator as correct-under-the-record.
    n_strict = correct + strict_inc
    lo_s, hi_s = wilson(strict_inc, n_strict)
    lo_g, hi_g = wilson(genuine, n_strict)
    print("\n=== ADJUDICATION OF THE INCORRECT VERDICTS ===")
    for k in ("GENUINE", "ABSENCE", "TERMINOLOGY"):
        print(f"  {k:<12}: {c[k]:>3}  ({c[k]/len(out)*100:.1f}%)")
    print(f"\n  UPPER BOUND (all flags)  : {strict_inc}/{n_strict} = {strict_inc/n_strict:.2%} "
          f"CI [{lo_s:.2%}, {hi_s:.2%}]")
    print(f"  CORRECTED (genuine only) : {genuine}/{n_strict} = {genuine/n_strict:.2%} "
          f"CI [{lo_g:.2%}, {hi_g:.2%}]")
    cost = (in_tok/1e6*PRICE_IN + out_tok/1e6*PRICE_OUT)*(1-DISC)
    print(f"\n  spend: ${cost:.2f}")
    json.dump({"batch_id": bid, "cost": cost, "source": src.name,
               "categories": dict(c), "correct": correct,
               "upper_bound": {"k": strict_inc, "n": n_strict, "rate": strict_inc/n_strict,
                               "ci": [lo_s, hi_s]},
               "corrected": {"k": genuine, "n": n_strict, "rate": genuine/n_strict,
                             "ci": [lo_g, hi_g]}}, open(STATS, "w"), indent=2)


if __name__ == "__main__":
    main()
