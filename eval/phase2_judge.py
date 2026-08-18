"""
eval/phase2_judge.py — PHASE 2 (PAID): judge the saved interpretations.

Two steps:
  A. DECOMPOSE  - Groq llama-3.3-70b @ temp 0 (FREE). Only sees the interpretation
                  (a few KB), so the truncation problem that disqualified Groq as a
                  judge does not apply here.
  B. JUDGE      - Claude Sonnet 4.5 @ temp 0, via the Anthropic MESSAGE BATCHES API
                  (50% cheaper than synchronous). Sees the big evidence bundle.

Results are merged back into the SAME eval/full_run/{set}.json written by phase 1,
then eval/summary.json is rebuilt.

Cost control:
  * --estimate  prices the whole batch with the free count_tokens endpoint and STOPS.
  * A confirmation prompt shows the estimate before anything is submitted.
  * Actual per-set cost is printed as results come back, with a running total.

Usage:
    .\\load_env.ps1
    python eval\\phase2_judge.py --estimate     # free: show projected cost, submit nothing
    python eval\\phase2_judge.py                # decompose (free) then judge (paid)
    python eval\\phase2_judge.py --limit 5      # only the first 5 unjudged sets
"""
from __future__ import annotations
import argparse, json, os, statistics, sys, time
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))

from evalkit import (DECOMPOSE_PROMPT, JUDGE_PROMPT, FULL_RUN_DIR, SUMMARY_FILE,
                     audit_citations, build_evidence, classify_error, enrichment_term_names,
                     gtext, parse_json_array, tally)

# --- models ---
# Decomposition defaults to Sonnet (paid, batched) so every set in the benchmark is
# decomposed by the same model. Groq is free but its DAILY token cap stops a 50-set
# run halfway. Either way the decomposer is a different family from Gemini, the tool
# under test - which is the independence that matters here.
DECOMPOSE_MODEL_GROQ = "llama-3.3-70b-versatile"     # Groq, free, daily-capped
DECOMPOSE_MODEL_SONNET = "claude-sonnet-4-5-20250929"
JUDGE_MODEL = "claude-sonnet-4-5-20250929"           # Anthropic, paid
DECOMPOSE_MODEL_USED = DECOMPOSE_MODEL_SONNET        # set from --decomposer in main()
RUN_DIR = FULL_RUN_DIR                               # set from --dir in main()
SUMMARY_OUT = SUMMARY_FILE                           # set from --dir in main()

# --- Sonnet 4.5 list price, $/million tokens (docs: $3 in / $15 out) ---
PRICE_IN_PER_MTOK, PRICE_OUT_PER_MTOK = 3.00, 15.00
BATCH_DISCOUNT = 0.50                                 # Batches API = 50% off
# RAGAS protocol: the answer is decomposed in full and EVERY claim is verified.
# These ceilings exist only so a runaway generation can't hang - they must never
# bind in practice, or claims get silently dropped and the ratio drifts.
JUDGE_MAX_TOKENS = 24000                              # ~45 tok/claim verdict
DECOMPOSE_MAX_TOKENS = 12000                          # ~25 tok/claim

POLL_SECONDS = 30
# Groq free tier is 12k tokens/minute and each decomposition costs ~2.7k
# (interpretation in + claims out). Pace to ~4/min so requests don't burn their
# retries on per-minute 429s and get dropped from the judged set.
DECOMPOSE_DELAY = 14
DECOMPOSE_RETRY_BACKOFF = (20, 45)


def money(x): return f"${x:,.4f}"


def batch_cost(in_tok, out_tok):
    return ((in_tok / 1e6) * PRICE_IN_PER_MTOK + (out_tok / 1e6) * PRICE_OUT_PER_MTOK) * (1 - BATCH_DISCOUNT)


def load_records(limit=None, force=False):
    """Phase-1 records to judge. force=True re-judges ones already scored."""
    recs = []
    for p in sorted(RUN_DIR.glob("*.json")):
        if p.name.startswith("_"): continue
        d = json.load(open(p, encoding="utf-8"))
        if not force and d.get("judge_claims") is not None:   # already judged
            continue
        if not (d.get("interpretation") or "").strip():
            continue
        recs.append((p, d))
    return recs[:limit] if limit else recs


def evenly_spaced(recs, n):
    """N records spread across the sorted list - a representative dry-run sample."""
    if n >= len(recs):
        return recs
    step = len(recs) / n
    return [recs[int(i * step)] for i in range(n)]


def cid(path):
    """Batch custom_id. Uses the FILE STEM, not the gene-set name: replicate runs
    share a gene_set ({set}__rep2, __rep3 ...) and duplicate custom_ids are a 400.
    For eval/full_run the stem IS the gene-set name, so existing batches still match.
    """
    return path.stem[:64]


def submit_and_wait(client, requests, label):
    """Submit a Message Batch and poll until it ends. Returns the batch id."""
    batch = client.messages.batches.create(requests=requests)
    print(f"  {label}: batch {batch.id} submitted with {len(requests)} requests")
    while True:
        b = client.messages.batches.retrieve(batch.id)
        if b.processing_status == "ended":
            break
        c = b.request_counts
        print(f"    {b.processing_status}: processing={c.processing} "
              f"succeeded={c.succeeded} errored={c.errored}")
        time.sleep(POLL_SECONDS)
    return batch.id


def decompose_all_sonnet(client, recs, batch_id=None):
    """Decompose every interpretation with Sonnet via the Batches API (50% off).

    batch_id re-reads an ALREADY COMPLETED batch instead of submitting a new one -
    results stay retrievable for 29 days, so a crash after submission is recoverable
    without paying twice.
    """
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    print(f"STEP A - decomposing {len(recs)} interpretations "
          f"({DECOMPOSE_MODEL_SONNET}, batched at {int(BATCH_DISCOUNT*100)}% off)\n")
    by_id = {cid(p): (p, d) for p, d in recs}
    if batch_id:
        print(f"  reusing completed decompose batch {batch_id} (no new charge)")
        bid = batch_id
    else:
        requests = [
            Request(custom_id=cid(p),
                    params=MessageCreateParamsNonStreaming(
                        model=DECOMPOSE_MODEL_SONNET, max_tokens=DECOMPOSE_MAX_TOKENS, temperature=0,
                        messages=[{"role": "user",
                                   "content": DECOMPOSE_PROMPT.format(interpretation=d["interpretation"])}]))
            for p, d in recs
        ]
        bid = submit_and_wait(client, requests, "decompose")

    out, in_tok, out_tok, failed = [], 0, 0, 0
    for result in client.messages.batches.results(bid):
        entry = by_id.get(result.custom_id)
        if entry is None:
            continue
        path, d = entry
        if result.result.type != "succeeded":
            print(f"  {d['gene_set']}: decompose {result.result.type} - skipped")
            failed += 1
            continue
        msg = result.result.message
        in_tok += msg.usage.input_tokens
        out_tok += msg.usage.output_tokens
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        claims = [c for c in parse_json_array(text) if isinstance(c, str) and c.strip()]
        if not claims:
            print(f"  {d['gene_set']}: no claims parsed - skipped")
            failed += 1
            continue
        print(f"  {d['gene_set']:34} {len(claims)} claims")
        out.append((path, d, claims))

    cost = batch_cost(in_tok, out_tok)
    print(f"\n  decomposition done: {len(out)} ok, {failed} failed | "
          f"{in_tok:,} in / {out_tok:,} out | {money(cost)}")
    return out, cost


# ---------------------------------------------------------------- step A: decompose
def decompose_all(recs):
    from langchain_groq import ChatGroq
    if not os.environ.get("GROQ_API_KEY"):
        sys.exit("ERROR: set GROQ_API_KEY first (.\\load_env.ps1)")
    llm = ChatGroq(model=DECOMPOSE_MODEL_GROQ, api_key=os.environ["GROQ_API_KEY"], temperature=0)

    out = []
    print(f"STEP A - decomposing {len(recs)} interpretations on Groq ({DECOMPOSE_MODEL_GROQ}, free)")
    print(f"         paced at ~{60//DECOMPOSE_DELAY}/min to stay inside the 12k TPM free tier "
          f"(~{len(recs)*DECOMPOSE_DELAY//60} min)\n")
    for i, (path, d) in enumerate(recs, 1):
        name = d["gene_set"]
        if i > 1:
            time.sleep(DECOMPOSE_DELAY)
        for attempt in range(1, 4):
            try:
                resp = llm.invoke(DECOMPOSE_PROMPT.format(interpretation=d["interpretation"]))
                claims = [c for c in parse_json_array(gtext(resp)) if isinstance(c, str) and c.strip()]
                if not claims:
                    raise ValueError("no claims parsed")
                print(f"  [{i}/{len(recs)}] {name}: {len(claims)} claims")
                out.append((path, d, claims))
                break
            except Exception as e:
                if classify_error(e) == "fatal":
                    print(f"\nStopping: Groq quota/auth exhausted at {name}. "
                          f"{len(out)} decompositions done; nothing paid yet.")
                    return out, True
                if attempt < 3:
                    time.sleep(DECOMPOSE_RETRY_BACKOFF[attempt - 1]); continue
                print(f"  [{i}/{len(recs)}] {name}: decompose FAILED, skipping ({str(e)[:120]})")
    return out, False


# ---------------------------------------------------------------- step B: judge
def build_prompts(decomposed):
    items = []
    for path, d, claims in decomposed:
        evidence = build_evidence(d.get("full_enrichment_results"), d.get("full_literature_results"))
        numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(claims))
        items.append({
            "path": path, "rec": d, "claims": claims, "evidence": evidence,
            "prompt": JUDGE_PROMPT.format(evidence=evidence, claims=numbered),
        })
    return items


def estimate(client, items):
    """Price the batch up front. count_tokens is free."""
    print(f"\nESTIMATE ({JUDGE_MODEL}, Batches API at {int(BATCH_DISCOUNT*100)}% off)\n")
    total_in = 0
    for it in items:
        n = client.messages.count_tokens(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": it["prompt"]}],
        ).input_tokens
        it["est_in"] = n
        total_in += n
    # output: judges emit ~1 short JSON object per claim; 45 tok/claim is a safe upper bound
    total_out = sum(min(JUDGE_MAX_TOKENS, 45 * len(it["claims"]) + 200) for it in items)
    lo = batch_cost(total_in, total_out * 0.6)
    hi = batch_cost(total_in, total_out)
    print(f"  sets            : {len(items)}")
    print(f"  input tokens    : {total_in:,}")
    print(f"  output tokens   : {total_out:,} (upper bound)")
    print(f"  est. cost       : {money(lo)} - {money(hi)}")
    print(f"  (sync would be  : {money(hi/(1-BATCH_DISCOUNT))})")
    return total_in, total_out, hi


def judge_batch(client, items, batch_id=None):
    """Judge every set. batch_id re-reads an already completed batch (no new charge)."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    if batch_id:
        print(f"\nSTEP B - reusing completed judge batch {batch_id} (no new charge)")
        bid = batch_id
    else:
        requests = [
            Request(
                custom_id=cid(it["path"]),
                params=MessageCreateParamsNonStreaming(
                    model=JUDGE_MODEL,
                    max_tokens=JUDGE_MAX_TOKENS,
                    temperature=0,
                    messages=[{"role": "user", "content": it["prompt"]}],
                ),
            )
            for it in items
        ]
        print("\nSTEP B - judging")
        bid = submit_and_wait(client, requests, "judge")

    by_id = {cid(it["path"]): it for it in items}
    running_in = running_out = 0
    judged = failed = 0

    for result in client.messages.batches.results(bid):
        it = by_id.get(result.custom_id)
        if it is None:
            continue
        name = it["rec"]["gene_set"]
        if result.result.type != "succeeded":
            print(f"  {name}: batch result {result.result.type} - skipped")
            failed += 1
            continue

        msg = result.result.message
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        labels = parse_json_array(text)
        metrics, scored = tally(it["claims"], labels)

        u = msg.usage
        running_in += u.input_tokens
        running_out += u.output_tokens
        cost_so_far = batch_cost(running_in, running_out)

        rec = it["rec"]
        rec["judge"] = {
            "decompose_model": DECOMPOSE_MODEL_USED,
            "judge_model": JUDGE_MODEL,
            "temperature": 0,
            "via": "anthropic_batches_api",
        }
        rec["evidence_bundle"] = it["evidence"]
        rec["judge_claims"] = scored          # claim + label + why
        rec["summary"] = metrics              # counts + hallucination_rate
        rec["phase"] = 2
        it["path"].write_text(json.dumps(rec, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

        judged += 1
        print(f"  {name:34} claims={metrics['n_claims']:>2} scored={metrics['n_scored']:>2} "
              f"G/B/U={metrics['grounded']}/{metrics['background']}/{metrics['unsupported']} "
              f"halluc={metrics['hallucination_rate']:.1%} | "
              f"tok {u.input_tokens:>6}+{u.output_tokens:<5} | running {money(cost_so_far)}")

    print(f"\nJudged {judged}, failed {failed}. "
          f"Actual spend: {money(batch_cost(running_in, running_out))} "
          f"({running_in:,} in / {running_out:,} out)")


# ---------------------------------------------------------------- summary
def build_summary():
    per_set, skipped, toolless = [], [], []
    # behaviour metrics span EVERY phase-1 record, judged or not - a tool-less run is
    # excluded from the hallucination aggregate but must still count here.
    all_records, fabricators, no_tool_runs, empty_runs = 0, [], [], []

    for p in sorted(RUN_DIR.glob("*.json")):
        if p.name.startswith("_"): continue
        d = json.load(open(p, encoding="utf-8"))
        name = d.get("gene_set", p.stem)

        all_records += 1
        # recompute rather than trust a stored audit - keeps the metric current
        ca = audit_citations(d.get("interpretation", ""), d.get("full_literature_results"),
                             enrichment_term_names(d.get("full_enrichment_results")))
        if ca.get("has_fabricated_citation"):
            fabricators.append({
                "gene_set": name,
                "n_papers_retrieved": ca.get("n_papers_retrieved", 0),
                "fabricated_pmids": ca.get("fabricated_pmids", []),
                "fabricated_titles": ca.get("fabricated_titles", []),
                "et_al_with_no_papers": ca.get("et_al_with_no_papers", 0),
            })
        if d.get("tool_less") or not d.get("tools_used"):
            no_tool_runs.append(name)
        # derived from the text, not a stored flag, so records written before the
        # empty-response check was added are still counted correctly
        if not (d.get("interpretation") or "").strip():
            empty_runs.append(name)

        if d.get("summary") is None:
            skipped.append({"gene_set": name, "reason": "not judged"})
            continue
        entry = {"gene_set": name, "n_genes": d.get("n_genes"),
                 "valid": d.get("valid"), "tools_used": d.get("tools_used"),
                 "n_evidence_terms": d.get("n_evidence_terms"),
                 "n_papers": d.get("n_papers"),
                 "fabricated_citation": bool(ca.get("has_fabricated_citation")),
                 **d["summary"]}
        per_set.append(entry)
        if not d.get("valid"):
            toolless.append(name)

    rates = [e["hallucination_rate"] for e in per_set if e["valid"]]
    grounded = [e["grounded_rate"] for e in per_set if e["valid"]]
    summary = {
        "judge_model": JUDGE_MODEL,
        "decompose_model": DECOMPOSE_MODEL_USED,
        "tool_model": "gemini-2.5-flash",
        "n_sets_judged": len(per_set),
        "n_sets_in_aggregate": len(rates),
        "hallucination_rate_mean": round(statistics.mean(rates), 4) if rates else None,
        "hallucination_rate_sd": round(statistics.stdev(rates), 4) if len(rates) > 1 else 0.0,
        "grounded_rate_mean": round(statistics.mean(grounded), 4) if grounded else None,
        # behaviour metrics over ALL phase-1 runs (not just the judged/valid subset)
        "n_runs_total": all_records,
        "tool_less_rate": round(len(no_tool_runs) / all_records, 4) if all_records else None,
        "empty_response_rate": round(len(empty_runs) / all_records, 4) if all_records else None,
        "empty_response_runs": empty_runs,
        "tool_less_sets": no_tool_runs,
        "fabricated_citation_rate": round(len(fabricators) / all_records, 4) if all_records else None,
        "fabricated_citation_sets": fabricators,
        "excluded_toolless_sets": toolless,
        "skipped_sets": skipped,
        "per_set": sorted(per_set, key=lambda e: -e["hallucination_rate"]),
    }
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== SUMMARY -> {SUMMARY_OUT.name} ===")
    if rates:
        print(f"hallucination rate      : {summary['hallucination_rate_mean']:.1%} "
              f"+/- {summary['hallucination_rate_sd']:.1%} (n={len(rates)} valid sets)")
        print(f"grounded rate           : {summary['grounded_rate_mean']:.1%}")
    if all_records:
        print(f"empty-response rate     : {summary['empty_response_rate']:.1%} "
              f"({len(empty_runs)}/{all_records} runs)"
              + (f" -> {', '.join(empty_runs)}" if empty_runs else ""))
        print(f"tool-less rate          : {summary['tool_less_rate']:.1%} "
              f"({len(no_tool_runs)}/{all_records} runs)")
        print(f"fabricated-citation rate: {summary['fabricated_citation_rate']:.1%} "
              f"({len(fabricators)}/{all_records} runs)")
    for f in fabricators:
        print(f"   !! {f['gene_set']}: {len(f['fabricated_pmids'])} PMIDs, "
              f"{len(f['fabricated_titles'])} titles, {f['n_papers_retrieved']} papers retrieved")
    if toolless:
        print(f"excluded (tool-less)    : {len(toolless)} -> {', '.join(toolless)}")
    if skipped:
        print(f"not judged              : {len(skipped)}")


def main():
    global DECOMPOSE_MODEL_USED, RUN_DIR, SUMMARY_OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimate", action="store_true",
                    help="price the JUDGE batch and stop. Note: with --decomposer sonnet the "
                         "decomposition batch has already run and been paid for by this point; "
                         "only --decomposer groq makes this fully free.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    ap.add_argument("--summary-only", action="store_true", help="just rebuild summary.json")
    ap.add_argument("--force", action="store_true",
                    help="re-judge sets that already have judge output (needed when the decomposer changes)")
    ap.add_argument("--decomposer", choices=["sonnet", "groq"], default="sonnet",
                    help="sonnet = paid but batched and not daily-capped; groq = free but caps out mid-run")
    ap.add_argument("--decompose-batch", default=None,
                    help="reuse a completed decompose batch id instead of submitting a new one (no charge)")
    ap.add_argument("--judge-batch", default=None,
                    help="reuse a completed judge batch id instead of submitting a new one (no charge)")
    ap.add_argument("--dir", default=None,
                    help="directory of phase-1 records to judge (default eval/full_run)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="judge only these gene sets (by name) - for testing a criterion change")
    ap.add_argument("--sample", type=int, default=None,
                    help="dry run: decompose N evenly-spaced sets, report the claim-count "
                         "distribution and projected full cost, then stop before judging")
    args = ap.parse_args()

    # must happen before RUN_DIR is read anywhere below
    if args.dir:
        RUN_DIR = Path(args.dir)
        SUMMARY_OUT = BASE / "results" / f"summary_{RUN_DIR.name}.json"
        print(f"records dir: {RUN_DIR}  ->  summary: {SUMMARY_OUT.name}\n")

    if args.summary_only:
        build_summary(); return

    if not RUN_DIR.exists():
        sys.exit("No eval/full_run/ - run phase1_run.py first.")
    recs = load_records(args.limit, force=args.force)
    if args.only:
        wanted = set(args.only)
        recs = [(p, d) for p, d in recs if d["gene_set"] in wanted]
        print(f"--only: {len(recs)} record(s) for {sorted(wanted)}\n")
    n_total_sets = len(recs)
    if args.sample:
        recs = evenly_spaced(recs, args.sample)
        print(f"DRY RUN: decomposing {len(recs)} of {n_total_sets} sets to size the full run\n")
    if not recs:
        print("Nothing to judge (every saved set already has judge output).")
        build_summary(); return

    DECOMPOSE_MODEL_USED = (DECOMPOSE_MODEL_SONNET if args.decomposer == "sonnet"
                            else DECOMPOSE_MODEL_GROQ)

    import anthropic
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ERROR: set ANTHROPIC_API_KEY first (.\\load_env.ps1)")
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    if args.decomposer == "sonnet":
        decomposed, decompose_spend = decompose_all_sonnet(client, recs,
                                                           batch_id=args.decompose_batch)
        stopped = False
    else:
        decomposed, stopped = decompose_all(recs)
        decompose_spend = 0.0
    if not decomposed:
        sys.exit("No claims decomposed - nothing to judge.")
    items = build_prompts(decomposed)

    if args.sample:
        import statistics as _st
        counts = sorted(len(c) for _, _, c in decomposed)
        judge_in, _, _ = estimate(client, items)
        per_set_in = judge_in / len(items)
        per_set_out = sum(45 * len(c) + 200 for _, _, c in decomposed) / len(items)
        scale = n_total_sets
        proj_decompose = decompose_spend / len(items) * scale
        proj_judge = batch_cost(per_set_in * scale, per_set_out * scale)
        print(f"\n=== DRY RUN: claim-count distribution (uncapped, n={len(counts)}) ===")
        print(f"  counts     : {counts}")
        print(f"  median {_st.median(counts):.0f} | mean {_st.mean(counts):.1f} | "
              f"min {min(counts)} | max {max(counts)}")
        for t in (60, 80, 100):
            print(f"  sets > {t:>3} claims : {sum(1 for c in counts if c > t)}/{len(counts)}")
        print(f"\n=== PROJECTED FULL RUN ({scale} sets) ===")
        print(f"  claims (est)   : {_st.mean(counts) * scale:,.0f}")
        print(f"  decomposition  : {money(proj_decompose)}")
        print(f"  judging        : {money(proj_judge)}")
        print(f"  TOTAL (est)    : {money(proj_decompose + proj_judge)}")
        print(f"  already spent on this sample: {money(decompose_spend)}")
        print("\n--sample: stopping before the judge batch.")
        return

    if args.judge_batch:
        est_hi = 0.0                       # reusing a completed batch: nothing new to price
    else:
        _, _, est_hi = estimate(client, items)
    if args.estimate:
        print("\n--estimate: stopping before the judge batch.")
        return
    if decompose_spend:
        print(f"  decomposition already spent : {money(decompose_spend)}")
        print(f"  projected PHASE 2 TOTAL     : {money(decompose_spend + est_hi)}")
    if not args.yes:
        ans = input(f"\nSubmit {len(items)} sets to the {JUDGE_MODEL} batch judge "
                    f"(up to ~{money(est_hi)})? [y/N] ").strip().lower()
        if ans != "y":
            print("Aborted. Nothing submitted, nothing charged.")
            return

    judge_batch(client, items, batch_id=args.judge_batch)
    build_summary()
    if stopped:
        print("\nNOTE: decomposition stopped early on a Groq quota error - "
              "re-run to judge the remaining sets.")


if __name__ == "__main__":
    main()
