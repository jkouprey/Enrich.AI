"""
eval/phase1_run.py — PHASE 1 (FREE): run all 50 Hallmark sets through the tool.

Tool = the deployed default (Gemini 2.5 Flash via the app's own ReasoningEngine).
NO judge is called here, so this phase costs nothing beyond the free Gemini tier.

Writes eval/full_run/{set}.json immediately after each set completes, so a crash
or a quota stop never loses finished work. Re-running skips sets already on disk.

Usage:
    .\\load_env.ps1                      # puts GOOGLE_API_KEY in the session
    python eval\\phase1_run.py           # run / resume
    python eval\\phase1_run.py --limit 3 # smoke-test on the first 3 sets
"""
from __future__ import annotations
import argparse, json, sys, time, traceback
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))

# headless: switch off every Streamlit call inside the engine
import reasoning_engine
reasoning_engine.st = None
from reasoning_engine import ReasoningEngine

from evalkit import (QUERY, FULL_RUN_DIR, audit_citations, cleanj, classify_error,
                     enrichment_term_names, load_contexts, load_gene_sets)

MAX_ATTEMPTS = 3          # per set, for transient failures
RETRY_BACKOFF = (10, 30)  # seconds before attempt 2 and 3


def extract_tool_calls(env: dict) -> list:
    """Each tool call with the parameters the agent chose, from reasoning_steps."""
    calls = []
    for step in env.get("reasoning_steps") or []:
        if step.get("type") == "thought_action":
            calls.append({
                "step": step.get("step"),
                "tool": step.get("tool_name") or step.get("action"),
                "args": step.get("args"),
                "thought": step.get("thought"),
                "observation": step.get("observation"),
            })
    return calls


def run_one(engine: ReasoningEngine, name: str, genes: list, context: str) -> dict:
    """One tool run. Raises on failure so the caller can classify the error."""
    engine.previous_envelopes = []          # memory reset per set
    query = QUERY.format(context=context, genes=", ".join(genes))
    env = engine.run(query)["envelope"]

    interp = env.get("final_text", "")
    fer = env.get("full_enrichment_results")
    if isinstance(fer, str):
        try: fer = json.loads(fer)
        except Exception: fer = None
    enr = (fer or {}).get("enrichment_results") or {}
    n_terms = sum(len(v) for v in enr.values() if isinstance(v, list))
    papers = env.get("full_literature_results") or []
    tools = env.get("tools_used", [])

    if "Agent execution failed" in interp or "I encountered an error" in interp:
        raise RuntimeError(interp[:400])

    # A zero-character answer passes both error checks above but is NOT a successful
    # run - the tool produced nothing. Recorded (so it stays auditable) but flagged as
    # a failure so completion and reliability denominators stay honest. Not retried:
    # retrying would measure "persistent empty" rather than the real per-query incidence.
    empty = not (interp or "").strip()

    return {
        "gene_set": name,
        "context": context,
        "genes": genes,
        "n_genes": len(genes),
        "query": query,
        # tool-less and empty runs are saved for audit but excluded from the aggregate
        "valid": bool(tools) and n_terms > 0 and not empty,
        "tool_less": not bool(tools),
        "empty_response": empty,
        "n_evidence_terms": n_terms,
        "n_papers": len(papers),
        # what the engine's post-generation guard stripped (empty = model cited cleanly)
        "citation_guard": env.get("citation_guard"),
        # independent check on the FINAL text the user would see
        "citation_audit": audit_citations(interp, papers, enrichment_term_names(fer)),
        "interpretation": interp,
        "reasoning_trace": env.get("reasoning_trace") or [],
        "reasoning_steps": env.get("reasoning_steps") or [],
        "tools_used": tools,
        "tool_calls": extract_tool_calls(env),
        "full_enrichment_results": cleanj(fer),
        "full_literature_results": cleanj(papers),
        "full_db_results": cleanj(env.get("full_db_results")),
        "gene_info": cleanj(env.get("gene_info")),
        "execution_time": env.get("execution_time"),
        "phase": 1,
    }


def reaudit_saved():
    """Recompute citation_audit on every saved record with the current auditor.

    Needed when the auditor is tightened after some records were already written -
    the raw text and papers are on disk, so no re-run and no API call is required.
    """
    n = changed = 0
    for p in sorted(FULL_RUN_DIR.glob("*.json")):
        if p.name.startswith("_"): continue
        d = json.load(open(p, encoding="utf-8"))
        old = (d.get("citation_audit") or {}).get("has_fabricated_citation")
        fresh = audit_citations(d.get("interpretation", ""), d.get("full_literature_results"),
                                enrichment_term_names(d.get("full_enrichment_results")))
        d["citation_audit"] = fresh
        p.write_text(json.dumps(d, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        n += 1
        if old != fresh["has_fabricated_citation"]:
            changed += 1
            print(f"  {d.get('gene_set')}: fabricated {old} -> {fresh['has_fabricated_citation']}")
    print(f"Re-audited {n} records ({changed} verdict change(s)).")


def reguard_saved():
    """Re-apply the engine's citation guard to saved records (no API calls).

    Used when the guard is tightened after records were written. The text as first
    saved is preserved in `interpretation_pre_guard` so the model's own behaviour
    stays auditable and the guard's effect is measurable.
    """
    from reasoning_engine import verify_citations
    n = cleaned = 0
    for p in sorted(FULL_RUN_DIR.glob("*.json")):
        if p.name.startswith("_"): continue
        d = json.load(open(p, encoding="utf-8"))
        papers = d.get("full_literature_results") or []
        before = d.get("interpretation", "")
        after, rep = verify_citations(before, papers)
        n += 1
        if rep.get("n_removed"):
            cleaned += 1
            d.setdefault("interpretation_pre_guard", before)
            d["interpretation"] = after
            d["citation_guard"] = rep
            print(f"  {d.get('gene_set')}: removed {rep['n_removed']} citation(s) "
                  f"(papers retrieved: {rep['n_papers_retrieved']})")
        d["citation_audit"] = audit_citations(d["interpretation"], papers,
                                              enrichment_term_names(d.get("full_enrichment_results")))
        p.write_text(json.dumps(d, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Re-guarded {n} records; {cleaned} had citations removed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only the first N sets (smoke test)")
    ap.add_argument("--reaudit", action="store_true",
                    help="recompute citation_audit on saved records (no API calls) and exit")
    ap.add_argument("--reguard", action="store_true",
                    help="re-apply the citation guard to saved records (no API calls) and exit")
    args = ap.parse_args()

    FULL_RUN_DIR.mkdir(exist_ok=True)
    if args.reaudit:
        reaudit_saved(); return
    if args.reguard:
        reguard_saved(); return
    gene_sets = load_gene_sets()
    contexts = load_contexts()
    names = list(gene_sets)[:args.limit] if args.limit else list(gene_sets)

    todo = [n for n in names if not (FULL_RUN_DIR / f"{n}.json").exists()]
    done_already = len(names) - len(todo)
    print(f"PHASE 1 (free tool pass) - {len(names)} sets, {done_already} already saved, {len(todo)} to run\n")
    if not todo:
        print("Nothing to do. All sets already saved.")
        return

    engine = ReasoningEngine()
    print(f"Engine: provider={engine.provider} model={engine.model_name}\n")

    completed, skipped, empty_runs, t_start = 0, [], [], time.time()
    for idx, name in enumerate(todo, 1):
        genes = gene_sets[name]
        context = contexts.get(name, name.replace("_", " "))
        print(f"[{idx}/{len(todo)}] {name} ({len(genes)} genes, context: {context!r}) ...", flush=True)

        for attempt in range(1, MAX_ATTEMPTS + 1):
            t0 = time.time()
            try:
                rec = run_one(engine, name, genes, context)
            except Exception as e:
                kind = classify_error(e)
                if kind == "fatal":
                    print(f"\n{'='*70}\nStopping: quota/credit exhausted at set {idx} of {len(todo)} "
                          f"({name}) - {completed} sets completed and saved this run.")
                    print(f"Reason: {type(e).__name__}: {str(e)[:300]}")
                    print(f"All finished work is on disk in {FULL_RUN_DIR.name}/. "
                          f"Re-run this script later to resume.\n{'='*70}")
                    return
                if attempt < MAX_ATTEMPTS:
                    wait = RETRY_BACKOFF[attempt - 1]
                    print(f"    transient failure ({type(e).__name__}) - retry {attempt}/{MAX_ATTEMPTS-1} in {wait}s")
                    time.sleep(wait)
                    continue
                print(f"    FAILED after {MAX_ATTEMPTS} attempts, skipping: {str(e)[:200]}")
                skipped.append({"gene_set": name, "error": f"{type(e).__name__}: {str(e)[:300]}"})
                break

            # save-as-you-go: on disk before the next set starts
            (FULL_RUN_DIR / f"{name}.json").write_text(
                json.dumps(rec, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
            if rec.get("empty_response"):
                empty_runs.append(name)
                print(f"    !! EMPTY RESPONSE (0 chars) - recorded as a FAILURE, not a completed run")
            else:
                completed += 1
            flag = "" if rec["valid"] else ("  [EMPTY-RESPONSE]" if rec.get("empty_response")
                                            else "  [TOOL-LESS - excluded from aggregate]")
            print(f"    ok in {time.time()-t0:.0f}s | tools={rec['tools_used']} | "
                  f"terms={rec['n_evidence_terms']} papers={rec['n_papers']} "
                  f"chars={len(rec['interpretation'])}{flag}")
            ca = rec["citation_audit"]
            if ca["has_fabricated_citation"]:
                bits = []
                if ca["fabricated_pmids"]: bits.append(f"{len(ca['fabricated_pmids'])} PMIDs")
                if ca["fabricated_titles"]: bits.append(f"{len(ca['fabricated_titles'])} titles")
                if ca["fabricated_author_years"]: bits.append(f"{len(ca['fabricated_author_years'])} author-year refs")
                if ca["et_al_with_no_papers"]: bits.append(f"{ca['et_al_with_no_papers']} 'et al.' w/ 0 papers")
                print(f"    !! FABRICATED CITATIONS: {', '.join(bits)}")
            break

    if skipped:
        (FULL_RUN_DIR / "_phase1_skipped.json").write_text(
            json.dumps(skipped, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nPhase 1 complete: {completed} successful, {len(empty_runs)} empty-response "
          f"failure(s), {len(skipped)} skipped, {time.time()-t_start:.0f}s total.")
    if empty_runs:
        print(f"  empty-response sets: {', '.join(empty_runs)}")
    print(f"Results: {FULL_RUN_DIR}/")
    print("No judge was called - this phase cost nothing.")


if __name__ == "__main__":
    main()
