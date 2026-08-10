"""
eval/phase3_reps.py — STEP 1: replicate runs for within-set variance.

Runs the SAME pinned query through the tool N extra times for a stratified set of
gene sets, so we can measure how much the hallucination rate moves run-to-run.
Rep 1 is the existing record in eval/full_run/; this script produces reps 2..N.

Free (Gemini). Save-as-you-go, resume, quota-aware - same policy as phase 1.

Usage:
    .\\load_env.ps1
    python eval\\phase3_reps.py --reps 5          # produces reps 2..5
    python eval\\phase3_reps.py --reps 5 --sets Adipogenesis Complement
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))
sys.path.insert(0, str(BASE))

import reasoning_engine
reasoning_engine.st = None
from reasoning_engine import ReasoningEngine

from evalkit import QUERY, FULL_RUN_DIR, classify_error, load_contexts, load_gene_sets
from phase1_run import run_one, MAX_ATTEMPTS, RETRY_BACKOFF

REPS_DIR = BASE / "runs" / "reps"
SETS_FILE = BASE / "data" / "_triplicate_sets.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5, help="total reps INCLUDING the existing run (rep 1)")
    ap.add_argument("--sets", nargs="*", default=None)
    args = ap.parse_args()

    REPS_DIR.mkdir(exist_ok=True)
    names = args.sets or json.load(open(SETS_FILE, encoding="utf-8"))
    gene_sets = load_gene_sets()
    contexts = load_contexts()

    # rep 1 = the existing 50-set record; copy it in so every rep lives together
    for name in names:
        src = FULL_RUN_DIR / f"{name}.json"
        dst = REPS_DIR / f"{name}__rep1.json"
        if src.exists() and not dst.exists():
            d = json.load(open(src, encoding="utf-8"))
            d["rep"] = 1
            dst.write_text(json.dumps(d, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    todo = [(n, r) for r in range(2, args.reps + 1) for n in names
            if not (REPS_DIR / f"{n}__rep{r}.json").exists()]
    print(f"STEP 1 - replicates: {len(names)} sets x reps 2..{args.reps} = {len(todo)} runs to go\n")
    if not todo:
        print("Nothing to do."); return

    engine = ReasoningEngine()
    print(f"Engine: provider={engine.provider} model={engine.model_name}\n")

    done, skipped, empty_runs, t0 = 0, [], [], time.time()
    for i, (name, rep) in enumerate(todo, 1):
        genes = gene_sets[name]
        ctx = contexts.get(name, name.replace("_", " "))
        print(f"[{i}/{len(todo)}] {name} rep{rep} ({len(genes)} genes) ...", flush=True)
        for attempt in range(1, MAX_ATTEMPTS + 1):
            t = time.time()
            try:
                rec = run_one(engine, name, genes, ctx)
            except Exception as e:
                if classify_error(e) == "fatal":
                    print(f"\nStopping: quota/credit exhausted at {name} rep{rep} "
                          f"({done} runs completed and saved). Re-run to resume.")
                    return
                if attempt < MAX_ATTEMPTS:
                    time.sleep(RETRY_BACKOFF[attempt - 1]); continue
                print(f"    FAILED, skipping: {str(e)[:150]}")
                skipped.append({"gene_set": name, "rep": rep, "error": str(e)[:300]})
                break
            rec["rep"] = rep
            (REPS_DIR / f"{name}__rep{rep}.json").write_text(
                json.dumps(rec, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
            if rec.get("empty_response"):
                empty_runs.append(f"{name} rep{rep}")
            else:
                done += 1
            ca = rec["citation_audit"]
            flag = ("  [EMPTY-RESPONSE]" if rec.get("empty_response")
                    else ("" if rec["valid"] else "  [TOOL-LESS]"))
            fab = "  !!FABRICATED" if ca["has_fabricated_citation"] else ""
            print(f"    ok in {time.time()-t:.0f}s | tools={len(rec['tools_used'])} "
                  f"terms={rec['n_evidence_terms']} papers={rec['n_papers']}{flag}{fab}")
            break

    if skipped:
        (REPS_DIR / "_skipped.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")
    print(f"\nDone: {done} successful, {len(empty_runs)} empty-response failure(s), "
          f"{len(skipped)} skipped, {time.time()-t0:.0f}s")
    if empty_runs:
        print(f"  empty-response runs: {', '.join(empty_runs)}")
    print(f"Results: {REPS_DIR}/  (judge them with: phase2_judge.py --dir eval/reps)")


if __name__ == "__main__":
    main()
