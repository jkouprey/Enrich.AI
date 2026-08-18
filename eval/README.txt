================================================================================
ENRICH.AI EVALUATION - START HERE
================================================================================

WANT THE ANSWER?           read  EVALUATION_FOR_PAPER.txt
WANT TO CHECK A CLAIM?     open  results/ALL_CLAIMS.csv
WANT TO RE-RUN IT?         read  HOW_THE_EVAL_CODE_WORKS.txt

--------------------------------------------------------------------------------
FOLDER LAYOUT
--------------------------------------------------------------------------------
eval/
  *.txt          the three documents (this file, the paper text, the code guide)
  *.py           the pipeline scripts, listed in run order further down
  data/          inputs and caches - the gene sets and the fetched database records
  runs/          raw per-run records: full_run/, reps/, _consistency/
  results/       everything derived: CSVs, summary JSONs, the figure
  logs/          console output from each stage, for provenance
../old/          superseded scripts and data. Nothing current reads it.

Rule of thumb: runs/ is evidence, results/ is what you cite, data/ is input,
old/ is history. If a file is not in one of those, it is a document or a script.

--------------------------------------------------------------------------------
THE HEADLINE
--------------------------------------------------------------------------------
50 MSigDB Hallmark gene sets -> Enrich.AI (Gemini 2.5 Flash) -> 4,737 atomic claims
-> judged by Claude Sonnet 4.5 against (a) the tool's own retrieved evidence and
(b) curated databases the tool never saw.

  Database-checkable claims confirmed correct   1,961 / 2,047 = 95.8%
  Genuine factual errors                        54 / 2,047 = 2.64%  CI [2.03, 3.43]
    strict upper bound (every flag counted)     86 / 2,047 = 4.20%  CI [3.41, 5.16]
    ~1 error per interpretation (94.7 claims each)
  Claims grounded in evidence OR true biology   4,628 / 4,737 = 97.7%
    unsupported by the run's own evidence       109 / 4,737 = 2.30%
  Fabricated citations                          0 / 90 runs
  Per-set reproducibility               +/- 3.4 pts SD - per-set rates are NOT stable
  Judge label reproducibility           98.0% on identical inputs (temp 0)
  Total API spend                       $13.81

Two DIFFERENT measurements, do not conflate them. The 2.64% / 4.20% pair is
FACTUAL CORRECTNESS, judged against external curated databases. The 2.30%
unsupported figure is EVIDENCE FAITHFULNESS - whether the tool stuck to what it
actually retrieved. A claim can be faithful and false, or unfaithful and true.

All of these are conservative: ground truth is external curated text rather than
model recall, the verifier may not use its own knowledge, and claims no database
could settle are excluded from the denominator instead of being credited as
correct. Counting those as correct would report 54/4,737 = 1.14%.

--------------------------------------------------------------------------------
THE THREE DOCUMENTS
--------------------------------------------------------------------------------
EVALUATION_FOR_PAPER.txt     Methods, all metrics, the 50-set results table, the
                             reproducibility table, limitations, conclusion.
                             This is the manuscript text.

HOW_THE_EVAL_CODE_WORKS.txt  What every script does, what it reads and writes, why
                             each design decision was made, the failures we hit and
                             how they were fixed, and the commands to reproduce
                             everything from scratch.

README.txt                   This file - the map.

--------------------------------------------------------------------------------
results/ - WHAT YOU CAN OPEN AND CHECK
--------------------------------------------------------------------------------
ALL_CLAIMS.csv               *** THE MASTER FILE *** All 4,737 claims, one per row:
                             the claim, which genes were checked, its evidence label
                             (GROUNDED/BACKGROUND/UNSUPPORTED) with the judge's
                             reasoning, its database label (correct/incorrect/
                             uncertain) with the verbatim source quote and a link to
                             UniProt or NCBI, and - for the 86 flagged claims - BOTH
                             adjudications (Sonnet's and the manual one) with reasons.
                             Column "counts_as_hallucination" is the bottom line:
                               YES                       -> one of the 54 genuine errors
                               flagged-but-not-an-error  -> one of the 32 artifacts
                               (blank)                   -> not flagged

manual_review_sample.csv     15 rows stratified across correct/incorrect/uncertain
                             for hand review, each carrying the FULL UniProt, NCBI
                             and GO text so the quote can be checked in context.

recheck_incorrect_adjudicated.csv
                             The 86 flags with Sonnet's category + reason and the
                             manual category + reason side by side. This is where to
                             argue with the 54-vs-14 disagreement, claim by claim.

verify_db_headline_v2.csv    The database verdicts behind the headline rate.
                             v2 = after the multi-gene fix; supersedes
                             verify_db_headline.csv, which is retained only because
                             it is the INPUT that verify_multigene.py reads.

verify_multigene_detail.csv  Per-gene verdicts for the 88 multi-gene claims, before
                             the aggregation rule is applied in code.

figure_reproducibility.pdf   The reproducibility figure (vector, for the manuscript)
figure_reproducibility.png   600 dpi raster for slides
figure_reproducibility_data.csv   The exact 47 points plotted, for the caption

summary.json                 Per-set and aggregate faithfulness counts, 50 sets.
summary_reps.json            The same for the 50 replicate runs.
summary__consistency.json    The same for the 8-set judge-consistency re-run.
                             CAUTION: in all three, the key "hallucination_rate_mean"
                             holds the UNSUPPORTED (faithfulness) fraction, not the
                             database-verified error rate. Legacy name.
verify_multigene_stats.json  Before/after rates for the multi-gene fix.
recheck_incorrect_stats.json Sonnet's adjudication counts.
final_rates.json             The two headline numbers.

--------------------------------------------------------------------------------
runs/ - THE RAW EVIDENCE
--------------------------------------------------------------------------------
full_run/                    50 files, one per Hallmark set. Each holds the gene
                             list, the exact query, the full reasoning trace, every
                             tool call WITH ITS ARGUMENTS, the complete tool outputs,
                             the final interpretation, the citation audit, and the
                             judged claims. This is the primary record - everything
                             else is derived from it. (_*.txt inside are run logs.)

reps/                        50 files: 10 stratified sets x 5 replicates, for the
                             reproducibility analysis. All on the final judge:
                             reps 2-5 were re-judged, rep1 is a copy of full_run.

_consistency/                8 full_run sets re-judged a second time with identical
                             claims. The evidence for the 98.0% judge reproducibility
                             figure. Not part of the reported results.

--------------------------------------------------------------------------------
data/ - INPUTS AND CACHES
--------------------------------------------------------------------------------
msigdb_sets.json             The 50 Hallmark gene sets. The benchmark input.
_gene_records.json           Cache of every UniProt/NCBI/GO record fetched (2.8 MB).
                             Keep it: every verification step replays offline from
                             this file. Delete it and re-fetching takes ~30 minutes.
_triplicate_sets.json        Which 10 sets were replicated.
steps.txt                    The original evaluation brief.

--------------------------------------------------------------------------------
THE CODE - run in this order (details in HOW_THE_EVAL_CODE_WORKS.txt)
--------------------------------------------------------------------------------
evalkit.py                   Shared constants and prompts. THE single source of
                             truth - the pinned query, the cutoffs, the judge
                             prompt, the citation auditor. Change nothing here
                             without re-running everything downstream.

fetch_msigdb.py              1. download the 50 Hallmark sets  -> data/msigdb_sets.json
phase1_run.py                2. run them through the tool      -> runs/full_run/     (free)
phase2_judge.py              3. decompose + judge              -> results/summary.json (paid)
phase3_reps.py               4. replicate runs                 -> runs/reps/         (free)
verify_db.py                 5. claim vs database              -> results/verify_db_headline.csv (paid)
verify_multigene.py          6. fix multi-gene claims          -> ..._v2.csv         (paid)
recheck_incorrect.py         7. adjudicate the 86 flags        -> results/recheck_incorrect.csv (paid)
make_figure_reproducibility.py   the figure                    -> results/figure_*   (free)

verify_claims.py             (helper) export claims + records for manual review
compare_judge_versions.py    (audit)  prove a re-judge changed / did not change labels

Every paid step has a free --estimate that prices it via count_tokens first, and
every paid step can be recovered from its batch id without paying twice.

--------------------------------------------------------------------------------
../old/ - NOT USED BY ANYTHING CURRENT
--------------------------------------------------------------------------------
Superseded first-attempt scripts, the results of a stricter GROUNDED criterion that
was tried and reverted, and the pre-re-judge backup of the replicates. See
../old/README.txt. No reported number depends on anything in there.
================================================================================
