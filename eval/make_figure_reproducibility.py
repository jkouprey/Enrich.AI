"""
eval/make_figure_reproducibility.py — publication figure for the reproducibility result.

Shows that a per-set UNSUPPORTED rate - evidence faithfulness, NOT the database-verified
factual error rate - is not a stable quantity across
replicate runs of the identical prompt, while the 50-set aggregate is.

Reads the actual replicate records in eval/reps/ (10 stratified sets x 5 reps) and the
50-set aggregate in eval/summary.json. Nothing is hardcoded from the write-up.

Usage:
    python eval\\make_figure_reproducibility.py
Outputs:
    eval/figure_reproducibility.pdf   (vector, for the manuscript)
    eval/figure_reproducibility.png   (600 dpi, for slides/preview)
    eval/figure_reproducibility_data.csv  (exact numbers plotted, for the caption)
"""
from __future__ import annotations
import csv, glob, json, math, os, statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = Path(__file__).resolve().parent
SCORED = ("GROUNDED", "BACKGROUND", "UNSUPPORTED")

# Okabe-Ito colourblind-safe palette
BLUE, ORANGE, GREY, DARK = "#0072B2", "#D55E00", "#9A9A9A", "#333333"


def load_replicates():
    per = defaultdict(list)
    excluded = []
    for f in sorted(glob.glob(str(BASE / "runs" / "reps" / "*.json"))):
        if os.path.basename(f).startswith("_"):
            continue
        rec = json.load(open(f, encoding="utf-8"))
        claims = rec.get("judge_claims") or []
        n = sum(1 for c in claims if c["label"] in SCORED)
        u = sum(1 for c in claims if c["label"] == "UNSUPPORTED")
        # same inclusion rule as the reported table: a run with no retrieved evidence
        # has nothing to be unfaithful to, so it is not scored
        if not n or not rec.get("valid", True):
            excluded.append((rec["gene_set"], rec.get("rep", "?"),
                             "empty" if rec.get("empty_response") else "tool-less"))
            continue
        per[rec["gene_set"]].append((rec.get("rep", 0), u / n * 100, u, n))
    return per, excluded


def main():
    per, excluded = load_replicates()
    summ = json.load(open(BASE / "results" / "summary.json", encoding="utf-8"))
    # NB: the JSON key is named "hallucination_rate_mean" for historical reasons, but the
    # value is the UNSUPPORTED fraction, i.e. evidence faithfulness - the same quantity
    # plotted per replicate here. It is NOT the database-verified factual error rate.
    agg_mean = summ["hallucination_rate_mean"] * 100
    agg_sd = summ["hallucination_rate_sd"] * 100
    n_sets = summ["n_sets_in_aggregate"]
    sem = agg_sd / math.sqrt(n_sets)
    agg_lo, agg_hi = agg_mean - 1.96 * sem, agg_mean + 1.96 * sem

    order = sorted(per, key=lambda s: st.mean(v[1] for v in per[s]))

    # ---- verification printout -------------------------------------------------
    print("PER-SET REPLICATE DATA (exact values plotted)\n")
    print(f"{'set':<28}{'n':>3}{'mean%':>8}{'SD':>7}{'min':>7}{'max':>7}   replicate values")
    print("-" * 100)
    rows, sds = [], []
    for s in order:
        v = sorted(per[s])
        vals = [x[1] for x in v]
        sd = st.stdev(vals) if len(vals) > 1 else 0.0
        sds.append(sd)
        print(f"{s[:26]:<28}{len(vals):>3}{st.mean(vals):>8.1f}{sd:>7.1f}"
              f"{min(vals):>7.1f}{max(vals):>7.1f}   "
              + ", ".join(f"{x:.1f}" for x in vals))
        for rep, rate, u, n in v:
            rows.append({"gene_set": s, "rep": rep, "unsupported_pct": round(rate, 4),
                         "n_unsupported": u, "n_scored_claims": n})
    flip = sum(1 for s in per if min(x[1] for x in per[s]) == 0 and max(x[1] for x in per[s]) > 0)
    print("-" * 100)
    print(f"mean within-set SD           : {st.mean(sds):.1f} pts")
    print(f"sets flipping 0 <-> non-zero : {flip}/{len(per)}")
    print(f"widest single set            : "
          f"{max(per, key=lambda s: max(x[1] for x in per[s]) - min(x[1] for x in per[s]))}")
    print(f"\n50-set aggregate FAITHFULNESS: {agg_mean:.1f}% unsupported (SD {agg_sd:.1f} across "
          f"{n_sets} sets), 95% CI of the mean [{agg_lo:.2f}, {agg_hi:.2f}]")
    print("  (this is evidence faithfulness, NOT the database-verified factual error rate)")
    if excluded:
        print(f"\nreplicates not scored ({len(excluded)}):")
        for s, r, why in excluded:
            print(f"   {s} rep{r}  ({why})")

    with open(BASE / "results" / "figure_reproducibility_data.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # ---- figure ----------------------------------------------------------------
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "pdf.fonttype": 42, "ps.fonttype": 42,          # editable text in the PDF
        "figure.facecolor": "white", "axes.facecolor": "white",
    })

    fig, ax = plt.subplots(figsize=(6.8, 3.9))

    # aggregate reference band (drawn first, sits behind the data)
    ax.axvspan(agg_lo, agg_hi, color=ORANGE, alpha=0.16, lw=0, zorder=0)
    ax.axvline(agg_mean, color=ORANGE, lw=1.1, zorder=1)

    for i, s in enumerate(order):
        vals = [x[1] for x in sorted(per[s])]
        lo, hi, mu = min(vals), max(vals), st.mean(vals)
        # range spanned by the replicates
        ax.plot([lo, hi], [i, i], color=GREY, lw=1.1, solid_capstyle="round", zorder=2)
        # individual replicate observations, deterministically offset so ties stay visible
        seen = {}
        for v in vals:
            k = round(v, 2)
            seen[k] = seen.get(k, 0) + 1
            off = (seen[k] - 1) * 0.115
            ax.plot(v, i + off, "o", ms=4.0, mfc=BLUE, mec="white", mew=0.55,
                    alpha=0.95, zorder=4)
        # mean of the replicates
        ax.plot(mu, i, "|", ms=11, color=DARK, mew=1.5, zorder=5)

    ax.set_yticks(range(len(order)))
    # every set had 5 replicate runs; mark the two where a run was tool-less and
    # therefore not scorable, so the reader is not told "5 runs" for a set showing 3
    ax.set_yticklabels([s.replace("_", " ") + ("" if len(per[s]) == 5 else f"  (n={len(per[s])})")
                        for s in order])
    ax.set_ylim(-0.7, len(order) - 0.25)
    ax.set_xlim(-0.8, 29)
    ax.set_xlabel("Unsupported claims per run (%)")
    # The quantity plotted is evidence faithfulness (claims not supported by the run's OWN
    # retrieved evidence). It is NOT the database-verified factual error rate, which is a
    # different measurement against external curated records - keep the two distinct.
    ax.set_title("Per-set evidence faithfulness is not reproducible across identical runs",
                 pad=26, loc="left", fontweight="bold")
    ax.text(0, 1.012, "Claims unsupported by the run's own retrieved evidence "
                      "— not the database-verified error rate",
            transform=ax.transAxes, fontsize=7.4, color="#555555", va="bottom")

    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#E3E3E3", lw=0.6, zorder=-1)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)

    handles = [
        Line2D([], [], marker="o", ls="none", ms=4.0, mfc=BLUE, mec="white", mew=0.55,
               label="individual replicate run (5 per set unless noted)"),
        Line2D([], [], color=GREY, lw=1.1, label="observed range across replicates"),
        Line2D([], [], marker="|", ls="none", ms=10, color=DARK, mew=1.5,
               label="per-set mean"),
        Line2D([], [], color=ORANGE, lw=1.1,
               label=f"50-set aggregate faithfulness ({agg_mean:.1f}% unsupported), 95% CI"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, handlelength=1.5,
              borderaxespad=0.4, labelspacing=0.42)

    fig.tight_layout(pad=0.6)
    fig.savefig(BASE / "results" / "figure_reproducibility.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(BASE / "results" / "figure_reproducibility.png", dpi=600, bbox_inches="tight")
    print(f"\nwrote figure_reproducibility.pdf / .png / _data.csv -> {BASE}")


if __name__ == "__main__":
    main()
