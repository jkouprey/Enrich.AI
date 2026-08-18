"""
eval/compare_judge_versions.py — did re-judging with the final judge change anything?

Two independent comparisons, both on BYTE-IDENTICAL claims (the decomposition batch was
reused), so every label difference is attributable to the judge alone:

  A. REPLICATES   eval/reps_backup_prejudge/ (as judged 14:17) vs eval/reps/ (re-judged)
  B. CONSISTENCY  eval/full_run/ vs eval/_consistency/ (same records, judged twice with
                  the CURRENT code). Stratified into the sets last judged in the 11:49
                  pass and those re-judged in the 16:45 pass, so a systematic difference
                  between the two eras shows up separately from ordinary judge noise.

Random label noise moves symmetrically between buckets. A one-way drift out of
UNSUPPORTED is the signature of a criterion or evidence change, not noise.

Usage:  python eval\\compare_judge_versions.py
"""
from __future__ import annotations
import glob, json, os, statistics as st
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
SCORED = ("GROUNDED", "BACKGROUND", "UNSUPPORTED")
REJUDGED_1645 = {"Mitotic_Spindle", "Pancreas_Beta_Cells", "Pperoxisome",
                 "Reactive_Oxygen_Species_Pathway", "p53_Pathway"}


def load(d, key="gene_set"):
    out = {}
    for f in sorted(glob.glob(str(d / "*.json"))):
        if os.path.basename(f).startswith("_"):
            continue
        r = json.load(open(f, encoding="utf-8"))
        out[Path(f).stem] = r
    return out


def rate(rec):
    c = rec.get("judge_claims") or []
    n = sum(1 for x in c if x["label"] in SCORED)
    u = sum(1 for x in c if x["label"] == "UNSUPPORTED")
    return u, n, (u / n * 100 if n else None)


def agreement(a, b):
    la = {x["claim"]: x["label"] for x in (a.get("judge_claims") or [])}
    lb = {x["claim"]: x["label"] for x in (b.get("judge_claims") or [])}
    shared = set(la) & set(lb)
    same = sum(1 for k in shared if la[k] == lb[k])
    shifts = Counter((la[k], lb[k]) for k in shared if la[k] != lb[k])
    return same, len(shared), shifts


def direction(shifts):
    out_of = sum(n for (a, b), n in shifts.items() if a == "UNSUPPORTED")
    into = sum(n for (a, b), n in shifts.items() if b == "UNSUPPORTED")
    return out_of, into


def per_set_stats(recs):
    per = defaultdict(list)
    for stem, r in recs.items():
        if not r.get("valid", True):
            continue
        u, n, pct = rate(r)
        if pct is not None:
            per[r["gene_set"]].append(pct)
    sds, flips = [], 0
    for s, v in per.items():
        sds.append(st.stdev(v) if len(v) > 1 else 0.0)
        if min(v) == 0 and max(v) > 0:
            flips += 1
    return per, st.mean(sds) if sds else 0.0, flips


def main():
    # ---------- A. replicates ----------
    old = load(BASE.parent / "old" / "eval" / "reps_backup_prejudge")
    new = load(BASE / "runs" / "reps")
    print("=" * 78)
    print("A. REPLICATES - old judge (14:17) vs final judge (re-judged)")
    print("=" * 78)
    tot_same = tot_n = 0
    all_shifts = Counter()
    changed = []
    for stem in sorted(set(old) & set(new)):
        if stem.endswith("__rep1"):
            continue                       # rep1 is a copy of full_run, not re-judged here
        same, n, shifts = agreement(old[stem], new[stem])
        tot_same += same; tot_n += n; all_shifts += shifts
        _, _, ro = rate(old[stem]); _, _, rn = rate(new[stem])
        if ro is not None and rn is not None and abs(ro - rn) >= 0.05:
            changed.append((stem, ro, rn))
    print(f"  claims compared (reps 2-5) : {tot_n:,}")
    print(f"  label agreement            : {tot_same:,}/{tot_n:,} = {tot_same/tot_n*100:.1f}%")
    out_of, into = direction(all_shifts)
    print(f"  moved OUT of UNSUPPORTED   : {out_of}")
    print(f"  moved INTO UNSUPPORTED     : {into}")
    print(f"  verdict                    : "
          f"{'ONE-WAY DRIFT (criterion/evidence change)' if out_of > 3*max(into,1) else 'symmetric - consistent with noise'}")
    print(f"  shift breakdown            : {dict(all_shifts)}")

    print(f"\n  runs whose rate changed ({len(changed)}):")
    for stem, ro, rn in sorted(changed, key=lambda x: -(abs(x[1] - x[2])))[:12]:
        print(f"    {stem:<38} {ro:>5.1f}%  ->  {rn:>5.1f}%")

    po, sdo, fo = per_set_stats(old)
    pn, sdn, fn = per_set_stats(new)
    print(f"\n  {'set':<28}{'OLD mean':>10}{'OLD SD':>9}   {'NEW mean':>10}{'NEW SD':>9}")
    print("  " + "-" * 68)
    for s in sorted(pn, key=lambda s: st.mean(pn[s])):
        o, n_ = po.get(s, []), pn[s]
        print(f"  {s[:26]:<28}{st.mean(o):>9.1f}%{(st.stdev(o) if len(o)>1 else 0):>9.1f}   "
              f"{st.mean(n_):>9.1f}%{(st.stdev(n_) if len(n_)>1 else 0):>9.1f}")
    print("  " + "-" * 68)
    print(f"  mean within-set SD   : {sdo:.1f} pts  ->  {sdn:.1f} pts")
    print(f"  sets flipping 0<->+  : {fo}/{len(po)}  ->  {fn}/{len(pn)}")

    # ---------- B. consistency ----------
    cons = BASE / "runs" / "_consistency"
    if not cons.exists():
        return
    fr = load(BASE / "runs" / "full_run")
    cc = load(cons)
    print("\n" + "=" * 78)
    print("B. CONSISTENCY - full_run vs the SAME records re-judged with the current code")
    print("=" * 78)
    groups = {"re-judged 16:45 (final judge)": [], "judged 11:49 (earlier pass)": []}
    for stem, r in cc.items():
        g = ("re-judged 16:45 (final judge)" if r["gene_set"] in REJUDGED_1645
             else "judged 11:49 (earlier pass)")
        groups[g].append(stem)
    for g, stems in groups.items():
        gs = gn = 0; gshift = Counter()
        print(f"\n  {g}")
        for stem in sorted(stems):
            same, n, shifts = agreement(fr[stem], cc[stem])
            gs += same; gn += n; gshift += shifts
            _, _, a = rate(fr[stem]); _, _, b = rate(cc[stem])
            print(f"    {stem:<34} agree {same:>4}/{n:<4} = {same/n*100:>5.1f}%   "
                  f"rate {a:>5.1f}% -> {b:>5.1f}%")
        o, i = direction(gshift)
        print(f"    {'GROUP':<34} agree {gs:>4}/{gn:<4} = {gs/gn*100:>5.1f}%   "
              f"out of UNSUP {o}, into {i}")


if __name__ == "__main__":
    main()
