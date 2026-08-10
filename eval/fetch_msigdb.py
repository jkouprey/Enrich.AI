"""
eval/fetch_msigdb.py — pull the FULL MSigDB Hallmark collection (50 sets) from
Enrichr's MSigDB_Hallmark_2020 library. This is the standard, defensible benchmark
for gene-set interpretation.
Output: eval/msigdb_sets.json
"""
import json, sys, urllib.request
from pathlib import Path

LIB = "MSigDB_Hallmark_2020"
URL = f"https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName={LIB}"
EXPECTED_N = 50  # the Hallmark collection is exactly 50 sets
OUT = Path(__file__).resolve().parent / "data" / "msigdb_sets.json"

def safe(s): return s.replace(" ", "_").replace("/", "_")

def main():
    print(f"Downloading {LIB} from Enrichr ...")
    try:
        raw = urllib.request.urlopen(URL, timeout=60).read().decode("utf-8", "ignore")
    except Exception as e:
        sys.exit(f"Download failed: {e}")

    lib = {}
    for line in raw.splitlines():
        if not line.strip(): continue
        tok = line.split("\t")
        term = tok[0].strip()
        genes = [t.split(",")[0].strip() for t in tok[1:] if t.strip()]
        if term: lib[term.lower()] = (term, genes)
    print(f"Library has {len(lib)} gene sets.\n")

    # Take the whole collection, sorted by name for a deterministic run order
    out = {}
    for _, (orig, genes) in sorted(lib.items()):
        out[safe(orig)] = {"genes": genes, "ground_truth": orig, "n_genes": len(genes)}
        print(f"[{orig}] {len(genes)} genes")

    if not out: sys.exit("Library came back empty.")
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nWrote {len(out)} sets -> {OUT.name}")
    if len(out) != EXPECTED_N:
        print(f"WARNING: expected {EXPECTED_N} Hallmark sets, got {len(out)}. "
              f"Check the library before running the benchmark.")
    else:
        print(f"OK: {EXPECTED_N} Hallmark sets, as expected.")

if __name__ == "__main__":
    main()