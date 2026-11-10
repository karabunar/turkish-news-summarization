"""Compare two runs on the test samples they share (paired bootstrap on per-sample ROUGE-L).

Usage (project root):
    python -m scripts.compare_runs --old experiments/04_final_1000train_200test/predictions.csv \
                                   --new outputs/predictions.csv
The two CSVs are aligned on `sample_id`; the article text of every shared id must be identical. Differences are new - old.
"""
import argparse

import numpy as np
import pandas as pd

from infrastructure.csv_exporter import read_predictions
from scripts.error_analysis import CONDS, boot_diff, rouge_sample


def align(old: pd.DataFrame, new: pd.DataFrame):
    """Return (old, new) restricted to the shared sample_ids, in the same order."""
    shared = [i for i in old["sample_id"] if i in set(new["sample_id"])]
    if not shared:
        raise ValueError("The two runs share no sample_id.")
    old = old.set_index("sample_id").loc[shared]
    new = new.set_index("sample_id").loc[shared]
    if (old["news_text"] != new["news_text"]).any():
        raise ValueError("Some shared sample_ids have different article text.")
    return old.reset_index(), new.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    args = ap.parse_args()
    old_all = read_predictions(args.old)
    new_all = read_predictions(args.new)
    old, new = align(old_all, new_all)
    print(f"old run: {len(old_all)} samples, new run: {len(new_all)} samples, shared: {len(old)}")
    print(f"{'condition':12s} {'R-L old':>8s} {'R-L new':>8s}  {'diff (new-old)':>15s}  95% CI")
    for c in CONDS:
        col = f"pred_{c}"
        a = np.array([rouge_sample(p, r)[2] for p, r in zip(old[col], old["reference_summary"])])
        b = np.array([rouge_sample(p, r)[2] for p, r in zip(new[col], new["reference_summary"])])
        m, lo, hi = boot_diff(b, a)
        same = int((old[col] == new[col]).sum())
        print(f"{c:12s} {a.mean()*100:8.2f} {b.mean()*100:8.2f}  {m:+15.2f}  [{lo:.2f}, {hi:.2f}]   identical outputs: {same}/{len(old)}")


if __name__ == "__main__":
    main()
