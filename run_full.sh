#!/usr/bin/env bash
# Full-data run (all usable train rows, whole test split). Run inside the container from the project root:
#   bash run_full.sh
# Every step's console output is also saved under outputs/logs/.
set -euo pipefail   # stop at the first failing step (a failed pytest must not be followed by a long GPU run)
OUT=outputs
mkdir -p "$OUT/logs"

python -m pytest tests/ -v 2>&1 | tee "$OUT/logs/01_pytest.txt"
python main.py --all-data --output-dir "$OUT" 2>&1 | tee "$OUT/logs/02_main_run.txt"

N_TRAIN=$(grep -m1 -o "train=[0-9]*" "$OUT/logs/02_main_run.txt" | cut -d= -f2)
python -m scripts.show_few_shot --n-train "$N_TRAIN" 2>&1 | tee "$OUT/logs/03_few_shot_examples.txt"
python -m scripts.evaluate_predictions --predictions "$OUT/predictions.csv" --output "$OUT/metrics_recomputed.csv" 2>&1 | tee "$OUT/logs/04_evaluate_predictions.txt"
python -m scripts.error_analysis --predictions "$OUT/predictions.csv" --raw "$OUT/raw_generations.csv" \
    --leakage --n-train "$N_TRAIN" 2>&1 | tee "$OUT/logs/05_error_analysis_and_leakage.txt"
python -m scripts.check_leakage --n-train "$N_TRAIN" --n-test 1000000 2>&1 | tee "$OUT/logs/06_check_leakage.txt"

# Inference cost: prompt length / KV cache / prefill and decode time, and the effect of merging the adapter
python -m scripts.kv_cache_benchmark --n-samples 50 2>&1 | tee "$OUT/logs/10_kv_cache_benchmark.txt"
python -m scripts.merge_benchmark --n-samples 30 2>&1 | tee "$OUT/logs/11_merge_benchmark.txt"
python -m scripts.merged_quality 2>&1 | tee "$OUT/logs/12_merged_quality.txt"
