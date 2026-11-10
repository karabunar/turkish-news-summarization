# Supporting experiments
Earlier and side runs. The 1000-train run (`04`, R1) and the 3B run (`05`, R3) are used for the training-data and model-size comparisons in the main README; none of their numbers appears in the main results table. The main results are the full-data run in `../outputs/` (`bash run_full.sh`: all usable training rows, whole test split). Folders `00`–`03` are superseded early iterations that led to code fixes.

| Folder | What it is |
|---|---|
| `00_smoke_test/` | 16 train / 8 test samples, pipeline sanity check (numbers are meaningless) |
| `01_seq1024_baseline/` | First full run, `max_seq_length=1024`: 113/1000 training examples had their summary truncated and were ignored in the loss. Includes its adapter config and predictions (old parser) |
| `02_seq2048_training/` | **Training of the earlier 2048 adapter** (`max_seq_length=2048`, 186 steps, 33 min). `run_log.txt` holds the loss logs. `predictions.csv` are the evaluation outputs of this same run, produced with the parser *before* the abbreviation/markdown fixes. |
| `03_seq2048_final_eval/` | **Earlier evaluation**: C1-C4 regenerated with the fixed parser from the saved adapter (`--skip-training`). Log plus its predictions and raw generations (superseded) |
| `04_final_1000train_200test/` | R1: complete run on 1000 training / 200 test samples (`python main.py`): predictions, raw generations, `metrics.csv`, `run_log.txt`, adapter config and its own error analysis. Basis of the 1000-sample vs. full-data comparison |
| `05_3b_full_data/` | R3: Qwen2.5-3B-Instruct on the full data (`python main.py --all-data --use-validation --model Qwen/Qwen2.5-3B-Instruct --output-dir experiments/05_3b_full_data`; epoch with the lowest validation loss kept): predictions, raw generations, `metrics.csv`, logs, adapter config, and `cot_budget/` (C3 with 160 new tokens, `python -m scripts.cot_budget`). |

Prediction files: `predictions.csv` in `03`–`05` use the long format of the main run (one row per sample and condition: `sample_id, condition, prediction, reference_summary, news_text`); the older `predictions.csv` in `01`–`02` keep the original wide format (one `pred_<condition>` column per condition). The analysis scripts read both (`infrastructure.csv_exporter.read_predictions`). Adapter weights (`adapter_model.safetensors`) and intermediate training checkpoints are not tracked in git; only the adapter configs are.

The 3B run (`05_3b_full_data/`) also holds the inference-cost benchmarks in `kv_cache/` (KV cache, prefill/decode time, adapter merging, merged-model scores) with their console logs in `logs/03_*` to `05_*`.
