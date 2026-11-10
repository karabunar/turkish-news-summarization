# Experiment details

Everything behind the numbers in the [README](../README.md). The full error analysis, with every supporting number, is in [`ERROR_ANALYSIS.md`](../ERROR_ANALYSIS.md). All commands run from the repo root, inside the Docker container.

## Runs

| Run | Model | Train / test | Where |
|---|---|---|---|
| R1 | Qwen2.5-1.5B-Instruct | 1000 / 200 | `experiments/04_final_1000train_200test/` |
| **R2 (main)** | Qwen2.5-1.5B-Instruct | 4907 / 273 | `outputs/` |
| R3 | Qwen2.5-3B-Instruct | 4907 / 273, epoch with the lowest validation loss | `experiments/05_3b_full_data/` |

Earlier iterations (`experiments/00`–`03`) are described in [`experiments/README.md`](../experiments/README.md).

`outputs/subset_shared200/metrics.csv` holds the main run's (R2) scores on only the 200 test news texts that R1 used. The training-data comparison (R1 vs. R2) uses these 200 texts.

## Data and outputs

**Data.** [`ahmetyaylalioglu/turkish-news-summarization-collection`](https://huggingface.co/datasets/ahmetyaylalioglu/turkish-news-summarization-collection), columns `text` / `target_summary`. Main run (`--all-data`): `train` split → 4907 samples for LoRA, plus 3 further random samples as few-shot examples (never among the 4907). The whole `test` split (273 samples) is used for all of C1–C4. Both splits use `shuffle(seed=42)`. The default `python main.py` (1000 train / 200 test) is the smaller R1 run.

**Outputs** (under `outputs/`):
- `predictions.csv` is in long format: one row per sample and condition, 273 × 4 = 1092 rows, with columns `sample_id, condition, prediction, reference_summary, news_text`. The analysis scripts read it with `infrastructure.csv_exporter.read_predictions`, which also accepts the older wide format of `experiments/01`–`02`.
- `raw_generations.csv` holds the unprocessed model outputs.
- `lora_adapter/` holds the adapter config. The weights are on the [Hub](https://huggingface.co/karabunar/qwen2.5-1.5b-turkish-news-summarization-qlora).

**`main.py` flags:** `--model Qwen/Qwen2.5-3B-Instruct`, `--batch-size 4` (if OOM), `--skip-training` (reuse `outputs/lora_adapter/`), `--use-validation`, `--output-dir` (default `outputs`), `--max-new-tokens 80`, `--no-supplementary`.

## Logs
Console output of each step. Main run and 1.5B experiments: `outputs/logs/`. 3B run: `experiments/05_3b_full_data/logs/` (numbered from 01).

| Log | Produced by |
|---|---|
| `outputs/logs/01_pytest.txt` | `./run_tests.sh` |
| `outputs/logs/02_main_run.txt` | `python main.py --all-data` (training, C1-C4, ROUGE and supplementary metrics) |
| `outputs/logs/03_few_shot_examples.txt` | `python -m scripts.show_few_shot` (the three few-shot examples) |
| `outputs/logs/04_evaluate_predictions.txt` | `python -m scripts.evaluate_predictions` (scores recomputed from the CSV) |
| `outputs/logs/05_error_analysis_and_leakage.txt`, `06_check_leakage.txt` | `scripts.error_analysis --leakage`, `scripts.check_leakage` |
| `outputs/logs/07_fewshot_sensitivity.txt`, `08_batch_sensitivity.txt`, `09_cot_budget.txt` | `scripts.fewshot_sensitivity`, `scripts.batch_sensitivity`, `scripts.cot_budget` |
| `outputs/logs/10_kv_cache_benchmark.txt`, `11_merge_benchmark.txt`, `12_merged_quality.txt` | `scripts.kv_cache_benchmark`, `scripts.merge_benchmark`, `scripts.merged_quality` (1.5B) |
| `outputs/decode_ablation/run_log.txt` | `scripts.decode_ablation` |
| `experiments/05_3b_full_data/logs/01_main_run.txt` | 3B run: `python main.py --all-data --use-validation --model Qwen/Qwen2.5-3B-Instruct` |
| `experiments/05_3b_full_data/logs/02_cot_budget.txt` | 3B: `scripts.cot_budget` |
| `experiments/05_3b_full_data/logs/03_kv_cache_benchmark.txt`, `04_merge_benchmark.txt`, `05_merged_quality.txt` | 3B: `scripts.kv_cache_benchmark`, `scripts.merge_benchmark`, `scripts.merged_quality` |
| `experiments/0*/run_log.txt` | Earlier runs (see `experiments/README.md`) |

## Reproducibility
| Item | Status |
|---|---|
| Data split | `shuffle(seed=42)`; 4907 train + 3 few-shot (disjoint) + 273 test (whole split); few-shot ids `9306bae92853156a`, `c471a6ba04edb934`, `1210ec7e2dd698e6` (`python -m scripts.show_few_shot --n-train 4907`) |
| Environment | Docker image, torch 2.2.1 / CUDA 12.1, transformers 4.44.2, peft 0.12.0, trl 0.9.6, bitsandbytes 0.43.3 |
| Generation | greedy, `max_new_tokens=80`, same decoding for all conditions |
| Metrics | deterministic: `python -m scripts.evaluate_predictions` regenerates `outputs/metrics.csv` from the predictions CSV |
| Analysis | `python -m scripts.error_analysis --leakage` regenerates every number in `ERROR_ANALYSIS.md` (bootstrap seed 42) |
| Prompts and configuration | `config/prompts.json`, `config/experiment_config.json` (kept in sync with the code by `tests/test_config_files.py`) |
| Tests | `./run_tests.sh` (2 tests are skipped on a machine without `evaluate` or `datasets`) |
| Figure | `python -m scripts.make_figure` rebuilds `docs/figures/runs_comparison.{pdf,png}` from the three `metrics.csv` files |

Not guaranteed to reproduce:
- **Retraining C4** may give a different adapter and different scores. The reported runs used one RTX A4000, bf16 and a 4-bit base. In R1, the second training also followed a code fix, so GPU noise and code changes are confounded (ROUGE-L +1.5 [0.1, 3.1], 160/200 outputs identical). The full-data run was trained once.
- **Greedy outputs** change with batch layout and GPU. For C1, about half of the outputs change when the batch layout changes, but ROUGE-L moved by less than 0.3 in that check (see `ERROR_ANALYSIS.md`).
- **The HF dataset revision** is not pinned.

## Extra experiments
None of these changes the main C1–C4 results.

| Command | What it does |
|---|---|
| `python -m scripts.baselines --predictions outputs/predictions.csv --n-train 4907` | Model-free baselines (lead-1/2, first-N words, another news text's reference as a floor) with the same metrics (CPU only) |
| `python -m scripts.compare_runs --old <csv> --new <csv>` | Paired bootstrap on ROUGE-L over the test samples two runs share |
| `python -m scripts.decode_ablation` | C4 with greedy/beam, 80/120 new tokens and a parser limit of 2/3 sentences, plus optional entity repair and "most faithful of 4 beams". The variant is chosen on the validation split only; the test split is scored for all variants (GPU) |
| `python main.py --all-data --use-validation --output-dir <dir>` | Evaluates on the validation split every epoch and keeps the epoch with the lowest validation loss |
| `python main.py --all-data --use-validation --model Qwen/Qwen2.5-3B-Instruct --output-dir experiments/05_3b_full_data` | Same pipeline with a larger model (add `--batch-size 4` if generation runs out of memory) |
| `python -m scripts.cot_budget [--model ...] --output-dir <dir>` | C3 only, with 160 new tokens (does a larger budget repair persona+CoT?) (GPU) |
| `python -m scripts.fewshot_sensitivity --n-sets 5` | C2 on the full test split with seven example sets: the main set, the three shortest, five random (GPU) |
| `python -m scripts.batch_sensitivity` | C1 with different batch layouts: determinism and batch-composition effect (GPU) |
| `python -m scripts.kv_cache_benchmark --n-samples 50` | Prompt length, KV-cache size, latency and tokens/s per condition (GPU; `--tokens-only` needs none). Results in `outputs/kv_cache/`; for the 3B model, add `--model Qwen/Qwen2.5-3B-Instruct --adapter-dir experiments/05_3b_full_data/lora_adapter --output-dir experiments/05_3b_full_data/kv_cache`. The C4 rows need the adapter weights |
| `python -m scripts.merge_benchmark --n-samples 30` | Decode time per token of the base, the unmerged LoRA adapter and the merged adapter, on a 4-bit and a bf16 base (GPU, adapter needed). Results in `outputs/kv_cache/merge_benchmark.csv` |
| `python -m scripts.merged_quality` | ROUGE-L and unsupported rate of C4 with the adapter unmerged vs. merged into the 4-bit or a bf16 base (GPU, adapter needed). Results in `outputs/kv_cache/merged_quality.csv` |

Caveat for `+repair`: it replaces a name that is absent from the news text with the most similar name from the news text. The unsupported-entity metric therefore improves partly by construction, so judge it by ROUGE and by reading the changed outputs.

## Notes and caveats
- **Hardware.** All reported runs used a single NVIDIA RTX A4000 (16 GB). The code uses bf16 (4-bit compute dtype, training, generation autocast), which needs an Ampere-or-newer GPU. On GPUs without bf16 support (e.g. Tesla T4 / free Colab), switch to fp16; scores will then differ slightly.
- **ROUGE** is the plain mean of per-sample scores, so `python -m scripts.evaluate_predictions` reproduces `outputs/metrics.csv` exactly. The `evaluate` bootstrap aggregator is avoided because it is not deterministic.
- **Turkish tokenization.** ROUGE uses a Unicode-aware Turkish tokenizer. The default `rouge_score` tokenizer deletes ç, ğ, ı, ö, ş, ü and would corrupt the scores.
- **C3 budget.** C3 asks for a short analysis, then `Nihai özet:`. With `max_new_tokens=80` the analysis can use up the budget, and the parser then falls back to non-bullet lines. For the 3B model this is usually a markdown heading, and C3 collapses to ROUGE-L 5.9. A 160-token ablation (`scripts.cot_budget`) did not repair it for either model.
- **Clipping.** News text is clipped to 2500 characters (few-shot examples to 800) so prompts fit in 16 GB. This applies equally to all conditions.
- **Few-shot examples** are 3 random training-split news texts (seed 42). They are never among the LoRA training samples or from the test set; `python -m scripts.show_few_shot` prints them. The main run's set gives the lowest C2 score of the seven sets tried.
- **C4 loss** is computed on summary tokens only (`DataCollatorForCompletionOnlyLM`).
- **Supplementary metrics.** chrF++ (`word_order=2`), BERTScore F1 with `dbmdz/bert-base-turkish-cased` (`num_layers=9`), and the corpus-level share of capitalized entities/numbers in a prediction that never appear in the news text (`domain/services/supplementary_metrics.py`; disable with `--no-supplementary`). The entity heuristic is approximate (capitalized non-sentence-initial words), so use it to compare conditions, not as an absolute hallucination rate.
- **Version pins.** trl 0.9.6 and transformers 4.44.2 are chosen for torch 2.2.1 compatibility. If a pip conflict appears, adjust the pins together.
