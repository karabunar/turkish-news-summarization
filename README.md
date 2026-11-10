# Fine-Tuning vs. Prompt Engineering for Turkish News Summarization

How far can prompting take a small instruction-tuned LLM on Turkish news summarization, and what does a cheap QLoRA fine-tune buy on top of it?
This repo compares four conditions on **Qwen2.5-1.5B-Instruct** (4-bit NF4), evaluated on the full test split of a Turkish news summarization dataset, with error analysis, significance tests, baselines and inference-cost benchmarks.

| Condition | What it is |
|---|---|
| **C1 Zero-shot** | Single instruction: *"Aşağıdaki haber metnini 1-2 cümle ile anlaşılır şekilde özetle:"* |
| **C2 Few-shot** | Three solved examples from the training split as a multi-turn chat, then the test article |
| **C3 Persona + CoT** | "Expert news editor" persona, short step-by-step analysis, then `Nihai özet:` |
| **C4 QLoRA** | LoRA (r=16, α=32, q/k/v/o) trained on 4907 article–summary pairs, evaluated with the zero-shot prompt |

Trained adapter: **[{{HF_REPO}}](https://huggingface.co/{{HF_REPO}})**

## Results

Main run (R2): Qwen2.5-1.5B-Instruct, 4907 training samples, whole test split (273 samples), greedy decoding, `max_new_tokens=80`.

| Condition | ROUGE-1 | ROUGE-2 | ROUGE-L | chrF++ | BERTScore-F1 | Unsupported ent./num. (%) ↓ |
|---|---|---|---|---|---|---|
| C1 Zero-shot | 25.24 | 13.48 | 21.57 | 27.92 | 56.89 | 7.90 |
| C2 Few-shot | 28.95 | 17.50 | 25.06 | 28.99 | 59.32 | **4.33** |
| C3 Persona + CoT | 25.80 | 14.92 | 22.02 | 28.42 | 56.63 | 8.14 |
| **C4 QLoRA** | **48.62** | **41.74** | **47.12** | **41.70** | **71.34** | 7.30 |
| *First-28-words baseline* | *43.61* | *36.18* | *40.07* | *42.92* | *65.87* | *0.00* |

![Scores of the three runs](docs/figures/runs_comparison.png)

Key findings (details and every supporting number in [`ERROR_ANALYSIS.md`](ERROR_ANALYSIS.md)):

- **Fine-tuning beats prompting by a wide margin.** C4 is +22 to +26 ROUGE-L over every prompting condition (paired bootstrap, all CIs far from zero).
- **CoT does not help a 1.5B model here.** C3 vs. C1 is +0.5 ROUGE-L [−0.9, 1.8], not significant. For the 3B model, persona+CoT collapses (ROUGE-L 5.9) because the analysis eats the token budget. A 160-token ablation does not fix it.
- **Few-shot helps, but it depends on which examples you pick.** Over seven example sets, C2 ROUGE-L ranges from 25.1 to 27.4.
- **Part of the gain is style imitation.** The fine-tuned model learns the dataset's lead-copying style (76 % of its 3-grams appear in the article vs. 62 % for references), and a trivial first-28-words baseline already reaches 40.1 ROUGE-L. C4 beats it on ROUGE and BERTScore, but not on chrF++.
- **More data and a bigger model help, modestly.** C4 gains +5.5 ROUGE-L [2.3, 8.9] going from 1000 to 4907 training samples (paired, on 200 shared test samples). The 3B model adds +1.6 ROUGE-L.
- **Merging a QLoRA adapter is not free.** The unmerged adapter makes each decoding step 33 % slower. Merging it into the requantized 4-bit base restores the speed but costs 20 ROUGE-L points. Merging into a bf16 base costs only 1.7 points, but doubles the weight memory.

| Run | Model | Train / test | Where |
|---|---|---|---|
| R1 | Qwen2.5-1.5B-Instruct | 1000 / 200 | `experiments/04_final_1000train_200test/` |
| **R2 (main)** | Qwen2.5-1.5B-Instruct | 4907 / 273 | `outputs/` |
| R3 | Qwen2.5-3B-Instruct | 4907 / 273, epoch with the lowest validation loss | `experiments/05_3b_full_data/` |

## Quick start

### Use the trained adapter
```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

base = "Qwen/Qwen2.5-1.5B-Instruct"
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, device_map="auto")
model = PeftModel.from_pretrained(model, "{{HF_REPO}}")  # keep it unmerged (see findings)

article = "..."  # Turkish news text (the experiments clip it to 2500 characters)
messages = [{"role": "user", "content": f"Aşağıdaki haber metnini 1-2 cümle ile anlaşılır şekilde özetle:\n\n{article}"}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(inputs, max_new_tokens=80, do_sample=False)
print(tok.decode(out[0, inputs.shape[1]:], skip_special_tokens=True))
```
This matches the evaluated setting (4-bit NF4 base, unmerged adapter, greedy). The reported scores also apply a light post-processing parser (`domain/services/summary_parser.py`).

### Reproduce the experiments
```bash
export HF_TOKEN=...                         # optional
docker compose build
docker compose run --rm summarizer bash     # opens a shell in the GPU container

# inside the container
nvidia-smi                                  # sanity check: GPU visible
bash run_tests.sh                           # unit tests
bash run_full.sh                            # tests, full-data pipeline (C1-C3 -> train -> C4 -> eval), analysis, inference-cost benchmarks; logs in outputs/logs/
# or only the pipeline: python main.py --all-data
```
Useful flags: `--model Qwen/Qwen2.5-3B-Instruct`, `--batch-size 4` (if OOM), `--skip-training` (reuse `outputs/lora_adapter/`), `--output-dir` (default `outputs`), `--max-new-tokens 80`.

To skip training and reuse the published adapter:
```bash
hf download {{HF_REPO}} --local-dir outputs/lora_adapter
python main.py --all-data --skip-training
```

To check the reported scores without a GPU, recompute them from the saved predictions: `python -m scripts.evaluate_predictions` (regenerates `outputs/metrics.csv`) and `python -m scripts.error_analysis --leakage --n-train 4907` (regenerates every number in `ERROR_ANALYSIS.md`).

## Layout
```
domain/          entities, PromptBuilder, SummaryParserService, EvaluationService
application/     ports, RunPromptSummarization / TrainLoraSummarizer / ExportResults use cases
infrastructure/  HuggingFaceModelAdapter (NF4 + QLoRA), dataset repository, CSV exporter
tests/           pytest suite (parser, evaluation, prompt builder, config consistency, analysis and benchmark helpers)
config/          prompts.json (all prompts) and experiment_config.json (model, data, decoding, QLoRA, metrics, runs); tests/test_config_files.py checks them against the code
scripts/         baselines, decode ablation, error analysis, leakage check, few-shot/batch/CoT-budget experiments, inference-cost benchmarks (KV cache, prefill/decode time, adapter merging), figure
outputs/         main-run results: predictions, metrics, logs/ (console output of each step), kv_cache/, decode_ablation/, ...
experiments/     earlier and side runs (R1, R3 and early iterations), see experiments/README.md
docs/figures/    run comparison figure
main.py          pipeline entry point
```

**Data.** [`ahmetyaylalioglu/turkish-news-summarization-collection`](https://huggingface.co/datasets/ahmetyaylalioglu/turkish-news-summarization-collection), columns `text` / `target_summary`. Main run (`--all-data`): `train` split → 4907 samples for LoRA, plus 3 further random samples as few-shot examples (never among the 4907). The whole `test` split (273 samples) is used for all of C1–C4. Both splits use `shuffle(seed=42)`. The default `python main.py` (1000 train / 200 test) is the smaller R1 run.

**Outputs** (under `outputs/`):
- `predictions.csv` is in long format: one row per sample and condition, 273 × 4 = 1092 rows, with columns `sample_id, condition, prediction, reference_summary, news_text`. The analysis scripts read it with `infrastructure.csv_exporter.read_predictions`, which also accepts the older wide format of `experiments/01`–`02`.
- `raw_generations.csv` holds the unprocessed model outputs.
- `lora_adapter/` holds the adapter config. Download the weights from the Hub as shown above.

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

`outputs/subset_shared200/metrics.csv` holds the main run's (R2) scores on only the 200 test news texts that R1 used. The training-data comparison (R1 vs. R2) uses these 200 texts.

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

| Command (inside the container) | What it does |
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

## License and acknowledgements
Code: [MIT](LICENSE).

- Dataset: [Turkish News Summarization (RSS Crawled)](https://huggingface.co/datasets/ahmetyaylalioglu/turkish-news-summarization-collection) by `ahmetyaylalioglu`, CC BY 4.0. The prediction CSVs in this repo contain news texts and reference summaries from it.
- Models: [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) (Apache 2.0), [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) (Qwen Research License), [dbmdz/bert-base-turkish-cased](https://huggingface.co/dbmdz/bert-base-turkish-cased) for BERTScore.
