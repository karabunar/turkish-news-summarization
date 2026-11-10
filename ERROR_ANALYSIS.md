# Error Analysis — main run (full data), from `outputs/predictions.csv` and `outputs/raw_generations.csv`
Setup: Qwen2.5-1.5B-Instruct (4-bit NF4), whole test split (273 samples), seed 42; C4 = QLoRA r=16, α=32, 3 epochs, 4907 training samples (918 steps, 2 h 45 min), `max_seq_length=2048`, greedy decoding, `max_new_tokens=80`. C2 uses 3 random training-pool news texts (ids `9306bae92853156a`, `c471a6ba04edb934`, `1210ec7e2dd698e6`) that are not among the 4907 training samples. The earlier 1000-train / 200-test run is documented in `experiments/04_final_1000train_200test/ERROR_ANALYSIS_1000_200.md`. The 3B run (R3) with the same recipe is in `experiments/05_3b_full_data/`; its results are in the section "3B model (R3)" below.

Every number below is reproduced by `python -m scripts.error_analysis --leakage --n-train 4907` (own LCS-based ROUGE, equal to the official scores to two decimals; logs in `outputs/logs/`). ROUGE in `metrics.csv` is the plain mean of per-sample scores (deterministic).

| Cond. | R-1 | R-2 | R-L | chrF++ | BERTScore-F1 | Unsupported ent./num. (%) |
|---|---|---|---|---|---|---|
| C1 Zero-shot | 25.24 | 13.48 | 21.57 | 27.92 | 56.89 | 7.90 |
| C2 Few-shot | 28.95 | 17.50 | 25.06 | 28.99 | 59.32 | 4.33 |
| C3 Persona+CoT | 25.80 | 14.92 | 22.02 | 28.42 | 56.63 | 8.14 |
| C4 LoRA | 48.62 | 41.74 | 47.12 | 41.70 | 71.34 | 7.30 |

**Model-free baselines** (`python -m scripts.baselines --predictions outputs/predictions.csv --n-train 4907`; N = 28 words, the mean reference length in the training split):

| Baseline | R-1 | R-2 | R-L | chrF++ | BERTScore-F1 | Unsupported ent./num. (%) |
|---|---|---|---|---|---|---|
| Lead-1 sentence | 33.37 | 26.48 | 31.62 | 30.06 | 60.76 | 0.00 |
| Lead-2 sentences | 39.04 | 31.54 | 35.93 | 42.29 | 64.39 | 0.00 |
| First 28 words | 43.61 | 36.18 | 40.07 | 42.92 | 65.87 | 0.00 |
| Other news text's reference (floor) | 2.14 | 0.06 | 1.86 | 13.06 | 41.43 | 92.31 |

C4 is above the first-28-words baseline on ROUGE (R-L 47.1 vs 40.1) and BERTScore (71.3 vs 65.9) but slightly below it on chrF++ (41.7 vs 42.9). With 1000 training samples C4 was only 1.3 R-L points above the baseline (42.1 vs 40.8, on the 200 test samples of that run) and below it on chrF++ (38.1 vs 42.1), so the extra data moved the model beyond lead copying on ROUGE, although most of what it learns is still the dataset's lead-copying style (see A). The floor row shows that BERTScore without rescaling starts near 41, not 0.

**Significance** (paired bootstrap over samples, 5000 resamples, per-sample ROUGE-L, 95 % CI): C4−C1 +25.6 [21.9, 29.3]; C4−C2 +22.1 [18.5, 25.8]; C4−C3 +25.1 [21.5, 28.9]; C2−C1 +3.5 [2.0, 5.1]; **C3−C1 +0.5 [−0.9, 1.8] (not significant)**.

**Scale comparison** (`python -m scripts.compare_runs`; the 200 test news texts of the earlier run are among the 273, compared paired on the 200 shared samples): C4 42.15 → 47.67 R-L (**+5.52 [2.32, 8.85]**); C1 21.54 → 21.75 (+0.21 [−0.72, 1.09], 101/200 outputs identical); C3 21.63 → 22.12 (+0.49 [−0.18, 1.26], 122/200 identical); **C2 28.13 → 24.88 (−3.25 [−5.05, −1.45], 6/200 identical)** because the three random few-shot examples differ between the runs. The C1/C3 outputs differ in about half of the samples although prompts and decoding are identical (verified with `python -m scripts.batch_sensitivity`: a repeat with the same batch layout gives 200/200 identical raw C1 outputs, the same 200 prompts batched with the full test set or with batch size 4 or 1 differ in 112, 87 and 151 of 200 outputs, and ROUGE-L stays between 21.49 and 21.75; the old 1000/200 outputs are reproduced exactly (200/200), so the code did not change in between).

**Sensitivity within the 1000/200 run** (see the archived analysis): shortest vs random few-shot examples +2.67 [0.71, 4.64] (the same rerun also fixed prompt clipping, the parser's list-line handling and the entity metric, so the examples alone are not isolated; the full-data check is in "Few-shot sensitivity" below); second C4 training +1.47 [0.12, 3.08] (confounded with a code fix); `max_seq_length` 1024→2048 −0.40 [−2.11, 1.21]. The full-data C4 was trained once; differences below ≈3 R-L points should not be over-interpreted.

## A. Extractive vs. abstractive trade-off
| | Ref | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|
| Mean length (words) | 30.3 | 25.2 | 22.8 | 25.7 | 20.4 |
| Predicted 3-grams found in news text | 0.617 | 0.240 | 0.414 | 0.307 | **0.758** |
| Predicted unigrams absent from news text | 0.187 | 0.365 | 0.215 | 0.327 | **0.117** |
| Outputs ending with "..." | 85 | 0 | 0 | 0 | 72 |

Zero-shot and CoT outputs are strongly abstractive (≈33–37 % novel words, about twice the reference rate), which ROUGE penalizes. The fine-tuned model overshoots in the opposite direction (76 % copied 3-grams vs. 62 % in references): it learned the dataset's lead-copying style. A dataset artifact is reproduced: **85/273 references (31 %) end with "..."**, and 72 C4 outputs do too (21/200 in the smaller run), versus none in C1–C3. Part of C4's ROUGE gain therefore reflects imitation of reference formatting. Example `968acf8c…`: "Yıl sonu burç yorumları, yeni yıl burç yorumları, 1 Ocak burç yorumları..." while the reference lists the zodiac signs.

## B. Entity / name hallucination and language drift
Samples with ≥1 entity/number absent from the news text (of 273): C1 53, C2 32, C3 65, C4 38. Examples:
- C4 `e7eca972…`: "Volker Türk, Uluslararası Hukuki Konseyi (ICCPR) kapsamında, …" — reference: BM İnsan Hakları Yüksek Komiseri Volker Türk; the council is invented and the statement is misrepresented.
- C4 `05802811…`: "Posta Turu (PTT)’i yöneten Çap Arası Posta ve Telekomünikasyon Kurumu (CHP) …" — distorted institution names and a wrong abbreviation.
- C4 `71235f72…`: "Tehlikeli Tür 3 (The Sixth Extinction)" — reference gives "(Species III)"; fluent but wrong title.
- **Language drift:** CJK characters appear in 12 (C1), 2 (C2), 3 (C3) and **0 (C4)** outputs, e.g. C1 `a90aaeb7…` continues in Japanese and Korean after the first words. Fine-tuning removes the code-switching here.
- Note that the Kiptaş name that the smaller run distorted (`0ef7de7b…`) is correct in the full-data C4.
Caveat: the heuristic (capitalized non-initial words and digit groups) misses role/relation errors, and C2's low rate is partly due to its short outputs.

## C. Degenerate generation: repetition and budget truncation
From raw (pre-parser) outputs: word-level loops (a trigram ≥3×) C1 6, C2 6, C3 4, C4 4; repeated sentences C1 3, C2 5, C3 3, C4 1 (of 273). Loops are rare, e.g. C3 `3ecd33d7…`: "… hem de hem de hem de …". The dominant failure is **truncation at `max_new_tokens=80`**: raw outputs without terminal punctuation C1 236 (86 %), C2 136 (50 %), **C3 255 (93 %)**, C4 63 (23 %). For C3, 42/273 (15 %) never reach the "Nihai özet" line. The 160-token ablation below did not improve C3 (1.5B: R-L 20.8 vs 22.0), so the budget is not the only cause.

## D. Superficial overlap vs. faithfulness (metric disagreement)
- C4 beats the first-28-words baseline on ROUGE and BERTScore but not on chrF++, and the baseline has no invented entities by construction.
- C4's gain over C2 (+22 R-L) comes with a *higher* unsupported-entity rate (7.30 % vs 4.33 %; 5.05 % in the smaller run), so overlap does not certify factuality.
- Many ROUGE misses are morphological or paraphrastic and are scored as 0 although correct.
- Source noise: some news texts contain mojibake, URLs and update timestamps that also appear in gold summaries.
Recommendation: report ROUGE together with BERTScore/chrF++ and a manual faithfulness check on a sample.

## Limitations
1. Single seed and a single fine-tuning run per data size; 273 test samples; two model sizes (1.5B and 3B). Differences below ≈3 R-L points are not reliable.
2. The parser still splits after ordinal numbers ("6. galibiyet").
3. BERTScore uses `dbmdz/bert-base-turkish-cased` with `num_layers=9` and no baseline rescaling; use for ranking, not absolute values.
4. The unsupported-entity metric is a heuristic; the 4-character prefix rule still accepts some wrong matches.
5. C3 uses the 80-token budget; a 160-token ablation (see below) did not repair it, and the 3B model's C3 collapses.
6. C2 depends on the three examples: ROUGE-L ranges from 25.1 to 27.4 over seven example sets (see "Few-shot sensitivity").
7. Validation loss was not tracked in the main run.
8. The inference-cost numbers are single runs (50 news texts, batch size 1, one GPU); a repeat moved tok/s by about 2, so differences of a few percent are noise. The cause of the C2 decoding slowdown was not profiled, and the merge quality was scored for one adapter per model size.

## Train/test leakage check (`python -m scripts.check_leakage --n-train 4907 --n-test 1000000`, subsets via `scripts.error_analysis --leakage --n-train 4907`)
Test (273) vs. the 4907 training news texts: 1 exact duplicate, 2 identical first-200-character prefixes, 35 near-duplicates (5-gram Jaccard ≥ 0.5); **37 test news texts flagged** in total (13/200 in the smaller run). Classified by their openings: **3 template pages** ("Kim Kime Dum Duma …" cartoon pages), **33 reports of the same event**, **1 topical overlap**. Reference summaries: 10 exact duplicates among training references, 3 identical prefixes, 1 near-duplicate.

Mean ROUGE-L (own implementation, %), removing groups:

| Cond. | All 273 | − template (3) | − same-event (33) | − all flagged (37) | Flagged 37 only | Same-event 33 only |
|---|---|---|---|---|---|---|
| C1 | 21.6 | 21.8 | 21.3 | 21.6 | 21.4 | 23.8 |
| C2 | 25.1 | 25.3 | 24.2 | 24.5 | 28.7 | 31.6 |
| C3 | 22.0 | 22.3 | 21.9 | 22.2 | 21.1 | 23.2 |
| C4 | 47.1 | 46.5 | 47.1 | **46.6** | 50.7 | 47.2 |

Conclusion: removing all flagged news texts lowers C4 by 0.5 R-L and C2 by 0.6; the ranking and the gaps are unchanged (C4−C2 ≈ +22). The three template pages are copied perfectly by C4 (100.0 R-L, C1–C3 0.0), a formatting artefact. The official results use all 273 samples; the subset table is a sensitivity analysis. The check detects surface similarity only.

## Decoding ablation for C4 (`python -m scripts.decode_ablation`, results in `outputs/decode_ablation/`)
Same adapter, other decoding. The base variant is chosen on validation ROUGE-L only; test is scored for all variants. Test split (273):

| Variant | Val. R-L | R-1 | R-2 | R-L | Unsupported % |
|---|---|---|---|---|---|
| Greedy, 80 tok., 2 sent. (main setting) | 48.84 | 48.65 | 41.72 | 47.11 | 6.85 |
| Greedy, 120 tok., 3 sent. | 49.57 | 49.34 | 42.39 | 47.77 | 6.46 |
| Beam 4, 80 tok., 2 sent. | 52.53 | 50.19 | 43.61 | 48.61 | 2.57 |
| **Beam 4, 120 tok., 3 sent. (selected on validation)** | 53.03 | 50.39 | 43.80 | 48.79 | 2.43 |
| + entity repair | 53.07 | 50.38 | 43.77 | 48.76 | 2.16 |
| + most faithful of 4 beams | 52.18 | 50.32 | 43.66 | 48.68 | 1.76 |

Selected vs. main greedy output (paired bootstrap on test ROUGE-L): +1.67 [−0.15, 3.63], 54/273 outputs identical, so the ROUGE gain is not clearly significant, while the unsupported-entity rate drops from 6.9 % to 2.4 % (below C2's 4.33 %; the significance of this drop was not tested). Selected − C2: +23.7 [20.1, 27.6]. Notes: (i) the greedy row regenerates the main setting in a separate run (R-L 47.11 vs 47.12 and unsupported 6.85 vs 7.30 in the main run), so greedy outputs are not perfectly reproducible across runs; (ii) the selected variant allows 3 sentences although the C1 instruction asks for 1–2, but beam search alone with 2 sentences gives almost the same result (48.61, 2.57 %); (iii) `+repair` lowers the unsupported rate partly by construction; (iv) beam search was applied to C4 only, so these rows are not comparable to C1–C3, which stay greedy in the main table.

## 3B model (R3: Qwen2.5-3B-Instruct, full data; `experiments/05_3b_full_data/`)
Same recipe and splits as the main run (`python main.py --all-data --use-validation --model Qwen/Qwen2.5-3B-Instruct --output-dir experiments/05_3b_full_data`). QLoRA with 7,372,800 trainable parameters (0.238 %), 918 steps, 4 h 58 min. Validation loss per epoch 0.864 / **0.822** / 0.825, so the epoch-2 adapter is kept. Same 273 test samples and the same few-shot examples as the main run.

| Cond. | R-1 | R-2 | R-L | chrF++ | BERTScore-F1 | Unsupported ent./num. (%) |
|---|---|---|---|---|---|---|
| C1 Zero-shot | 27.43 | 14.37 | 22.79 | 28.89 | 58.58 | 5.90 |
| C2 Few-shot | 29.07 | 15.64 | 24.63 | 30.75 | 59.91 | 3.63 |
| C3 Persona+CoT | 6.74 | 3.62 | 5.95 | 8.23 | 38.44 | 67.02 |
| C4 LoRA | 50.69 | 43.40 | 48.74 | 43.38 | 72.52 | 6.21 |

Paired bootstrap on per-sample ROUGE-L, 3B minus 1.5B (same 273 samples): C1 +1.2 [−0.3, 2.9]; C2 −0.4 [−2.1, 1.2]; C3 −16.1 [−18.5, −13.7]; C4 +1.6 [−0.4, 3.7] (61/273 C4 outputs identical, none for C1–C3). Within the 3B run: C4−C1 +25.9 [22.4, 29.5]; C4−C2 +24.1 [20.7, 27.6]. So the larger model does not change the picture except for C3.

**C3 collapse.** Under 80 new tokens only 2/273 raw 3B outputs contain "Nihai özet"; the model writes a long, headed (markdown) analysis, the parser then falls back to a non-bullet line, usually the heading ("### Ana Olayı İnceleme"), so the parsed summaries average 9 words.

## C3 with a 160-token budget (`python -m scripts.cot_budget`, C3 only, test split; results in `outputs/cot_budget/` and `experiments/05_3b_full_data/cot_budget/`)
| Model | New tokens | R-1 | R-2 | R-L | chrF++ | BERTScore-F1 | Unsupported (%) | Outputs with marker | Mean words |
|---|---|---|---|---|---|---|---|---|---|
| 1.5B | 80 (main) | 25.80 | 14.92 | 22.02 | 28.42 | 56.63 | 8.14 | 231/273 (derived from 42 without) | 25.7 |
| 1.5B | 160 | 24.53 | 13.95 | 20.81 | 29.47 | 56.10 | 6.64 | 239/273 | 32.8 |
| 3B | 80 (main run) | 6.74 | 3.62 | 5.95 | 8.23 | 38.44 | 67.02 | 2/273 | 9.1 |
| 3B | 160 | 8.40 | 4.57 | 7.38 | 9.74 | 39.68 | 63.42 | 33/273 | 10.3 |

Doubling the budget does not repair C3 for either model: the 1.5B score does not improve, and the 3B analysis is longer than 160 tokens (191/273 raw 3B outputs start with a markdown heading, 180 parsed outputs still start with "###"). Larger budgets or a prompt that forces a shorter analysis were not tried. The main tables keep 80 tokens for every condition.

## Few-shot sensitivity on the full test split (`python -m scripts.fewshot_sensitivity --n-sets 5`, results in `outputs/fewshot_sensitivity/`)
C2 (1.5B, 273 test samples) with seven example sets: the set of the main run, the three shortest training news texts, and five further random triples from the 4907 training news texts. Paired bootstrap difference in ROUGE-L against the main set:

| Example set | R-1 | R-2 | R-L | Unsupp. (%) | R-L vs main [95 % CI] |
|---|---|---|---|---|---|
| main (ids `9306bae9…`, `c471a6ba…`, `1210ec7e…`) | 28.95 | 17.50 | 25.06 | 4.33 | – |
| 3 shortest | 30.38 | 19.36 | 27.03 | 2.94 | +1.97 [0.16, 3.78] |
| random 1 | 29.04 | 17.73 | 25.23 | 4.75 | +0.17 [−1.61, 1.93] |
| random 2 | 29.09 | 17.51 | 25.31 | 4.31 | +0.25 [−1.36, 1.85] |
| random 3 | 28.92 | 17.95 | 25.54 | 3.52 | +0.48 [−1.25, 2.24] |
| random 4 | 30.10 | 19.02 | 26.85 | 5.80 | +1.79 [−0.16, 3.70] |
| random 5 | 30.98 | 19.48 | 27.42 | 4.54 | +2.36 [0.34, 4.41] |

Random sets: mean R-L 26.07, sd 1.00, min 25.23, max 27.42; all seven sets span 2.36 points. The main run's set is the lowest, so the main C2 row is on the low side; with the best set (random 5) C4−C2 would be +19.7 [16.2, 23.3] instead of +22.1, which does not change the conclusion. The earlier finding that the three shortest news texts scored lower (1000/200 run, confounded with other fixes) does not carry over: these shortest news texts (a different set, drawn from the larger pool) score higher here, so no claim is made about which kind of example is best. The "main" row reproduces the main run exactly (25.06 / 4.33).

## Batch sensitivity (`python -m scripts.batch_sensitivity`, results in `outputs/batch_sensitivity/`)
C1 raw outputs for the 200 test news texts shared with the 1000/200 run, compared pairwise (identical raw outputs out of 200): A1 (200 news texts, batch size 8) = A2 (repeat) 200/200; A1 = stored 1000/200 run 200/200; B (the 200 news texts inside the full 273-sample test set, batch size 8) = stored full run 200/200; A1 vs B 88; A1 vs C (batch size 4) 113; A1 vs D (batch size 1) 49. ROUGE-L: A1 21.54, A2 21.54, B 21.75, C 21.61, D 21.49. Greedy generation is deterministic for a fixed batch layout, but the batch layout changes about half of the C1 outputs without changing ROUGE-L by more than 0.3 (only C1 was tested).

## Inference cost and adapter merging (`python -m scripts.kv_cache_benchmark`, `scripts.merge_benchmark`, `scripts.merged_quality`; results in `outputs/kv_cache/`)

Qwen2.5-1.5B (28 layers, 2 KV heads) needs 28 672 bytes of KV cache per token in bf16. Batch size 1, greedy, first 50 test news texts, one RTX A4000 (prefill = prompt pass plus first token, timed in a separate call; decode = remaining time per token):

| Run | Prompt tok. | New tok. | Prefill (s) | Decode (ms/tok) | tok/s | KV (MB) |
|---|---|---|---|---|---|---|
| C1 zero-shot | 647 | 79.7 | 0.10 | 19.6 | 48.5 | 19.8 |
| C2 few-shot | 1747 | 67.1 | 0.27 | 22.0 | 38.9 | 49.6 |
| C3 persona+CoT | 712 | 80.0 | 0.13 | 20.3 | 46.2 | 21.6 |
| C4 LoRA (unmerged) | 647 | 55.8 | 0.14 | 27.6 | 33.8 | 19.2 |
| C1, adapter bypassed | 647 | 79.7 | 0.13 | 20.7 | 45.3 | 19.8 |

- The measured KV size equals the theoretical one. The 4-bit weights occupy 1.5 GB, so the extra 30 MB of the C2 cache adds only about 2% to the bytes read per token, yet the decoding step is 12% slower. The cause was not profiled.
- C4 has the same prompt as C1 but a 33% slower decoding step than the adapter-bypassed model (separate LoRA branches in every layer); it also stops earlier, so tok/s is lower while the time per sample is equal. A repeat of the benchmark moved tok/s by about 2, so differences of a few percent are noise.
- Merging (30 news texts, exactly 64 decoded tokens, C1 prompt): decode ms/token, 4-bit base 18.7 (no adapter), 26.8 (adapter), 18.8 (merged); bf16 base 16.2, 20.1, 17.2.
- Quality of the merged model (273 test news texts, C4, same decoding and parser as the main run):

| Variant | ROUGE-1 | ROUGE-2 | ROUGE-L | Unsupported % | Identical to unmerged |
|---|---|---|---|---|---|
| adapter, 4-bit (reported setting) | 48.65 | 41.72 | 47.11 | 6.85 | 100% |
| merged into 4-bit | 30.77 | 19.07 | 27.05 | 6.16 | 0.4% |
| merged into bf16 | 47.07 | 39.61 | 45.43 | 7.29 | 27% |

Merging into the requantized 4-bit weights removes the speed cost but loses 20 ROUGE-L points; merging into bf16 loses 1.7 and doubles the weight memory. The adapter was trained against the quantized base, so its update does not transfer unchanged. Single runs, one adapter.

### Same benchmarks with Qwen2.5-3B-Instruct (R3 adapter; `experiments/05_3b_full_data/kv_cache/`, logs `experiments/05_3b_full_data/logs/03_*` to `05_*`)

36 layers, 2 KV heads: 36 864 bytes of KV cache per token (1.5B: 28 672). Same setup as above (batch 1, greedy, first 50 test news texts; merge test: 30 texts, 64 decoded tokens).

| Run | Prompt tok. | New tok. | Prefill (s) | Decode (ms/tok) | tok/s | KV (MB) |
|---|---|---|---|---|---|---|
| C1 zero-shot | 647 | 80.0 | 0.23 | 27.7 | 33.4 | 25.5 |
| C2 few-shot | 1747 | 77.4 | 0.51 | 32.9 (+19%) | 25.7 | 64.1 |
| C3 persona+CoT | 712 | 80.0 | 0.26 | 28.6 (+3%) | 32.0 | 27.8 |
| C4 LoRA (unmerged) | 647 | 56.8 | 0.25 | 35.7 | 25.1 | 24.7 |
| C1, adapter bypassed | 647 | 80.0 | 0.24 | 28.0 | 32.8 | 25.5 |

Merging (decode ms/token): 4-bit base 27.4 (no adapter), 36.3 (adapter, +32%), 28.4 (merged); bf16 base 29.8, 36.0 (+21%), 30.1 (+1%). Generations identical to the unmerged adapter: 0% (4-bit) and 73% (bf16). Weights occupy 2.5 GB (4-bit) and 6.6 GB (bf16).

- The C2 decoding slowdown is not smaller with the larger model (+19% vs. +12% for 1.5B), although the C2 cache is only about 2.5% of the 4-bit weight bytes (64 MB vs. 2.5 GB). Reading the cache alone therefore does not explain it; the cause was not profiled.
- The unmerged adapter costs 28-32% per decoding step for both model sizes, and merging removes it in the 4-bit and bf16 setting alike.
- Quality of the merged 3B model (273 test news texts, C4, greedy, 2-sentence parser; `merged_quality.csv`):

| Variant | ROUGE-1 | ROUGE-2 | ROUGE-L | Unsupported % | Identical to unmerged |
|---|---|---|---|---|---|
| adapter, 4-bit | 50.54 | 43.21 | 48.64 | 5.63 | 100% |
| merged into 4-bit | 31.73 | 18.72 | 27.35 | 2.39 | 0.0% |
| merged into bf16 | 49.84 | 42.39 | 47.93 | 5.57 | 42% |

Merging into the 4-bit base loses 21 ROUGE-L points again; bf16 loses 0.7. The low unsupported rate of the 4-bit merge does not mean better summaries (its ROUGE collapses; it was not inspected further).
