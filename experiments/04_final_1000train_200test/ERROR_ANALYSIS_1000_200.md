# Error Analysis — earlier 1000-train / 200-test run (R1), from `predictions.csv` and `raw_generations.csv` in this folder (written to `outputs/` at the time of the run)
Setup: Qwen2.5-1.5B-Instruct (4-bit NF4), 200 held-out samples, seed 42; C4 = QLoRA r=16, α=32, 3 epochs, 1000 samples, `max_seq_length=2048`, greedy decoding, `max_new_tokens=80`. C2 uses 3 random training-pool articles (ids `0c8fb9296627b93b`, `844ea7981b24afe1`, `38c225fa881fe1f3`) that are not in the 1000 training samples.

Every number below is reproduced by `python -m scripts.error_analysis --leakage` (own LCS-based ROUGE, equal to the official scores to two decimals). ROUGE in `metrics.csv` is the plain mean of per-sample scores (deterministic).

| Cond. | R-1 | R-2 | R-L | chrF++ | BERTScore-F1 | Unsupported ent./num. (%) |
|---|---|---|---|---|---|---|
| C1 Zero-shot | 25.48 | 13.54 | 21.54 | 26.90 | 56.65 | 6.34 |
| C2 Few-shot | 31.79 | 20.39 | 28.13 | 29.54 | 60.55 | 3.98 |
| C3 Persona+CoT | 25.76 | 14.69 | 21.63 | 27.19 | 56.49 | 8.59 |
| C4 LoRA | 44.08 | 36.71 | 42.15 | 38.11 | 68.02 | 5.05 |

**Model-free baselines** (`python -m scripts.baselines --predictions <csv> --n-train 1000`, same metrics; N = mean reference length of the training split, 28 words):

| Baseline | R-1 | R-2 | R-L | chrF++ | BERTScore-F1 | Unsupported ent./num. (%) |
|---|---|---|---|---|---|---|
| Lead-1 sentence | 33.04 | 26.03 | 31.27 | 28.43 | 60.70 | 0.00 |
| Lead-2 sentences | 39.37 | 31.90 | 36.40 | 41.10 | 64.69 | 0.00 |
| First 28 words | 44.28 | 36.93 | 40.82 | 42.05 | 66.33 | 0.00 |
| Other article's reference (floor) | 2.08 | 0.06 | 1.75 | 12.81 | 41.48 | 91.98 |

The first-28-words baseline equals C4 on ROUGE (R-L 40.8 vs 42.1), exceeds it on chrF++ (42.1 vs 38.1) and is slightly below it on BERTScore (66.3 vs 68.0). The floor row (a fluent reference summary of a different article) shows that BERTScore without rescaling starts near 41, not 0: rescaled with this floor, C1/C3 are at about 0.26, C4 at 0.45 and the 28-word baseline at 0.42 of the way to 1. On ROUGE and chrF++ all prompted conditions are far below the baseline. On this dataset, fine-tuning mostly teaches the model to copy the opening of the article at the reference length.

**Significance** (paired bootstrap over samples, 5000 resamples, per-sample ROUGE-L, 95 % CI): C4−C1 +20.6 [16.8, 24.6]; C4−C2 +14.0 [10.0, 18.1]; C4−C3 +20.5 [16.6, 24.5]; C2−C1 +6.6 [4.5, 8.8]; **C3−C1 +0.1 [−1.7, 1.9] (not significant)**.

**Sensitivity to design choices** (same test samples, paired bootstrap on ROUGE-L):
- *Few-shot example choice:* the first version of C2 used the three shortest training articles (R-L 25.46); three random articles give 28.13 (+2.67 [0.71, 4.64]; 6/200 outputs identical). C2 therefore depends noticeably on which examples are shown, and the C2 number should be read with an uncertainty of at least a few points.
- *Retraining C4:* the same configuration trained twice (the second time after fixing prompt clipping, 160/200 outputs identical) differs by +1.47 [0.12, 3.08]. Code and numerical effects are confounded here, so this is only a rough indication of retraining variability; differences below ≈3 R-L points should not be over-interpreted.
- *Prompt-clipping / parser / metric fixes* changed C1 and C3 by +0.05 and −0.03 R-L (187/200 and 191/200 outputs identical).
- `max_seq_length` 1024 → 2048 (11.3 % of training examples were previously truncated before the summary): first run C4 R-L 42.54, final 42.15, difference −0.4 [−2.1, 1.2]; no evidence that the longer length helps (the 1024 predictions were produced with the older parser).

## A. Extractive vs. abstractive trade-off
| | Ref | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|
| Mean length (words) | 31.5 | 25.0 | 22.6 | 25.8 | 21.6 |
| Predicted 3-grams found in article | 0.636 | 0.249 | 0.520 | 0.323 | **0.817** |
| Predicted unigrams absent from article | 0.174 | 0.351 | 0.167 | 0.326 | **0.091** |
| Outputs ending with "..." | 67 | 0 | 1 | 0 | 21 |

Zero-shot and CoT outputs are strongly abstractive (≈33–35 % novel words, twice the reference rate), which ROUGE penalizes. The fine-tuned model overshoots in the opposite direction (82 % copied 3-grams vs. 64 % in references): it learned the dataset's lead-copying style. A dataset artifact is reproduced: **67/200 references (33.5 %) end with "..."**, and 21 C4 outputs do too, versus none in C1 and C3 (e.g. `24d499ef…`: "HBO’nun yeni dizisi Merteuil (The Seduction) hakkında detaylar..."). Part of C4's ROUGE gain therefore reflects imitation of reference formatting and not better summarization. Few-shot examples already move C2 toward the reference style (52 % copied 3-grams).

## B. Entity / name hallucination and language drift
Samples with ≥1 entity/number absent from the article (prefix matching only for words of ≥4 characters; digits and short words need an exact token): C1 40, C2 16, C3 52, C4 23 (of 200). Examples:
- C4 `e7eca972…`: "Volker Türk, Uluslararası Hukuk Konseyi'nde yaptığı açıklamasında …" — reference: BM İnsan Hakları Yüksek Komiseri Volker Türk; the institution is invented and legal terms are garbled ("ulusal hakları hukukuna").
- C4 `0ef7de7b…`: "Kıbrıs İliçesi’nin (KİP) Bahçeşehir …" — reference: Kiptaş (company name distorted into a non-existent entity); C2 and C3 get the name right.
- C4 `b736c579…`: "Türkiye Cumhuriyeti Cumhuriyet Senatooyu (TCS) Başkanı Prof. Dr. Süheyl Batum …" — invented institution/title.
- C4 `945d4555…`: "İncil Haberleri - …" — invented source prefix.
- C4 `e2b2f1fa…`: summarizes a different game of the same article (Houston Rockets–Brooklyn Nets) instead of the Clippers–Jazz game in the reference; C2 now gets the game right.
- **Language drift:** CJK characters appear in 8 (C1), 2 (C2), 5 (C3) and **1 (C4)** outputs, e.g. "…sesi播出未定。" or an entirely Chinese summary (C1, `a90aaeb7…`); the single C4 case is `a0997ec7…` ("…Ana Haber播出。"). Fine-tuning nearly eliminates code-switching but not completely.
Caveat: the heuristic (capitalized non-initial words and digit groups) misses role/relation errors, and C2's low rate is partly due to its short outputs.

## C. Degenerate generation: repetition and budget truncation
From raw (pre-parser) outputs: word-level loops (a trigram ≥3×) C1 4, C2 5, C3 3, C4 2; repeated sentences C1 2, C2 5, C3 2, C4 2 (of 200). Loops are rare (≤2.5 %), e.g. C3 `945d4555…`: "… isimli isimli isimli …". The dominant failure is **truncation at `max_new_tokens=80`**: raw outputs without terminal punctuation C1 176 (88 %), C2 114 (57 %), **C3 188 (94 %)**, C4 60 (30 %). For C3, 30/200 (15 %) never reach the "Nihai özet" line because the analysis consumes the budget; this is consistent with C3 being indistinguishable from C1 and means C3 is a *budget-limited* result, not evidence that CoT cannot help. A larger budget (e.g., 160 tokens) is the obvious ablation.

## D. Superficial overlap vs. faithfulness (metric disagreement)
- With the final few-shot examples C2 beats C1 on every metric (R-L 28.1 vs 21.5, chrF++ 29.5 vs 26.9), so the ROUGE-vs-chrF++ disagreement seen with the first example choice (17-word outputs) does not persist.
- C4's gain over C2 (+14 R-L) comes with a *higher* unsupported-entity rate (5.05 % vs 3.98 %) and with fabricated institutions/titles (B), so overlap does not certify factuality.
- Many ROUGE misses are morphological or paraphrastic (yakalandı / yakalandığını; "motor süren" / "motosiklet sürüp") and are scored as 0 although correct.
- Source noise: some articles contain mojibake (ı→ý, ş→þ), URLs and update timestamps that also appear in gold summaries.
Recommendation: report ROUGE together with BERTScore/chrF++ and a manual faithfulness check on a sample.

## Limitations
1. Single seed, 200 test samples, one model size: differences below ≈3 R-L points are not reliable (see CIs and the sensitivity section above).
2. The parser (no split after initials such as "M.A." and abbreviations such as "Dr."; markdown asterisks removed; bullet lines dropped only for C3) still splits after ordinal numbers ("6. galibiyet"), e.g. C2 `e2b2f1fa…` formerly ended at "…Clippers'ın 6."
3. BERTScore uses `dbmdz/bert-base-turkish-cased` with `num_layers=9` (uncalibrated for this model) and no baseline rescaling; use for ranking, not absolute values.
4. The unsupported-entity metric is a heuristic; the 4-character prefix rule still accepts some wrong matches.
5. C3 is limited by the 80-token budget (see C).
6. C2 depends on the three examples (see sensitivity); the examples are random, not tuned, and do not appear in the training set.

## Train/test leakage check (`python -m scripts.check_leakage`, subset scores via `scripts.error_analysis --leakage`)
Test (200) vs. the 1000 training articles: 0 exact duplicates, 0 shared first-200-character prefixes, **13/200 near-duplicates** (5-gram Jaccard ≥ 0.5). Reference summaries: 1 exact duplicate. Pairs were classified manually from the first ~150 characters: **3 template pages** ("Kim Kime Dum Duma - <date> …" cartoon pages whose reference is the page title and whose text is site navigation), **9 reports of the same event by other outlets**, **1 topical overlap** ("sarı serum").

Mean ROUGE-L (own implementation, %), removing groups from the 200 test samples:

| Cond. | All 200 | − template (3) | − same-event (9) | − all 13 | Flagged 13 only | Same-event 9 only |
|---|---|---|---|---|---|---|
| C1 | 21.5 | 21.9 | 21.6 | 22.0 | 14.2 | 20.0 |
| C2 | 28.1 | 28.6 | 27.5 | 28.0 | 29.8 | 40.9 |
| C3 | 21.6 | 22.0 | 21.7 | 22.1 | 15.0 | 19.9 |
| C4 | 42.1 | 41.3 | 41.9 | **41.0** | 58.2 | 47.1 |

Conclusion: of C4's ≈1.1-point inflation by the 13 flagged articles, ≈0.8 comes from the three template pages (C4 copies the title: 100.0 ROUGE-L, C1–C3 0.0), which is a formatting artefact and not content memorization; same-event overlap changes C4 by 0.2 points and C2 by 0.6, so there is no evidence that C4 memorized training stories. The ranking and the gaps are unchanged without the flagged articles (C4−C2 ≈ +13 R-L). The official results use all 200 samples; the subset table is a sensitivity analysis. The check detects surface similarity only and does not catch paraphrased duplicates.
