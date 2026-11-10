"""LoRA (C4) vs prompting (C1-C3): prompt length, KV-cache size, latency and peak GPU memory at inference.

C4 uses the same prompt as C1, so C1 vs C4 isolates the cost of the (unmerged) LoRA adapter, while C1 vs C2/C3
shows what few-shot and persona+CoT prompts cost in context length and KV cache.

Two parts:
  1. Prompt length and theoretical KV cache (tokenizer + model config only; no GPU; `--tokens-only`).
     KV bytes/token = 2 (K and V) * layers * kv_heads * head_dim * dtype_bytes (bf16: 2).
  2. Measured, batch size 1 (no padding), greedy, same decoding as the main run: latency (prefill and decode separately), tokens/s,
     real KV-cache tensor size from `past_key_values`, and peak GPU memory above the idle model (GPU).

Usage (project root, inside the container):
    python -m scripts.kv_cache_benchmark --tokens-only                       # no GPU needed
    python -m scripts.kv_cache_benchmark --n-samples 50                      # full benchmark (C1-C3 base, C4 adapter)
    python -m scripts.kv_cache_benchmark --n-samples 50 --model Qwen/Qwen2.5-3B-Instruct --adapter-dir experiments/05_3b_full_data/lora_adapter
"""
import argparse
import os
import time
from typing import Dict, List, Sequence

import pandas as pd

from scripts.merge_benchmark import decode_ms_per_token

BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}


def kv_bytes_per_token(num_layers: int, num_kv_heads: int, head_dim: int, dtype_bytes: int = 2) -> int:
    """K and V tensors of every layer for one token."""
    return 2 * num_layers * num_kv_heads * head_dim * dtype_bytes


def kv_bytes_from_config(config, dtype_bytes: int = 2) -> int:
    heads = config.num_attention_heads
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // heads
    kv_heads = getattr(config, "num_key_value_heads", None) or heads
    return kv_bytes_per_token(config.num_hidden_layers, kv_heads, head_dim, dtype_bytes)


def cache_nbytes(past_key_values) -> int:
    """Total size of the K/V tensors in a legacy tuple or a transformers Cache object."""
    if hasattr(past_key_values, "to_legacy_cache"):
        past_key_values = past_key_values.to_legacy_cache()
    return sum(t.numel() * t.element_size() for layer in past_key_values for t in layer)


def mb(n_bytes: float) -> float:
    return n_bytes / 1024 ** 2


def build_prompts(tokenizer, builder, articles, conditions) -> Dict[str, List[str]]:
    return {c.value: [tokenizer.apply_chat_template(builder.build(c, a.article_text), tokenize=False,
                                                    add_generation_prompt=True) for a in articles]
            for c in conditions}


def prompt_token_table(tokenizer, prompts: Dict[str, List[str]], per_token_bytes: int,
                       max_new_tokens: int) -> pd.DataFrame:
    rows = []
    for cond, texts in prompts.items():
        lengths = pd.Series([len(tokenizer(t)["input_ids"]) for t in texts])
        rows.append({"condition": cond, "prompt_tokens_mean": lengths.mean(), "prompt_tokens_max": lengths.max(),
                     "kv_MB_prompt_mean": mb(lengths.mean() * per_token_bytes),
                     "kv_MB_worst_case": mb((lengths.max() + max_new_tokens) * per_token_bytes)})
    return pd.DataFrame(rows)


def main() -> None:
    import torch
    from transformers import AutoConfig, AutoTokenizer
    from domain.entities import Condition
    from domain.services.prompt_builder import PromptBuilder, split_few_shot_pool
    from infrastructure.dataset_repository import TurkishNewsDatasetRepository
    from infrastructure.hf_model_adapter import DEFAULT_MODEL

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter-dir", default="outputs/lora_adapter")
    ap.add_argument("--n-samples", type=int, default=50, help="Test articles used for the measured part")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--tokens-only", action="store_true", help="Only prompt lengths and theoretical KV (no GPU)")
    ap.add_argument("--output-dir", default="outputs/kv_cache")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Same split as main.py --all-data: 3 few-shot examples are the last 3 of the pool; test = whole test split.
    pool, test = TurkishNewsDatasetRepository(42).load_splits(None, None)
    _, few_shot = split_few_shot_pool(pool, len(pool) - 3, 3)
    builder = PromptBuilder(few_shot_examples=few_shot)
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    config = AutoConfig.from_pretrained(args.model)
    per_token = kv_bytes_from_config(config, BYTES["bfloat16"])
    conditions = [Condition.C1_ZERO_SHOT, Condition.C2_FEW_SHOT, Condition.C3_COT, Condition.C4_LORA]

    # Part 1: prompt length and theoretical KV (whole test split).
    table = prompt_token_table(tokenizer, build_prompts(tokenizer, builder, test, conditions),
                               per_token, args.max_new_tokens)
    print(f"\n[INFO] {args.model}: layers={config.num_hidden_layers} kv_heads={config.num_key_value_heads} "
          f"-> KV cache {per_token} bytes ({per_token / 1024:.1f} KB) per token (bf16), {len(test)} test articles")
    print(table.round(1).to_string(index=False))
    table.to_csv(os.path.join(args.output_dir, "prompt_tokens_theoretical_kv.csv"), index=False)
    if args.tokens_only:
        return

    # Part 2: measured, batch size 1.
    from infrastructure.hf_model_adapter import HuggingFaceModelAdapter
    assert torch.cuda.is_available(), "The measured part needs a GPU; use --tokens-only otherwise."
    subset = test[:args.n_samples]
    prompts = build_prompts(tokenizer, builder, subset, conditions)
    model = HuggingFaceModelAdapter(args.model, 42)
    decoding = {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}

    @torch.no_grad()
    def measure(label: str, condition: Condition, with_adapter: bool) -> List[dict]:
        rows = []
        texts = prompts[condition.value]
        for k, text in enumerate(list(texts[:args.warmup]) + list(texts)):
            enc = model.tokenizer(text, return_tensors="pt").to(model.model.device)
            torch.cuda.synchronize(); torch.cuda.empty_cache()
            with torch.autocast("cuda", dtype=torch.bfloat16):  # prefill only: prompt pass + first token
                t0 = time.perf_counter()
                model.model.generate(**enc, max_new_tokens=1, **decoding, pad_token_id=model.tokenizer.pad_token_id,
                                     use_cache=True)
                torch.cuda.synchronize()
                prefill = time.perf_counter() - t0
            torch.cuda.empty_cache()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.model.generate(**enc, max_new_tokens=args.max_new_tokens, **decoding,
                                           pad_token_id=model.tokenizer.pad_token_id, use_cache=True,
                                           return_dict_in_generate=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if k < args.warmup:
                continue
            prompt_len = enc["input_ids"].shape[1]
            new_tokens = out.sequences.shape[1] - prompt_len
            rows.append({"run": label, "condition": condition.value, "adapter": with_adapter,
                         "prompt_tokens": prompt_len, "new_tokens": new_tokens, "seconds": elapsed,
                         "prefill_s": prefill, "decode_ms_per_token": decode_ms_per_token(elapsed, prefill, new_tokens),
                         "tokens_per_s": new_tokens / elapsed,
                         "kv_MB_measured": mb(cache_nbytes(out.past_key_values)),
                         "kv_MB_theory": mb((prompt_len + new_tokens - 1) * per_token),
                         "peak_MB_above_model": mb(torch.cuda.max_memory_allocated() - baseline)})
            del enc, out
        print(f"[INFO] {label} done")
        return rows

    rows: List[dict] = []
    for c in (Condition.C1_ZERO_SHOT, Condition.C2_FEW_SHOT, Condition.C3_COT):
        rows += measure(f"{c.value} (base)", c, False)
    model.load_adapter(args.adapter_dir)  # unmerged adapter on the 4-bit base, as in the reported C4 run
    rows += measure("C4_LoRA (adapter)", Condition.C4_LORA, True)
    with model.model.disable_adapter():  # control: identical C1 prompt on the same PeftModel, adapter bypassed
        rows += measure("C1_ZeroShot (PeftModel, adapter off)", Condition.C1_ZERO_SHOT, False)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.output_dir, "kv_cache_per_sample.csv"), index=False)
    summary = df.groupby("run", sort=False).agg(
        prompt_tokens=("prompt_tokens", "mean"), new_tokens=("new_tokens", "mean"), seconds=("seconds", "mean"),
        prefill_s=("prefill_s", "mean"), decode_ms_per_token=("decode_ms_per_token", "mean"),
        tokens_per_s=("tokens_per_s", "mean"), kv_MB_measured=("kv_MB_measured", "mean"),
        kv_MB_theory=("kv_MB_theory", "mean"), peak_MB_above_model=("peak_MB_above_model", "mean")).round(2)
    summary.to_csv(os.path.join(args.output_dir, "kv_cache_summary.csv"))
    print("\n" + summary.to_string())
    print(f"\n[INFO] adapter file size: "
          f"{os.path.getsize(os.path.join(args.adapter_dir, 'adapter_model.safetensors')) / 1024 ** 2:.1f} MB"
          if os.path.isfile(os.path.join(args.adapter_dir, 'adapter_model.safetensors')) else "")


if __name__ == "__main__":
    main()
