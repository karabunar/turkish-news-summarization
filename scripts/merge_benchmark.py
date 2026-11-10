"""Does merging the LoRA adapter into the base weights remove its decode-time cost?

Decode speed per generated token (ms) for the C1/C4 prompt (the same prompt, so the KV cache is identical), batch size 1,
greedy, exactly --new-tokens tokens per sample (min_new_tokens = max_new_tokens, so an early EOS cannot change the
number of decoded steps). Prefill (prompt pass + first token) is timed in a separate call, as in kv_cache_benchmark.

Variants, for the 4-bit NF4 base used in the reported runs and for a bf16 base:
  base            plain base model
  adapter         PeftModel, adapter branches executed separately in every layer
  merged          adapter folded into the base weights (merge_and_unload)
For 4-bit weights, merging dequantizes, adds the update and requantizes, so outputs can change slightly;
`same_as_adapter` reports the share of samples whose generated tokens are identical to the unmerged adapter's.

Usage (project root, inside the container, GPU):
    python -m scripts.merge_benchmark --n-samples 30 --adapter-dir outputs/lora_adapter
"""
import argparse
import gc
import os
import time
from typing import Dict, List

import pandas as pd


def decode_ms_per_token(total_s: float, prefill_s: float, new_tokens: int) -> float:
    """Time of the decoding steps after the first token, in ms per token."""
    return 1000 * (total_s - prefill_s) / max(new_tokens - 1, 1)


def share_identical(candidate, reference) -> float:
    """Fraction of positions where the two sequences of generations are equal."""
    return sum(a == b for a, b in zip(candidate, reference)) / len(candidate)


def main() -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from domain.entities import Condition
    from domain.services.prompt_builder import PromptBuilder, split_few_shot_pool
    from infrastructure.dataset_repository import TurkishNewsDatasetRepository
    from infrastructure.hf_model_adapter import DEFAULT_MODEL

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter-dir", default="outputs/lora_adapter")
    ap.add_argument("--n-samples", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--output-dir", default="outputs/kv_cache")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "needs a GPU"
    os.makedirs(args.output_dir, exist_ok=True)

    pool, test = TurkishNewsDatasetRepository(42).load_splits(None, None)
    _, few_shot = split_few_shot_pool(pool, len(pool) - 3, 3)
    builder = PromptBuilder(few_shot_examples=few_shot)
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    texts = [tok.apply_chat_template(builder.build(Condition.C1_ZERO_SHOT, a.article_text), tokenize=False,
                                     add_generation_prompt=True) for a in test[:args.n_samples]]
    nf4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    decoding = {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}

    def load_base(precision: str):
        kwargs = {"quantization_config": nf4} if precision == "4bit" else {}
        return AutoModelForCausalLM.from_pretrained(args.model, device_map={"": 0}, torch_dtype=torch.bfloat16,
                                                    **kwargs).eval()

    @torch.no_grad()
    def measure(model, precision: str, variant: str, reference: Dict[str, List]) -> dict:
        rows, generated = [], []
        for k, text in enumerate(texts[:args.warmup] + texts):
            enc = tok(text, return_tensors="pt").to(model.device)
            kw = dict(pad_token_id=tok.pad_token_id, use_cache=True, **decoding)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                model.generate(**enc, max_new_tokens=1, **kw)
                torch.cuda.synchronize(); prefill = time.perf_counter() - t0
                t0 = time.perf_counter()
                out = model.generate(**enc, max_new_tokens=args.new_tokens, min_new_tokens=args.new_tokens, **kw)
                torch.cuda.synchronize(); total = time.perf_counter() - t0
            if k < args.warmup:
                continue
            generated.append(out[0, enc["input_ids"].shape[1]:].tolist())
            rows.append((prefill, total, decode_ms_per_token(total, prefill, args.new_tokens)))
        df = pd.DataFrame(rows, columns=["prefill_s", "total_s", "decode_ms_per_token"])
        weights_mb = torch.cuda.memory_allocated() / 1024 ** 2
        reference[f"{precision}/{variant}"] = generated
        adapter_out = reference.get(f"{precision}/adapter")
        same = share_identical(generated, adapter_out) if adapter_out else float("nan")
        print(f"[INFO] {precision}/{variant}: decode {df.decode_ms_per_token.mean():.2f} ms/token")
        return {"precision": precision, "variant": variant, "prefill_s": df.prefill_s.mean(),
                "decode_ms_per_token": df.decode_ms_per_token.mean(), "decode_ms_sd": df.decode_ms_per_token.std(),
                "gpu_weights_MB": weights_mb, "same_as_adapter": same}

    summary, reference = [], {}
    for precision in ("4bit", "bf16"):
        model = load_base(precision)
        summary.append(measure(model, precision, "base", reference))
        model = PeftModel.from_pretrained(model, args.adapter_dir).eval()
        summary.append(measure(model, precision, "adapter", reference))
        model = model.merge_and_unload().eval()
        summary.append(measure(model, precision, "merged", reference))
        del model; gc.collect(); torch.cuda.empty_cache()

    df = pd.DataFrame(summary)
    df.to_csv(os.path.join(args.output_dir, "merge_benchmark.csv"), index=False)
    print("\n" + df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
