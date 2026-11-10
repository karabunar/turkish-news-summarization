"""Does merging the LoRA adapter into the base weights change the C4 scores?

Generates C4 summaries for the whole test split (same prompt, greedy, 80 new tokens, batch 8, 2-sentence parser as
in main.py) with
  adapter-4bit   unmerged adapter on the 4-bit base (the reported setting; regenerated here, because greedy outputs
                 are not bit-exact across runs)
  merged-4bit    adapter merged into the dequantized 4-bit weights (requantized)
  merged-bf16    adapter merged into a bf16 base (no quantization)
and scores ROUGE and the unsupported entity/number rate. Needs a GPU and a trained adapter.

Usage (project root, inside the container):
    python -m scripts.merged_quality --adapter-dir outputs/lora_adapter
"""
import argparse
import gc
import os

import pandas as pd


def main() -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    from scripts.merge_benchmark import share_identical
    from application.use_cases import RunPromptSummarizationUseCase
    from domain.entities import Condition
    from domain.services.evaluation_service import EvaluationService
    from domain.services.prompt_builder import PromptBuilder
    from domain.services.summary_parser import SummaryParserService
    from domain.services.supplementary_metrics import count_unsupported
    from infrastructure.dataset_repository import TurkishNewsDatasetRepository
    from infrastructure.hf_model_adapter import DEFAULT_MODEL, HuggingFaceModelAdapter

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter-dir", default="outputs/lora_adapter")
    ap.add_argument("--n-test", type=int, default=10 ** 9)
    ap.add_argument("--output-dir", default="outputs/kv_cache")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    _, test = TurkishNewsDatasetRepository(42).load_splits(1, args.n_test)
    model = HuggingFaceModelAdapter(args.model, 42)
    runner = RunPromptSummarizationUseCase(model, PromptBuilder(), SummaryParserService(max_sentences=2), 80, 8)
    evaluator = EvaluationService()
    rows, outputs = [], {}

    def run(name: str) -> None:
        print(f"[INFO] {name}")
        texts = [s.generated_text for s in runner.execute(test, Condition.C4_LORA)]
        outputs[name] = texts
        r = evaluator.evaluate(texts, [a.reference_summary for a in test])
        unsupported = total = 0
        for t, a in zip(texts, test):
            u, n = count_unsupported(t, a.article_text)
            unsupported, total = unsupported + u, total + n
        same = (share_identical(texts, outputs["adapter-4bit"]) if "adapter-4bit" in outputs else float("nan"))
        rows.append({"variant": name, "n": len(test), "rouge1": r.rouge1 * 100, "rouge2": r.rouge2 * 100,
                     "rougeL": r.rougeL * 100, "unsupported_pct": 100 * unsupported / total,
                     "same_as_adapter_4bit": same})

    model.load_adapter(args.adapter_dir)
    run("adapter-4bit")
    model.model = model.model.merge_and_unload()
    run("merged-4bit")
    del model.model; gc.collect(); torch.cuda.empty_cache()

    base = AutoModelForCausalLM.from_pretrained(args.model, device_map={"": 0}, torch_dtype=torch.bfloat16)
    model.model = PeftModel.from_pretrained(base, args.adapter_dir).merge_and_unload().eval()
    run("merged-bf16")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.output_dir, "merged_quality.csv"), index=False)
    pd.DataFrame(outputs).assign(sample_id=[a.id for a in test]).to_csv(
        os.path.join(args.output_dir, "merged_quality_predictions.csv"), index=False)
    print("\n" + df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
