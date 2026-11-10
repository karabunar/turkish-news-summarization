"""Hugging Face adapter: 4-bit NF4 loading, batched generation and QLoRA training."""
import os
from typing import Dict, List, Optional, Sequence

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

from application.ports import Message

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ASSISTANT_TEMPLATE = "<|im_start|>assistant\n"


class HuggingFaceModelAdapter:
    def __init__(self, model_name: str = DEFAULT_MODEL, seed: int = 42, max_batch_tokens: int = 4096):
        self.max_batch_tokens = max_batch_tokens
        self.model_name = model_name
        self.seed = seed
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = self._load_base_model()

    def _load_base_model(self):
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        )
        return AutoModelForCausalLM.from_pretrained(
            self.model_name, quantization_config=bnb, device_map={"": 0},
            torch_dtype=torch.bfloat16,
        )

    def _render(self, conversation: List[Message], add_generation_prompt: bool) -> str:
        return self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=add_generation_prompt)

    def _make_batches(self, prompts: List[str], batch_size: int, beams: int = 1) -> List[List[int]]:
        """Group prompt indices (sorted by length) so batch_size * max_len stays within the token budget.

        Prefill materializes fp32 logits for every prompt token (vocab ~152k), so long
        few-shot prompts need smaller batches to fit in 16 GB. Beam search expands every prompt `beams` times in the
        prefill, so the token budget is divided by the number of beams.
        """
        budget = self.max_batch_tokens // max(beams, 1)
        lengths = [len(self.tokenizer(p)["input_ids"]) for p in prompts]
        order = sorted(range(len(prompts)), key=lambda i: lengths[i])
        batches: List[List[int]] = []
        current: List[int] = []
        for idx in order:
            candidate_len = lengths[idx] * (len(current) + 1)  # sorted ascending: idx is the longest
            if current and (len(current) >= batch_size or candidate_len > budget):
                batches.append(current)
                current = []
            current.append(idx)
        if current:
            batches.append(current)
        return batches

    @torch.no_grad()
    def _generate_all(self, conversations: Sequence[List[Message]], max_new_tokens: int, batch_size: int,
                      gen_kwargs: Optional[Dict], num_return: int) -> List[List[str]]:
        self.model.eval()
        decoding = {"do_sample": False, "temperature": None, "top_p": None, "top_k": None, **(gen_kwargs or {})}
        if num_return > 1:
            decoding["num_return_sequences"] = num_return
        prompts = [self._render(c, True) for c in conversations]
        outputs: List[List[str]] = [[""] * num_return for _ in prompts]
        done = 0
        for indices in self._make_batches(prompts, batch_size, decoding.get("num_beams") or 1):
            enc = self.tokenizer([prompts[i] for i in indices], return_tensors="pt",
                                 padding=True).to(self.model.device)
            # Autocast keeps dtypes consistent after kbit preparation upcast norms to fp32.
            with torch.autocast("cuda", dtype=torch.bfloat16):
                generated = self.model.generate(
                    **enc, max_new_tokens=max_new_tokens, **decoding,
                    pad_token_id=self.tokenizer.pad_token_id, use_cache=True,
                )
            new_tokens = generated[:, enc["input_ids"].shape[1]:]
            texts = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for position, i in enumerate(indices):  # the k candidates of one prompt are consecutive rows
                outputs[i] = texts[position * num_return:(position + 1) * num_return]
            del enc, generated, new_tokens
            torch.cuda.empty_cache()
            done += len(indices)
            print(f"[INFO] Generated {done}/{len(prompts)}")
        return outputs

    def generate(self, conversations: Sequence[List[Message]], max_new_tokens: int = 80,
                 batch_size: int = 8, gen_kwargs: Optional[Dict] = None) -> List[str]:
        """Greedy by default; `gen_kwargs` (e.g. num_beams, min_new_tokens) override the decoding settings for ablations."""
        return [c[0] for c in self._generate_all(conversations, max_new_tokens, batch_size, gen_kwargs, 1)]

    def generate_candidates(self, conversations: Sequence[List[Message]], max_new_tokens: int, batch_size: int,
                            num_candidates: int, gen_kwargs: Optional[Dict] = None) -> List[List[str]]:
        """Top `num_candidates` outputs per prompt, best first (needs num_beams >= num_candidates)."""
        return self._generate_all(conversations, max_new_tokens, batch_size, gen_kwargs, num_candidates)

    def fine_tune(self, conversations: Sequence[List[Message]], output_dir: str,
                  eval_conversations: Sequence[List[Message]] = ()) -> str:
        """QLoRA fine-tuning: r=16, alpha=32, q/k/v/o projections, 3 epochs, SFTTrainer.

        With `eval_conversations` the validation loss is computed after every epoch and the adapter of the epoch with
        the lowest validation loss is kept.
        """
        print("\n[INFO] Starting QLoRA fine-tuning")
        dataset = Dataset.from_dict({"text": [self._render(c, False) for c in conversations]})
        eval_dataset = (Dataset.from_dict({"text": [self._render(c, False) for c in eval_conversations]})
                        if eval_conversations else None)
        selection = ({"eval_strategy": "epoch", "save_strategy": "epoch", "save_total_limit": 2,
                      "load_best_model_at_end": True, "metric_for_best_model": "eval_loss",
                      "greater_is_better": False, "per_device_eval_batch_size": 2}
                     if eval_dataset is not None else {"save_strategy": "no"})
        model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=True)
        lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                          task_type="CAUSAL_LM")
        args = SFTConfig(
            output_dir=output_dir, num_train_epochs=3, per_device_train_batch_size=2,
            gradient_accumulation_steps=8, learning_rate=2e-4, lr_scheduler_type="cosine",
            warmup_ratio=0.05, logging_steps=5, bf16=True, **selection,
            gradient_checkpointing=True, optim="paged_adamw_8bit", seed=self.seed,
            dataset_text_field="text", max_seq_length=2048, packing=False, report_to="none",
        )
        # Loss is computed on the assistant (summary) tokens only.
        collator = DataCollatorForCompletionOnlyLM(
            response_template=self.tokenizer.encode(ASSISTANT_TEMPLATE, add_special_tokens=False),
            tokenizer=self.tokenizer)
        self.tokenizer.padding_side = "right"
        try:
            trainer = SFTTrainer(model=model, args=args, train_dataset=dataset, eval_dataset=eval_dataset,
                                 peft_config=lora, tokenizer=self.tokenizer, data_collator=collator)
            trainer.model.print_trainable_parameters()
            trainer.train()
            trainer.model.save_pretrained(output_dir)
        finally:
            self.tokenizer.padding_side = "left"  # generation assumes left padding even if training fails
        self.model = trainer.model
        self.model.config.use_cache = True
        print(f"[INFO] LoRA adapter saved to {output_dir}")
        return output_dir

    def load_adapter(self, adapter_dir: str) -> None:
        """Attach a previously trained adapter (used with --skip-training)."""
        if not os.path.isfile(os.path.join(adapter_dir, "adapter_config.json")):
            raise FileNotFoundError(f"No LoRA adapter found in {adapter_dir}; train first or pass --adapter-dir.")
        self.model = PeftModel.from_pretrained(self.model, adapter_dir)
