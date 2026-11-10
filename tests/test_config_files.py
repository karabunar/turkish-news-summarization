"""The documented prompts and configuration (config/*.json) must match the values the code actually uses."""
import ast
import inspect
import json
from pathlib import Path

from domain.services import prompt_builder
from domain.services.prompt_builder import PromptBuilder
from domain.services.summary_parser import SummaryParserService
from domain.services.supplementary_metrics import DEFAULT_BERT_MODEL, SupplementaryEvaluationService

ROOT = Path(__file__).resolve().parent.parent


def _load(name):
    return json.loads((ROOT / "config" / name).read_text(encoding="utf-8"))


def _call_keywords(path, func_name):
    """Constant keyword arguments of the first call to `func_name` in a source file (no import needed)."""
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) == func_name:
            out = {}
            for kw in node.keywords:
                try:
                    out[kw.arg] = ast.literal_eval(kw.value)
                except (ValueError, SyntaxError):
                    pass
            return out
    raise AssertionError(f"{func_name} not found in {path}")


def _argparse_defaults(path):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    defaults = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
            flag = ast.literal_eval(node.args[0])
            for kw in node.keywords:
                if kw.arg == "default":
                    try:
                        defaults[flag] = ast.literal_eval(kw.value)
                    except (ValueError, SyntaxError):
                        pass
    return defaults


def test_prompts_file_matches_the_prompt_builder_constants():
    prompts = _load("prompts.json")
    assert prompts["c1_c4_instruction"] == prompt_builder.ZERO_SHOT_INSTRUCTION
    assert prompts["c3_persona"] == prompt_builder.COT_PERSONA
    assert prompts["c3_format_hint"] == prompt_builder.COT_FORMAT_HINT
    assert prompts["c3_final_summary_marker"] == prompt_builder.FINAL_SUMMARY_MARKER
    params = inspect.signature(PromptBuilder.__init__).parameters
    assert prompts["max_article_chars"] == params["max_article_chars"].default
    assert prompts["max_few_shot_example_chars"] == params["max_example_chars"].default


def test_prompts_file_describes_the_messages_the_builder_really_builds():
    prompts = _load("prompts.json")
    builder = PromptBuilder()
    c1 = builder.build(prompt_builder.Condition.C1_ZERO_SHOT, "ARTICLE")
    assert c1[0]["content"] == prompts["c1_c4_instruction"] + "\n\nARTICLE"
    c3 = builder.build(prompt_builder.Condition.C3_COT, "ARTICLE")
    assert c3[0]["content"] == prompts["c3_persona"] + "\n" + prompts["c3_format_hint"] + "\n\nARTICLE"


def test_lora_and_training_hyperparameters_match_the_adapter_source():
    cfg = _load("experiment_config.json")["qlora"]
    lora = _call_keywords("infrastructure/hf_model_adapter.py", "LoraConfig")
    for key, value in cfg["lora"].items():
        assert lora[key] == value, key
    training = _call_keywords("infrastructure/hf_model_adapter.py", "SFTConfig")
    for key, value in cfg["training"].items():
        assert training[key] == value, key


def test_decoding_and_seed_defaults_match_main_and_the_parser():
    cfg = _load("experiment_config.json")
    defaults = _argparse_defaults("main.py")
    assert defaults["--seed"] == cfg["data"]["seed"]
    assert defaults["--batch-size"] == cfg["decoding"]["batch_size"]
    assert defaults["--max-new-tokens"] == cfg["decoding"]["max_new_tokens"]
    assert inspect.signature(SummaryParserService.__init__).parameters["max_sentences"].default == \
        cfg["decoding"]["parser_max_sentences"]


def test_metric_settings_match_the_supplementary_service():
    metrics = _load("experiment_config.json")["metrics"]
    params = inspect.signature(SupplementaryEvaluationService.__init__).parameters
    assert metrics["bertscore"]["model"] == DEFAULT_BERT_MODEL == params["bert_model"].default
    assert metrics["bertscore"]["num_layers"] == params["bert_num_layers"].default
