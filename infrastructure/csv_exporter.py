"""CSV exporter for per-sample predictions."""
from typing import Dict, List, Optional, Sequence

import pandas as pd

from domain.entities import (Condition, NewsArticle, RougeMetricResult,
                             SupplementaryMetricResult)

COLUMN_NAMES = {
    Condition.C1_ZERO_SHOT: "pred_C1_ZeroShot",
    Condition.C2_FEW_SHOT: "pred_C2_FewShot",
    Condition.C3_COT: "pred_C3_CoT",
    Condition.C4_LORA: "pred_C4_LoRA",
}


LONG_COLUMNS = ["sample_id", "condition", "prediction", "reference_summary", "news_text"]


def read_predictions(path: str) -> pd.DataFrame:
    """Read a predictions CSV and return it in wide form (one row per sample:
    sample_id, news_text, reference_summary, pred_<condition>...), in the original sample order.
    Accepts the long format written by CsvResultExporter.export and the older wide format."""
    frame = pd.read_csv(path, keep_default_na=False)
    if "condition" not in frame.columns:
        return frame
    order = list(dict.fromkeys(frame["sample_id"]))
    first = frame.drop_duplicates("sample_id").set_index("sample_id").loc[order]
    wide = pd.DataFrame({"sample_id": order,
                         "news_text": first["news_text"].tolist(),
                         "reference_summary": first["reference_summary"].tolist()})
    for condition in dict.fromkeys(frame["condition"]):
        part = frame[frame["condition"] == condition].set_index("sample_id")["prediction"]
        wide[f"pred_{condition}"] = part.reindex(order).tolist()
    return wide


class CsvResultExporter:
    def export(self, articles: Sequence[NewsArticle],
               predictions: Dict[Condition, List[str]], path: str) -> None:
        """Long format: one row per (sample, condition)."""
        rows = []
        for condition, column in COLUMN_NAMES.items():
            for article, prediction in zip(articles, predictions[condition]):
                rows.append({"sample_id": article.id, "condition": column[len("pred_"):],
                             "prediction": prediction,
                             "reference_summary": article.reference_summary,
                             "news_text": article.article_text})
        pd.DataFrame(rows, columns=LONG_COLUMNS).to_csv(path, index=False, encoding="utf-8")

    def export_metrics(self, rouge: Dict[Condition, RougeMetricResult],
                       supplementary: Optional[Dict[Condition, SupplementaryMetricResult]],
                       n_samples: int, path: str) -> None:
        """One row per condition; all scores in percent."""
        rows = []
        for condition, r in rouge.items():
            row = {"condition": condition.value, "n_test_samples": n_samples,
                   "rouge1": r.rouge1 * 100, "rouge2": r.rouge2 * 100,
                   "rougeL": r.rougeL * 100, "rougeLsum": r.rougeLsum * 100}
            if supplementary is not None:
                s = supplementary[condition]
                row.update({"chrf_pp": s.chrf_pp, "bertscore_f1": s.bertscore_f1 * 100,
                            "unsupported_entity_rate": s.unsupported_entity_rate * 100})
            rows.append(row)
        pd.DataFrame(rows).round(4).to_csv(path, index=False)

    def export_raw(self, summaries: Dict[Condition, list], path: str) -> None:
        """Unprocessed model outputs (before SummaryParserService), for error analysis."""
        frame = pd.DataFrame({"sample_id": [s.sample_id for s in next(iter(summaries.values()))]})
        for condition, items in summaries.items():
            frame[f"raw_{COLUMN_NAMES[condition][5:]}"] = [s.raw_text for s in items]
        frame.to_csv(path, index=False, encoding="utf-8")
