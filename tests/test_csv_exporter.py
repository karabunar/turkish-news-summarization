import pandas as pd

from domain.entities import Condition, RougeMetricResult, SupplementaryMetricResult
from infrastructure.csv_exporter import CsvResultExporter


def test_export_metrics_writes_one_row_per_condition_in_percent(tmp_path):
    rouge = {Condition.C1_ZERO_SHOT: RougeMetricResult(0.25, 0.1, 0.2, 0.21),
             Condition.C4_LORA: RougeMetricResult(0.5, 0.4, 0.45, 0.46)}
    extra = {Condition.C1_ZERO_SHOT: SupplementaryMetricResult(26.8, 0.56, 0.07),
             Condition.C4_LORA: SupplementaryMetricResult(37.6, 0.67, 0.03)}
    path = tmp_path / "metrics.csv"
    CsvResultExporter().export_metrics(rouge, extra, 200, str(path))
    frame = pd.read_csv(path)
    assert list(frame["condition"]) == ["C1_ZeroShot", "C4_LoRA"]
    assert frame.loc[1, "rouge1"] == 50.0 and frame.loc[1, "bertscore_f1"] == 67.0
    assert (frame["n_test_samples"] == 200).all()


def test_export_metrics_without_supplementary_omits_those_columns(tmp_path):
    path = tmp_path / "m.csv"
    CsvResultExporter().export_metrics({Condition.C1_ZERO_SHOT: RougeMetricResult(0.1, 0.1, 0.1, 0.1)},
                                       None, 10, str(path))
    assert "chrf_pp" not in pd.read_csv(path).columns
