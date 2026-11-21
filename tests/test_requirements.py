import json

from smartattend.requirements import REQUIREMENTS, evaluate, format_table

CFG = {"product": {"detector": "yolov8s"}}


def _write(root, rel, obj):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))


def test_missing_results_are_reported_not_crashing(tmp_path):
    rows = evaluate(tmp_path, CFG)
    assert len(rows) == len(REQUIREMENTS)
    assert all(r["status"] == "NOT RUN" for r in rows)
    assert "stage face" in format_table(rows)


def test_pass_and_fail_respect_direction(tmp_path):
    _write(tmp_path, "reports/eval/pipeline_latency.json", {"total": {"p95": 650.0}, "peak_rss_mb": 700.0})
    rows = {r["id"]: r for r in evaluate(tmp_path, CFG)}
    assert rows["R8"]["status"] == "FAIL"     # 650 ms > 500 ms target (lower is better)
    assert rows["R9"]["status"] == "PASS"     # 700 MB <= 2000 MB
    _write(tmp_path, "reports/experiments/engagement/engagement_cv.json",
           {"resnet18": {"mean": 0.60}, "googlenet": {"mean": 0.70}})
    rows = {r["id"]: r for r in evaluate(tmp_path, CFG)}
    assert rows["R7"]["status"] == "PASS" and rows["R7"]["measured"] == 0.70   # best backbone, higher is better


def test_malformed_result_is_an_error(tmp_path):
    _write(tmp_path, "reports/eval/pipeline_latency.json", {"unexpected": 1})
    rows = {r["id"]: r for r in evaluate(tmp_path, CFG)}
    assert rows["R8"]["status"] == "ERROR"
