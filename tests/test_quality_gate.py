import json

import pytest

from src import check_quality


def write_metrics(tmp_path, monkeypatch, candidate_ap):
    metrics = {
        "baseline": {
            "average_precision": 0.15
        },
        "regression_logistique": {
            "average_precision": 0.75
        },
        "random_forest": {
            "average_precision": candidate_ap
        },
    }

    path = tmp_path / "metrics.json"
    path.write_text(
        json.dumps(metrics),
        encoding="utf-8",
    )

    monkeypatch.setattr(check_quality, "METRICS_PATH", path)


def test_accepts_candidate_above_reference(tmp_path, monkeypatch):
    write_metrics(tmp_path, monkeypatch, candidate_ap=0.85)

    # Ne doit pas provoquer d'erreur
    check_quality.main()


def test_rejects_candidate_below_reference(tmp_path, monkeypatch):
    write_metrics(tmp_path, monkeypatch, candidate_ap=0.60)

    with pytest.raises(SystemExit) as error:
        check_quality.main()

    assert error.value.code == 1


def test_rejects_invalid_metric(tmp_path, monkeypatch):
    write_metrics(tmp_path, monkeypatch, candidate_ap=1.20)

    with pytest.raises(SystemExit) as error:
        check_quality.main()

    assert error.value.code == 1