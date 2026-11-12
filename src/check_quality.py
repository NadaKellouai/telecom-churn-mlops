from pathlib import Path
import json
import math
import sys


ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "artifacts" / "validation_metrics.json"


def main():
    metrics = json.loads(
        METRICS_PATH.read_text(encoding="utf-8")
    )

    # Vérifier que toutes les métriques sont finies et dans [0, 1].
    errors = []

    for model_name, model_metrics in metrics.items():
        for metric_name, value in model_metrics.items():
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                errors.append(
                    f"Métrique invalide : "
                    f"{model_name}.{metric_name} = {value}"
                )

    if errors:
        for error in errors:
            print(f"ECHEC : {error}")
        sys.exit(1)

    baseline_ap = metrics["baseline"]["average_precision"]
    reference_ap = metrics[
        "regression_logistique"
    ]["average_precision"]
    candidate_ap = metrics[
        "random_forest"
    ]["average_precision"]

    checks = [
        (
            "Le candidat dépasse la baseline",
            candidate_ap > baseline_ap,
        ),
        (
            "Le candidat égale ou dépasse la régression logistique",
            candidate_ap >= reference_ap,
        ),
    ]

    print(f"Baseline AP : {baseline_ap:.4f}")
    print(f"Régression logistique AP : {reference_ap:.4f}")
    print(f"Random Forest AP : {candidate_ap:.4f}")

    for description, passed in checks:
        status = "OK" if passed else "ECHEC"
        print(f"{status} : {description}")

    if not all(passed for _, passed in checks):
        print("\nQuality gate refusé.")
        sys.exit(1)

    print("\nQuality gate accepté.")


if __name__ == "__main__":
    main()