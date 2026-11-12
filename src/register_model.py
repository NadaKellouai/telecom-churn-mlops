from pathlib import Path
import json
import os
import subprocess
import sys

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from train import evaluate


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
MODEL_NAME = "telecom-churn-random-forest"


def main():
    # Refuser l'enregistrement si le quality gate échoue.
    subprocess.run(
        [sys.executable, str(ROOT / "src" / "check_quality.py")],
        check=True,
        cwd=ROOT,
    )

    metadata = json.loads(
        (ARTIFACTS / "model_metadata.json").read_text(
            encoding="utf-8"
        )
    )

    saved_metrics = json.loads(
        (ARTIFACTS / "validation_metrics.json").read_text(
            encoding="utf-8"
        )
    )

    # Charger uniquement notre propre artefact local.
    model = joblib.load(ARTIFACTS / "model.joblib")

    validation = pd.read_csv(
        ROOT / "data" / "prepared" / "validation.csv"
    )
    X_val = validation[metadata["features"]]
    y_val = validation["churn"]

    threshold = float(metadata["threshold"])

    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Seuil invalide.")

    # Vérifier que le modèle sauvegardé correspond aux métriques.
    actual_metrics = evaluate(model, X_val, y_val, 0.5)
    expected_metrics = saved_metrics["random_forest"]

    for metric, value in actual_metrics.items():
        if not np.isclose(
            value,
            expected_metrics[metric],
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(
                f"Modèle et métriques incohérents : {metric}"
            )

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    )
    mlflow.set_experiment("telecom-churn")

    with mlflow.start_run(run_name="register_candidate"):
        mlflow.log_params({
            "decision_threshold": threshold,
            "training_run_id": metadata["mlflow_run_id"],
        })

        mlflow.log_metrics({
            f"val_{key}": value
            for key, value in actual_metrics.items()
        })

        mlflow.log_dict(metadata, "model_metadata.json")

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=X_val.head(3),
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        # Vérifier la sauvegarde avant d'inscrire une version.
        restored = mlflow.sklearn.load_model(
            model_info.model_uri
        )

        np.testing.assert_allclose(
            restored.predict_proba(X_val),
            model.predict_proba(X_val),
            rtol=1e-12,
            atol=1e-12,
        )

        version = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=MODEL_NAME,
        )

    client = MlflowClient()

    client.set_model_version_tag(
        name=MODEL_NAME,
        version=version.version,
        key="decision_threshold",
        value=str(threshold),
    )

    client.set_model_version_tag(
        name=MODEL_NAME,
        version=version.version,
        key="status",
        value="candidate",
    )

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="candidate",
        version=version.version,
    )

    print(f"\nModèle enregistré : {MODEL_NAME}")
    print(f"Version : {version.version}")
    print(f"Seuil : {threshold}")
    print("Rechargement vérifié.")
    print("\nPour lancer cette version dans PowerShell :")
    print(f'$env:MODEL_VERSION = "{version.version}"')
    print(
        "python -m uvicorn app:app "
        "--host 127.0.0.1 --port 8000"
    )


if __name__ == "__main__":
    main()