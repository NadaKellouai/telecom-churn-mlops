from pathlib import Path
import hashlib
import json
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "prepared"
OUTPUT_DIR = ROOT / "artifacts"

# Choix déjà fixés dans le notebook
THRESHOLD = 0.3
CATEGORICAL = ["complains", "age_group", "tariff_plan"]


def evaluate(model, X, y, threshold):
    scores = model.predict_proba(X)[:, 1]
    predictions = (scores >= threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "average_precision": float(
            average_precision_score(y, scores)
        ),
        "roc_auc": float(roc_auc_score(y, scores)),
    }


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    validation = pd.read_csv(DATA_DIR / "validation.csv")

    X_train = train.drop(columns="churn")
    y_train = train["churn"]
    X_val = validation.drop(columns="churn")
    y_val = validation["churn"]

    numeric = [
        column
        for column in X_train.columns
        if column not in CATEGORICAL
    ]

    logistic_preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            CATEGORICAL,
        ),
    ])

    forest_preprocessor = ColumnTransformer([
        ("numeric", "passthrough", numeric),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            CATEGORICAL,
        ),
    ])

    models = {
        "baseline": DummyClassifier(strategy="prior"),
        "regression_logistique": Pipeline([
            ("preprocessing", logistic_preprocessor),
            ("classifier", LogisticRegression(
                max_iter=2000,
                random_state=42,
            )),
        ]),
        "random_forest": Pipeline([
            ("preprocessing", forest_preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
    }

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    )
    mlflow.set_experiment("telecom-churn")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data_hashes = {
        name: hashlib.sha256(
            (DATA_DIR / f"{name}.csv").read_bytes()
        ).hexdigest()
        for name in ["train", "validation"]
    }

    results = {}
    forest_run_id = None

    for name, model in models.items():
        with mlflow.start_run(run_name=f"pipeline_{name}") as run:
            mlflow.set_tags({
                "execution": "script",
                "dataset": "UCI Iranian Churn",
            })

            mlflow.log_params({
                "n_train": len(train),
                "n_validation": len(validation),
                "comparison_threshold": 0.5,
                "train_sha256": data_hashes["train"],
                "validation_sha256": data_hashes["validation"],
            })

            estimator = (
                model.named_steps["classifier"]
                if hasattr(model, "named_steps")
                else model
            )

            mlflow.log_params({
                f"model_{key}": value
                for key, value in estimator.get_params(
                    deep=False
                ).items()
            })

            model.fit(X_train, y_train)

            metrics = evaluate(model, X_val, y_val, 0.5)
            results[name] = metrics

            mlflow.log_metrics({
                f"val_{key}": value
                for key, value in metrics.items()
            })

            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                input_example=X_train.head(3),
                skops_trusted_types=(
                    ["sklearn.tree._tree.Tree"]
                    if name == "random_forest"
                    else []
                ),
            )

            if name == "random_forest":
                forest_run_id = run.info.run_id

                selected_metrics = evaluate(
                    model, X_val, y_val, THRESHOLD
                )
                results["random_forest_threshold_0.3"] = (
                    selected_metrics
                )

                mlflow.log_param("selected_threshold", THRESHOLD)
                mlflow.log_metrics({
                    f"val_selected_{key}": value
                    for key, value in selected_metrics.items()
                })

                # Pipeline complet destiné aux étapes suivantes
                joblib.dump(
                    model,
                    OUTPUT_DIR / "model.joblib",
                )

            print(
                f"{name} : validation AP = "
                f"{metrics['average_precision']:.4f}"
            )

    metadata = {
        "model": "random_forest",
        "threshold": THRESHOLD,
        "features": X_train.columns.tolist(),
        "mlflow_run_id": forest_run_id,
        "data_sha256": data_hashes,
        "selection": "Choices fixed in the initial notebook",
    }

    (OUTPUT_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    (OUTPUT_DIR / "validation_metrics.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\nEntraînement terminé. Artefacts dans artifacts/.")


if __name__ == "__main__":
    main()