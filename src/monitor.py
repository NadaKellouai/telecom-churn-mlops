from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "logs" / "requests.jsonl"


def main():
    if not LOG_PATH.exists():
        raise FileNotFoundError(
            "Journal absent : lance l'API et envoie des requêtes."
        )

    events = [
        json.loads(line)
        for line in LOG_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if not events:
        print("Le journal est vide.")
        return

    logs = pd.DataFrame(events)

    # Analyser uniquement les appels de prédiction
    predictions = logs.loc[
        (logs["path"] == "/predict")
        & (logs["method"] == "POST")
    ].copy()

    if predictions.empty:
        print("Aucun appel POST /predict enregistré.")
        return

    total = len(predictions)
    success = predictions.loc[
        predictions["status_code"].between(200, 299)
    ]

    print("MONITORING POST /predict")
    print("Périmètre : fichier requests.jsonl courant")
    print(f"Nombre de requêtes : {total}")
    print(f"Prédictions réussies : {len(success)}")

    print(
        "Taux de rejets de validation (422) : "
        f"{100 * predictions['validation_error'].mean():.2f} %"
    )

    print(
        "Taux d'erreurs serveur (5xx) : "
        f"{100 * predictions['server_error'].mean():.2f} %"
    )

    if not success.empty:
        durations = success["duration_ms"]

        print("\nLatence des prédictions réussies :")
        print(f"Moyenne : {durations.mean():.3f} ms")
        print(f"P95 : {durations.quantile(0.95):.3f} ms")

        if len(success) < 20:
            print(
                "Petit échantillon : latences descriptives, "
                "insuffisantes pour évaluer la performance."
            )


if __name__ == "__main__":
    main()