from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "raw" / "Customer Churn.csv"
OUTPUT_DIR = ROOT / "data" / "prepared"

FEATURES = [
    "call_failure",
    "complains",
    "subscription_length",
    "charge_amount",
    "seconds_of_use",
    "frequency_of_use",
    "frequency_of_sms",
    "distinct_called_numbers",
    "age_group",
    "tariff_plan",
]

SEED = 42


def main():
    df = pd.read_csv(INPUT_PATH)

    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )

    required = FEATURES + ["churn"]
    missing = sorted(set(required) - set(df.columns))

    if missing:
        raise ValueError(f"Colonnes absentes : {missing}")

    if df[required].isna().any().any():
        raise ValueError("Valeurs manquantes dans les données.")

    if not df["churn"].isin([0, 1]).all():
        raise ValueError("La cible churn doit contenir uniquement 0 ou 1.")

    X = df[FEATURES].copy()
    y = df["churn"].copy()

    # Ne jamais séparer des profils identiques entre les ensembles.
    groups = X.groupby(
        FEATURES, dropna=False, sort=False
    ).ngroup()

    split_test = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    dev_idx, test_idx = next(split_test.split(X, y, groups))

    X_dev = X.iloc[dev_idx]
    y_dev = y.iloc[dev_idx]
    groups_dev = groups.iloc[dev_idx]

    split_val = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=SEED,
    )

    train_local, val_local = next(
        split_val.split(X_dev, y_dev, groups_dev)
    )

    # Convertir les positions relatives en positions du dataset initial.
    train_idx = dev_idx[train_local]
    val_idx = dev_idx[val_local]

    partitions = {
        "train": train_idx,
        "validation": val_idx,
        "test": test_idx,
    }

    # Contrôler l'absence de profils partagés.
    group_sets = {
        name: set(groups.iloc[indices])
        for name, indices in partitions.items()
    }

    for left, right in [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]:
        if not group_sets[left].isdisjoint(group_sets[right]):
            raise ValueError(f"Profils partagés : {left} / {right}")

    all_indices = [
        int(index)
        for indices in partitions.values()
        for index in indices
    ]

    if sorted(all_indices) != list(range(len(df))):
        raise ValueError("Certaines lignes sont absentes ou répétées.")

    for name, indices in partitions.items():
        if y.iloc[indices].nunique() != 2:
            raise ValueError(f"Une classe est absente de {name}.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "random_state": SEED,
        "features": FEATURES,
        "indices": {},
    }

    for name, indices in partitions.items():
        subset = df.iloc[indices][required]
        subset.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)

        manifest["indices"][name] = [
            int(index) for index in indices
        ]

        print(
            f"{name}: {len(subset)} clients, "
            f"churn = {100 * subset['churn'].mean():.2f} %"
        )

    (OUTPUT_DIR / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\nPréparation terminée : aucun profil partagé.")


if __name__ == "__main__":
    main()