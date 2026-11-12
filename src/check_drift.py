from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "prepared" / "train.csv"

COLUMN = "seconds_of_use"
SEED = 42


def distribution_distance(reference, current, bins):
    """Distance de variation totale entre deux histogrammes."""
    reference_counts, _ = np.histogram(reference, bins=bins)
    current_counts, _ = np.histogram(current, bins=bins)

    reference_freq = reference_counts / reference_counts.sum()
    current_freq = current_counts / current_counts.sum()

    return float(
        0.5 * np.abs(reference_freq - current_freq).sum()
    )


def main():
    train = pd.read_csv(TRAIN_PATH)
    values = train[COLUMN].to_numpy()

    # Deux lots distincts, issus uniquement de l'entraînement.
    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(values)

    reference = shuffled[:900]
    normal_batch = shuffled[900:1800]

    # Simulation : baisse de 80 % de l'usage.
    drifted_batch = np.floor(normal_batch * 0.2).astype(int)

    # Bornes définies à partir de la référence.
    inner_edges = np.unique(
        np.quantile(reference, np.linspace(0, 1, 11))
    )
    bins = np.concatenate((
        [-np.inf],
        inner_edges,
        [np.inf],
    ))

    # Seuil empirique : variabilité attendue sous une distribution
    # inchangée, estimée par rééchantillonnage de la référence.
    null_distances = []

    for _ in range(500):
        sample = rng.choice(
            reference,
            size=len(normal_batch),
            replace=True,
        )
        null_distances.append(
            distribution_distance(reference, sample, bins)
        )

    threshold = float(np.quantile(null_distances, 0.95))

    print(f"Variable surveillée : {COLUMN}")
    print(f"Seuil empirique de démonstration : {threshold:.4f}")

    for name, batch in [
        ("Lot sans modification", normal_batch),
        ("Lot simulé avec usage réduit", drifted_batch),
    ]:
        distance = distribution_distance(
            reference, batch, bins
        )

        print(f"\n{name}")
        print(f"Moyenne : {batch.mean():.2f} secondes")
        print(f"Distance : {distance:.4f}")
        print(
            "Signal :",
            "Dérive à examiner"
            if distance > threshold
            else "Pas de dérive signalée",
        )

    print(
        "\nSimulation sur une seule variable : "
        "un signal de dérive ne prouve pas "
        "une baisse des performances du modèle."
    )


if __name__ == "__main__":
    main()