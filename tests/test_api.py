from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import app, FEATURES, MODEL_VERSION

VALID_CUSTOMER = {
    "call_failure": 5,
    "complains": 1,
    "subscription_length": 34,
    "charge_amount": 1,
    "seconds_of_use": 1200,
    "frequency_of_use": 25,
    "frequency_of_sms": 10,
    "distinct_called_numbers": 10,
    "age_group": 3,
    "tariff_plan": 1,
}


@pytest.fixture(scope="module")
def client():
    # Le contexte déclenche le chargement du modèle.
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["model_version"] == MODEL_VERSION
    assert body["threshold"] == pytest.approx(0.3)


def test_prediction_matches_loaded_model(client):
    response = client.post("/predict", json=VALID_CUSTOMER)

    assert response.status_code == 200
    body = response.json()

    inputs = pd.DataFrame(
        [VALID_CUSTOMER], columns=FEATURES
    ).astype("int64")

    expected_score = float(
        app.state.model.predict_proba(inputs)[0, 1]
    )

    assert np.isfinite(body["churn_score"])
    assert 0 <= body["churn_score"] <= 1
    assert body["churn_score"] == pytest.approx(expected_score)
    assert body["prediction"] == int(
        expected_score >= app.state.threshold
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("seconds_of_use", -100),
        ("tariff_plan", 3),
        ("age_group", 6),
        ("complains", 2),
        ("call_failure", "cinq"),
    ],
)
def test_invalid_values_are_rejected(client, field, value):
    payload = deepcopy(VALID_CUSTOMER)
    payload[field] = value

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert any(
        error["loc"] == ["body", field]
        for error in response.json()["detail"]
    )


def test_missing_field_is_rejected(client):
    payload = deepcopy(VALID_CUSTOMER)
    del payload["seconds_of_use"]

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_extra_field_is_rejected(client):
    payload = {**VALID_CUSTOMER, "churn": 1}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422