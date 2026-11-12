from contextlib import asynccontextmanager
from typing import Literal
import math
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from mlflow import MlflowClient
from pydantic import BaseModel, ConfigDict, Field

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import Request

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

monitor_logger = logging.getLogger("telecom_churn.monitor")
monitor_logger.setLevel(logging.INFO)
monitor_logger.propagate = False

if not monitor_logger.handlers:
    handler = RotatingFileHandler(
        LOG_DIR / "requests.jsonl",
        maxBytes=5_000_000,
        backupCount=2,
        encoding="utf-8",
    )

    handler.setFormatter(logging.Formatter("%(message)s"))
    monitor_logger.addHandler(handler)


# Configuration : version précise pour un déploiement reproductible
TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"
)
MODEL_NAME = "telecom-churn-random-forest"
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")

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


# Validation des données reçues
class CustomerInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    call_failure: int = Field(ge=0)
    complains: Literal[0, 1]
    subscription_length: int = Field(ge=0)
    charge_amount: int = Field(ge=0, le=10)
    seconds_of_use: int = Field(ge=0)
    frequency_of_use: int = Field(ge=0)
    frequency_of_sms: int = Field(ge=0)
    distinct_called_numbers: int = Field(ge=0)
    age_group: Literal[1, 2, 3, 4, 5]
    tariff_plan: Literal[1, 2]


class PredictionOutput(BaseModel):
    churn_score: float
    prediction: Literal[0, 1]
    threshold: float
    model_name: str
    model_version: str


# Charger une seule fois le modèle et son seuil au démarrage
@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    version_info = client.get_model_version(
        name=MODEL_NAME,
        version=MODEL_VERSION,
    )

    threshold = float(
        version_info.tags["decision_threshold"]
    )

    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Seuil de décision invalide.")

    model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    model = mlflow.sklearn.load_model(model_uri)

    if list(model.feature_names_in_) != FEATURES:
        raise ValueError(
            "Les variables du modèle ne correspondent pas à l'API."
        )

    if list(model.classes_) != [0, 1]:
        raise ValueError("Classes du modèle inattendues.")

    app.state.model = model
    app.state.threshold = threshold

    print(f"Modèle chargé : {model_uri}")
    print(f"Seuil : {threshold}")

    yield

    app.state.model = None


app = FastAPI(
    title="Telecom Churn Prediction API",
    description="Estimation du risque de désabonnement client.",
    version="1.0.0",
    lifespan=lifespan,
)

@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    request_id = str(uuid4())
    started_at = perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response

    finally:
        duration_ms = (perf_counter() - started_at) * 1000

        event = {
            "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": status_code,
            "duration_ms": round(duration_ms, 3),
            "validation_error": status_code == 422,
            "server_error": status_code >= 500,
            "model_version": MODEL_VERSION,
        }

        monitor_logger.info(json.dumps(event))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "threshold": app.state.threshold,
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(customer: CustomerInput):
    # Respecter les colonnes et types utilisés à l'entraînement
    inputs = pd.DataFrame(
        [customer.model_dump()],
        columns=FEATURES,
    ).astype("int64")

    score = float(
        app.state.model.predict_proba(inputs)[0, 1]
    )

    decision = int(score >= app.state.threshold)

    return PredictionOutput(
        churn_score=score,
        prediction=decision,
        threshold=app.state.threshold,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
    )