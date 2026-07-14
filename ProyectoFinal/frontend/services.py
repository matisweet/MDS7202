"""Funciones para comunicarse con la API de predicción."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

import requests
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_DIR / ".env")

BACKEND_URL = os.getenv("BACKEND_URL")

if not BACKEND_URL:
    raise RuntimeError("No se encontró BACKEND_URL en las variables de entorno.")

PREDICT_URL: Final[str] = f"{BACKEND_URL.rstrip('/')}/predict"
TIMEOUT_SECONDS: Final[int] = 60
VALID_PRIORITIES: Final[set[str]] = {
    "Baja",
    "Media",
    "Alta",
    "Critica",
}


def enviar_prediccion(asunto: str, contenido: str) -> str:
    """Envía un ticket al backend y retorna su prioridad.

    Parameters
    ----------
    asunto:
        Asunto del ticket.
    contenido:
        Descripción completa del ticket.

    Returns
    -------
    str
        Prioridad predicha por el modelo.
    """
    payload = {
        "asunto": asunto,
        "contenido": contenido,
    }

    try:
        response = requests.post(
            PREDICT_URL,
            json=payload,
            timeout=TIMEOUT_SECONDS,
        )
    except requests.RequestException as error:
        raise RuntimeError(
            "No fue posible conectarse con el backend. Compruebe que FastAPI se encuentre ejecutándose."
        ) from error

    if response.status_code == 422:
        try:
            detail = response.json().get(
                "detail",
                "Datos de entrada inválidos.",
            )
        except requests.JSONDecodeError:
            detail = response.text

        raise ValueError(f"La API rechazó los datos enviados: {detail}")

    try:
        response.raise_for_status()
    except requests.HTTPError as error:
        raise RuntimeError(f"El backend no pudo generar la predicción. Código HTTP: {response.status_code}.") from error

    try:
        response_data = response.json()
    except requests.JSONDecodeError as error:
        raise RuntimeError("El backend retornó una respuesta que no es JSON.") from error

    prediction = response_data.get("prediction")

    if prediction not in VALID_PRIORITIES:
        raise RuntimeError(f"El backend retornó una prioridad no reconocida: {prediction!r}.")

    return prediction
