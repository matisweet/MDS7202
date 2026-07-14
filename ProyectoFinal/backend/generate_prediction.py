"""Generación de predicciones para la prioridad de tickets."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import cloudpickle
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "modelo_final.pkl"
ENV_PATH = PROJECT_DIR / ".env"

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSION = 1024

# El .env está en ProyectoFinal/.env
load_dotenv(ENV_PATH)


def _normalize_newlines(text: str) -> str:
    """Normaliza saltos de línea para mantener un formato consistente."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _build_embedding_text(asunto: str, contenido: str) -> str:
    """Construye exactamente el texto solicitado por el enunciado."""
    asunto = _normalize_newlines(asunto)
    contenido = _normalize_newlines(contenido)

    return f"Asunto_Ticket: {asunto}\nContenido_Ticket: {contenido}\n"


@lru_cache(maxsize=1)
def _load_model() -> Any:
    """Carga una sola vez el pipeline entrenado."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo en: {MODEL_PATH}")

    with MODEL_PATH.open("rb") as model_file:
        return cloudpickle.load(model_file)


@lru_cache(maxsize=1)
def _load_embedding_client() -> GoogleGenerativeAIEmbeddings:
    """Inicializa una sola vez el cliente de embeddings."""
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise RuntimeError(f"No se encontró GOOGLE_API_KEY. Revise el archivo {ENV_PATH}.")

    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
        output_dimensionality=EMBEDDING_DIMENSION,
    )


def _get_embedding_columns(model: Any) -> list[str]:
    """Recupera del pipeline las columnas usadas por el modelo."""
    preprocessing = model.named_steps.get("Preprocessing")

    if preprocessing is None:
        raise RuntimeError("El pipeline no contiene el paso 'Preprocessing'.")

    for name, _transformer, columns in preprocessing.transformers_:
        if name == "Emb":
            embedding_columns = list(columns)

            if len(embedding_columns) != EMBEDDING_DIMENSION:
                raise RuntimeError(
                    "El pipeline no utiliza 1024 dimensiones de embedding. "
                    f"Dimensiones encontradas: {len(embedding_columns)}."
                )

            return embedding_columns

    raise RuntimeError("No se encontró el transformador 'Emb' en el pipeline.")


def _build_model_input(
    model: Any,
    embedding: list[float],
    asunto: str,
    contenido: str,
) -> pd.DataFrame:
    """Construye un DataFrame compatible con el pipeline entrenado."""
    embedding_columns = _get_embedding_columns(model)

    if len(embedding) != len(embedding_columns):
        raise ValueError(
            "La dimensión del embedding no coincide con el modelo: "
            f"{len(embedding)} recibidas y "
            f"{len(embedding_columns)} esperadas."
        )

    # El pipeline fue entrenado recibiendo X_full. Aunque el MLP utiliza
    # exclusivamente las columnas Emb, mantenemos el esquema de entrada
    # original para asegurar compatibilidad con feature_names_in_.
    expected_columns = list(getattr(model, "feature_names_in_", embedding_columns))

    row: dict[str, object] = {column: np.nan for column in expected_columns}

    row.update(dict(zip(embedding_columns, embedding, strict=True)))

    # Estas columnas fueron creadas durante la preparación de X_full.
    # El MLP final no las utiliza, pero se reconstruyen cuando están
    # presentes en el esquema esperado.
    asunto_normalizado = _normalize_newlines(asunto)
    contenido_normalizado = _normalize_newlines(contenido)

    if "N_Caracteres_Ticket" in row:
        row["N_Caracteres_Ticket"] = len(asunto_normalizado + contenido_normalizado)

    if "Texto" in row:
        row["Texto"] = asunto_normalizado + " " + contenido_normalizado

    return pd.DataFrame([row], columns=expected_columns)


def generate_prediction(asunto: str, contenido: str) -> str:
    """Predice la prioridad de un ticket.

    Parameters
    ----------
    asunto:
        Asunto o título del ticket.
    contenido:
        Descripción completa del ticket.

    Returns
    -------
    str
        Categoría predicha: Baja, Media, Alta o Critica.
    """
    if not isinstance(asunto, str):
        raise TypeError("El asunto debe ser un string.")

    if not isinstance(contenido, str):
        raise TypeError("El contenido debe ser un string.")

    if not asunto.strip():
        raise ValueError("El asunto no puede estar vacío.")

    if not contenido.strip():
        raise ValueError("El contenido no puede estar vacío.")

    embedding_text = _build_embedding_text(asunto, contenido)

    embedding_client = _load_embedding_client()
    embedding = embedding_client.embed_query(embedding_text)

    model = _load_model()
    model_input = _build_model_input(
        model=model,
        embedding=embedding,
        asunto=asunto,
        contenido=contenido,
    )

    prediction = model.predict(model_input)[0]

    return str(prediction)


if __name__ == "__main__":
    sample_asunto = "No puedo acceder a mi cuenta"
    sample_contenido = (
        "Cambié de celular y ahora la aplicación no me permite "
        "iniciar sesión. Necesito recuperar el acceso porque tengo "
        "dinero guardado en la cuenta."
    )

    sample_prediction = generate_prediction(
        asunto=sample_asunto,
        contenido=sample_contenido,
    )

    print("Asunto:", sample_asunto)
    print("Contenido:", sample_contenido)
    print("Prioridad predicha:", sample_prediction)
