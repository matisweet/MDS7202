"""Modelos de entrada y salida de la API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, StrictStr, field_validator

Priority = Literal["Baja", "Media", "Alta", "Critica"]


class PredictionRequest(BaseModel):
    """Datos mínimos necesarios para clasificar un ticket."""

    model_config = ConfigDict(extra="forbid")

    asunto: StrictStr
    contenido: StrictStr

    @field_validator("asunto", "contenido")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        """Rechaza strings vacíos o compuestos únicamente por espacios."""
        cleaned_value = value.strip()

        if not cleaned_value:
            raise ValueError("El campo no puede estar vacío.")

        return cleaned_value


class PredictionResponse(BaseModel):
    """Respuesta entregada por el modelo."""

    model_config = ConfigDict(extra="forbid")

    prediction: Priority
