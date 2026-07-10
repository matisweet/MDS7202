"""API para clasificar la prioridad de tickets."""

import logging

from fastapi import FastAPI, HTTPException, status

from .generate_prediction import generate_prediction
from .models import PredictionRequest, PredictionResponse

logger = logging.getLogger(__name__)

app = FastAPI(
    title="ChaucherApp Ticket Priority API",
    description="API para clasificar la prioridad de tickets de soporte.",
    version="1.0.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Comprueba que la API se encuentra disponible."""
    return {"status": "ok"}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
)
def predict_ticket(payload: PredictionRequest) -> PredictionResponse:
    """Genera la predicción de prioridad para un ticket."""
    try:
        prediction = generate_prediction(
            asunto=payload.asunto,
            contenido=payload.contenido,
        )

        return PredictionResponse(prediction=prediction)

    except Exception as error:
        logger.exception("Error al generar la predicción.")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No fue posible generar la predicción.",
        ) from error
