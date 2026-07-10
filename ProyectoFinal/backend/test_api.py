"""Pruebas manuales para el endpoint de predicción."""

import json

import requests

API_URL = "http://127.0.0.1:8000/predict"
TIMEOUT_SECONDS = 60


def print_result(
    case_name: str,
    payload: dict[str, object],
    response: requests.Response,
) -> None:
    """Imprime el input y output de una llamada."""
    print("=" * 70)
    print(case_name)
    print("Input:")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("Status code:", response.status_code)
    print("Output:")
    print(
        json.dumps(
            response.json(),
            ensure_ascii=False,
            indent=2,
        )
    )


successful_payload = {
    "asunto": "Transferencia desconocida en mi cuenta",
    "contenido": ("Detecté una transferencia que no realicé. Necesito bloquearla y recuperar el dinero."),
}

successful_response = requests.post(
    API_URL,
    json=successful_payload,
    timeout=TIMEOUT_SECONDS,
)

print_result(
    case_name="LLAMADA EXITOSA",
    payload=successful_payload,
    response=successful_response,
)


invalid_payload = {
    "asunto": 123,
    "contenido": "Este asunto tiene un tipo de dato incorrecto.",
}

invalid_response = requests.post(
    API_URL,
    json=invalid_payload,
    timeout=TIMEOUT_SECONDS,
)

print_result(
    case_name="LLAMADA NO EXITOSA",
    payload=invalid_payload,
    response=invalid_response,
)
