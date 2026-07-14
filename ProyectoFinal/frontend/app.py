"""Interfaz Gradio para la clasificación de tickets."""

from __future__ import annotations

import os

import gradio as gr
from services import enviar_prediccion

PRIORITY_DESCRIPTIONS = {
    "Baja": ("El ticket puede atenderse dentro del flujo normal de soporte."),
    "Media": ("El ticket requiere atención, pero no representa una emergencia inmediata."),
    "Alta": ("El ticket debe priorizarse debido a su posible impacto."),
    "Critica": ("El ticket requiere atención inmediata y escalamiento."),
}

CSS = """
.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}

.hero {
    text-align: center;
    padding: 1rem 0 1.5rem 0;
}

.result-card {
    border-radius: 14px;
    padding: 0.5rem;
}
"""


def solicitar_prediccion(
    asunto: str,
    contenido: str,
) -> tuple[str, str]:
    """Solicita una predicción al backend."""
    try:
        prediction = enviar_prediccion(
            asunto=asunto,
            contenido=contenido,
        )
    except (ValueError, RuntimeError) as error:
        raise gr.Error(str(error)) from error

    description = PRIORITY_DESCRIPTIONS[prediction]

    return prediction, f"### {prediction}\n\n{description}"


def limpiar_formulario() -> tuple[str, str, str, str]:
    """Limpia los componentes del formulario."""
    return "", "", "", ""


theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="orange",
)

with gr.Blocks(
    title="ChaucherApp - Clasificador de Tickets",
) as demo:
    gr.Markdown(
        """
        # ChaucherApp — Clasificador de prioridad
        Ingresa la información del ticket para estimar su nivel de prioridad.
        """,
        elem_classes=["hero"],
    )

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Group():
                gr.Markdown("## Atributos del ticket")

                asunto_input = gr.Textbox(
                    label="Asunto del ticket",
                    placeholder=("Ejemplo: Transferencia desconocida en mi cuenta"),
                    max_lines=3,
                )

                contenido_input = gr.Textbox(
                    label="Contenido del ticket",
                    placeholder=("Describe detalladamente el problema reportado..."),
                    lines=8,
                )

            with gr.Group():
                gr.Markdown("## Atributos del usuario")
                gr.Markdown(
                    """
                    El modelo final seleccionado utiliza únicamente la
                    información textual del ticket. Por esta razón, no
                    requiere atributos adicionales del usuario.
                    """
                )

            with gr.Row():
                predict_button = gr.Button(
                    "Predecir prioridad",
                    variant="primary",
                )
                clear_button = gr.Button(
                    "Limpiar formulario",
                    variant="secondary",
                )

        with gr.Column(scale=1, elem_classes=["result-card"]):
            gr.Markdown("## Resultado")

            prediction_output = gr.Textbox(
                label="Prioridad predicha",
                interactive=False,
            )

            explanation_output = gr.Markdown("Completa el formulario y presiona **Predecir prioridad**.")

    predict_button.click(
        fn=solicitar_prediccion,
        inputs=[
            asunto_input,
            contenido_input,
        ],
        outputs=[
            prediction_output,
            explanation_output,
        ],
    )

    clear_button.click(
        fn=limpiar_formulario,
        inputs=[],
        outputs=[
            asunto_input,
            contenido_input,
            prediction_output,
            explanation_output,
        ],
    )

    gr.Examples(
        examples=[
            [
                "Transferencia desconocida",
                ("Encontré una transferencia que no realicé y necesito bloquear la operación de manera urgente."),
            ],
            [
                "Problema para iniciar sesión",
                ("Cambié de teléfono y ahora no puedo acceder a mi cuenta."),
            ],
            [
                "Consulta sobre movimientos",
                ("Quisiera conocer dónde puedo revisar el historial completo de mis movimientos."),
            ],
        ],
        inputs=[
            asunto_input,
            contenido_input,
        ],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("FRONTEND_PORT", "7860")),
        share=False,
        theme=theme,
        css=CSS,
    )
