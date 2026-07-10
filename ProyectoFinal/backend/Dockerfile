FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY backend/pyproject.toml backend/uv.lock ./

RUN uv sync --frozen --no-dev

COPY backend/ ./backend/
COPY modelo_final.pkl ./modelo_final.pkl

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
