FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY frontend/pyproject.toml frontend/uv.lock ./

RUN uv sync --frozen --no-dev

COPY frontend/ ./frontend/

EXPOSE 7860

CMD ["python", "frontend/app.py"]
