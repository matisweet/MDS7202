FROM python:3.14-slim

WORKDIR /app

# instalamos primero las dependencias para aprovechar el cache de capas de docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copiamos la api y el modelo ya entrenado
COPY main.py .
COPY models/ models/

EXPOSE 8000

# ojo con el host: con el 127.0.0.1 por defecto la api no se ve desde fuera del contenedor
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
