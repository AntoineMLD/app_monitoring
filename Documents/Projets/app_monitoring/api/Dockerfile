FROM python:3.9-slim

# Défini le répertoire de travail dans le conteneur
WORKDIR /app

# Copie le fichier requirements.txt depuis la racine du projet
COPY requirements.txt /app/

RUN mkdir -p /app/model
RUN mkdir -p /app/monitoring/reports

# Installe les dépendances
RUN pip install --upgrade pip && \
    pip install --no-cache-dir pandas && \
    pip install --no-cache-dir scikit-learn && \
    pip install --no-cache-dir fastapi uvicorn && \
    pip install --no-cache-dir prometheus_client && \
    pip install --no-cache-dir prometheus_fastapi_instrumentator && \
    pip install --no-cache-dir evidently && \
    pip install --no-cache-dir joblib

COPY model/ /app/model/
COPY monitoring/ /app/monitoring/
COPY ./api /app

# Expose le port de l'API
EXPOSE 8000

# Lance l'API avec Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
