# Dockerfile pour FastAPI avec modèle GradientBoosting
FROM python:3.10-slim

WORKDIR /app

# Copier les fichiers nécessaires
COPY requirements.txt .
COPY main.py .
COPY gradient_boosting_best.pkl .
COPY scaler_gb.pkl .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port
EXPOSE 8000

# Commande pour lancer l'API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]