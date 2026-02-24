# Déploiement Docker de l'API FastAPI

## Description
Ce dossier contient tout le nécessaire pour déployer l'API FastAPI avec le modèle GradientBoosting (R²=0.85) dans un conteneur Docker.

## Fichiers inclus
- `Dockerfile` : Configuration pour construire l'image
- `main.py` : API FastAPI avec 3 endpoints
- `requirements.txt` : Dépendances Python
- `gradient_boosting_best.pkl` : Modèle entraîné
- `scaler_gb.pkl` : Scaler pour normalisation

## Commandes Docker

### 1. Construire l'image
```bash
docker build -t fastapi-energie:v1 .