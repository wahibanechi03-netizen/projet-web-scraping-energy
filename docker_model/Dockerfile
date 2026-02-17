# Dockerfile modifié
FROM python:3.9-slim

WORKDIR /app

# Copier les fichiers depuis leur emplacement réel
COPY preprocessing_final.py .
COPY training_multiple_models_corrige.py .
COPY best_model_reel.pkl .
COPY scaler_reel.pkl .
COPY requirements.txt .
COPY predict_api.py . 

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install mlflow pandas numpy scikit-learn xgboost joblib

EXPOSE 5000
CMD ["python", "predict_api.py"]
