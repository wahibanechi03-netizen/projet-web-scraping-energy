"""
API de prédiction avec Flask pour Docker
"""
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import mlflow
import os

app = Flask(__name__)

# Charger le modèle et le scaler au démarrage
print("🔄 Chargement du modèle...")
model = joblib.load('best_model_reel.pkl')
scaler = joblib.load('scaler_reel.pkl')
print("✅ Modèle chargé!")

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "API de prédiction - Modèle GradientBoosting",
        "status": "active",
        "model_type": str(type(model).__name__)
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données JSON
        data = request.get_json()
        
        # Convertir en DataFrame
        df = pd.DataFrame([data])
        
        # Normaliser
        df_scaled = scaler.transform(df)
        
        # Prédire
        prediction = model.predict(df_scaled)
        
        return jsonify({
            "prediction": float(prediction[0]),
            "status": "success"
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)