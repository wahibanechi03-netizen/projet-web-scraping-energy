"""
API FastAPI avec 3 endpoints:
- /health : Vérifier l'état
- /predict : Faire une prédiction
- /model/info : Informations sur le modèle
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from typing import List, Optional
import uvicorn
import os


# ==================== INITIALISATION FASTAPI ====================
app = FastAPI(
    title="API Prédiction - Énergie des Vagues",
    description="API pour le modèle GradientBoosting (R²=0.85)",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc"  # Alternative documentation
)

# Configuration CORS pour permettre au frontend React de communiquer
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("=" * 60)
print("DÉMARRAGE DE L'API FASTAPI")
print("=" * 60)

# ==================== CHARGEMENT DU MODÈLE ====================
print("\n" + "=" * 60)
print("CHARGEMENT DU MODÈLE")
print("=" * 60)

try:
    global model, scaler, FEATURES_COUNT
    import os
    import joblib
    import numpy as np
    
    print(f"📁 Dossier courant: {os.getcwd()}")
    
    # Lister tous les fichiers .pkl
    print("\n📁 Fichiers .pkl dans le dossier:")
    pkl_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
    for f in pkl_files:
        size = os.path.getsize(f) / 1024
        print(f"   - {f} ({size:.1f} KB)")
    
    # Charger le modèle
    model_path = 'gradient_boosting_best.pkl'
    scaler_path = 'scaler_gb.pkl'
    
    print(f"\n🔍 Chargement du modèle: {model_path}")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"✅ Modèle chargé: {type(model).__name__}")
        print(f"✅ Modèle parameters: {model.get_params() if hasattr(model, 'get_params') else 'N/A'}")
    else:
        print(f"❌ Fichier modèle introuvable: {model_path}")
        model = None
    
    print(f"\n🔍 Chargement du scaler: {scaler_path}")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"✅ Scaler chargé: {type(scaler).__name__}")
        FEATURES_COUNT = scaler.mean_.shape[0] if hasattr(scaler, 'mean_') else 48
        print(f"✅ Features attendues: {FEATURES_COUNT}")
    else:
        print(f"❌ Fichier scaler introuvable: {scaler_path}")
        scaler = None
        FEATURES_COUNT = 48
    
    if model is None or scaler is None:
        print("\n⚠️  MODE DÉGRADÉ - Utilisation de valeurs par défaut")
        
except Exception as e:
    print(f"\n❌ ERREUR CRITIQUE: {e}")
    model = None
    scaler = None
    FEATURES_COUNT = 48

print("\n" + "=" * 60)

# ==================== VÉRIFICATION GLOBALE ====================
print("\n" + "=" * 60)
print("VÉRIFICATION FINALE")
print("=" * 60)
print(f"Variable 'model' existe: {'model' in locals()}")
print(f"Variable 'scaler' existe: {'scaler' in locals()}")
print(f"model is None: {model is None}")
print(f"scaler is None: {scaler is None}")
if 'model' in locals() and model is not None:
    print(f"Type model: {type(model).__name__}")
if 'scaler' in locals() and scaler is not None:
    print(f"Type scaler: {type(scaler).__name__}")
print("=" * 60)
# ==================== MODÈLES DE DONNÉES ====================
class PredictionInput(BaseModel):
    """Entrée pour une prédiction"""
    features: List[float] = Field(
        ..., 
        description=f"Liste de {FEATURES_COUNT} valeurs numériques",
        example=[10.5, 20.3, 30.1] + [0.0] * 45
    )
    
    class Config:
        schema_extra = {
            "example": {
                "features": [10.5, 20.3, 30.1, 40.2, 50.4] + [0.0] * 43
            }
        }

class PredictionOutput(BaseModel):
    """Sortie d'une prédiction"""
    prediction: float
    model_used: str
    r2_score: float
    features_used: int
    status: str

class HealthOutput(BaseModel):
    """Sortie du endpoint health"""
    status: str
    model_loaded: bool
    model_type: str
    features_count: int
    r2_score: float
    api_version: str

class ModelInfoOutput(BaseModel):
    """Informations sur le modèle"""
    model_type: str
    r2_score: float
    features_count: int
    performance: dict
    parameters: dict

# ==================== ENDPOINT 1: HEALTH ====================
@app.get("/health", 
         response_model=HealthOutput,
         tags=["Monitoring"],
         summary="Vérifier la santé de l'API",
         description="Retourne l'état de l'API et du modèle")
async def health_check():
    """
    ## Endpoint de santé
    
    Vérifie que l'API et le modèle fonctionnent correctement.
    
    ### Retourne:
    - **status**: "healthy" ou "unhealthy"
    - **model_loaded**: True si modèle chargé
    - **model_type**: Type du modèle
    - **features_count**: Nombre de features attendues
    - **r2_score**: Score R² du modèle
    - **api_version**: Version de l'API
    """
    if model is None:
        return HealthOutput(
            status="degraded",
            model_loaded=False,
            model_type="None",
            features_count=FEATURES_COUNT,
            r2_score=0.85,
            api_version="1.0.0"
        )
    
    return HealthOutput(
        status="healthy",
        model_loaded=True,
        model_type=type(model).__name__,
        features_count=FEATURES_COUNT,
        r2_score=0.85,
        api_version="1.0.0"
    )

# ==================== ENDPOINT 2: PREDICT ====================
@app.post("/predict",
          response_model=PredictionOutput,
          tags=["Prédiction"],
          summary="Faire une prédiction",
          description="Envoie des features et reçoit une prédiction")
async def predict(input_data: PredictionInput):
    """
    ## Endpoint de prédiction
    
    Fait une prédiction à partir des features fournies.
    
    ### Paramètres:
    - **features**: Liste de {FEATURES_COUNT} valeurs numériques
    
    ### Retourne:
    - **prediction**: Valeur prédite
    - **model_used**: Modèle utilisé
    - **r2_score**: Score du modèle
    - **features_used**: Nombre de features utilisées
    - **status**: Succès ou erreur
    
    ### Exemple:
    ```json
    {
        "features": [10.5, 20.3, 30.1, 40.2, 50.4, 0, 0, ...]
    }
    ```
    """
    # Vérifier que le modèle est chargé
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non disponible"
        )
    
    # Vérifier le nombre de features
    if len(input_data.features) != FEATURES_COUNT:
        raise HTTPException(
            status_code=400,
            detail=f"Nombre de features incorrect. Attendu: {FEATURES_COUNT}, Reçu: {len(input_data.features)}"
        )
    
    try:
        # Convertir en tableau numpy
        features_array = np.array(input_data.features).reshape(1, -1)
        
        # Normaliser
        features_scaled = scaler.transform(features_array)
        
        # Prédire
        prediction = model.predict(features_scaled)[0]
        
        return PredictionOutput(
            prediction=float(prediction),
            model_used=type(model).__name__,
            r2_score=0.85,
            features_used=FEATURES_COUNT,
            status="success"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur de prédiction: {str(e)}"
        )

# ==================== ENDPOINT 3: MODEL INFO ====================
@app.get("/model/info",
         response_model=ModelInfoOutput,
         tags=["Information"],
         summary="Informations sur le modèle",
         description="Retourne les détails du modèle entraîné")
async def model_info():
    """
    ## Informations du modèle
    
    Retourne les caractéristiques du modèle GradientBoosting.
    
    ### Informations:
    - Type de modèle
    - Score R²
    - Nombre de features
    - Performances (train/test)
    - Paramètres du modèle
    """
    if model is None:
        # Informations par défaut si modèle non chargé
        return ModelInfoOutput(
            model_type="GradientBoosting (non chargé)",
            r2_score=0.85,
            features_count=FEATURES_COUNT,
            performance={
                "train_r2": 0.9552,
                "test_r2": 0.8496,
                "train_rmse": 6949.76,
                "test_rmse": 12758.20
            },
            parameters={
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1
            }
        )
    
    # Extraire les paramètres du modèle
    params = {}
    if hasattr(model, 'get_params'):
        params = model.get_params()
    
    return ModelInfoOutput(
        model_type=type(model).__name__,
        r2_score=0.85,
        features_count=FEATURES_COUNT,
        performance={
            "train_r2": 0.9552,
            "test_r2": 0.8496,
            "train_rmse": 6949.76,
            "test_rmse": 12758.20
        },
        parameters=params
    )

# ==================== ENDPOINT RACINE ====================
@app.get("/",
         tags=["Information"],
         summary="Page d'accueil",
         include_in_schema=False)
async def root():
    """Page d'accueil de l'API"""
    return {
        "message": "API Prédiction - Énergie des Vagues",
        "version": "1.0.0",
        "model": "GradientBoosting (R²=0.85)",
        "endpoints": {
            "/health": "GET - Vérifier l'état",
            "/predict": "POST - Faire une prédiction",
            "/model/info": "GET - Infos du modèle",
            "/docs": "GET - Documentation Swagger",
            "/redoc": "GET - Documentation alternative"
        },
        "documentation": "http://localhost:8000/docs"
    }

# ==================== LANCEMENT ====================
if __name__ == "__main__":
    print("\n🚀 Démarrage du serveur FastAPI...")
    print("📝 Documentation: http://localhost:8000/docs")
    print("🔍 Swagger UI: http://localhost:8000/docs")
    print("📊 Health check: http://localhost:8000/health")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)