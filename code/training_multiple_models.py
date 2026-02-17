"""
TRAINING MULTI-MODÈLES AVEC VRAIES DONNÉES
"""
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime

print("=" * 70)
print("TRAINING MULTI-MODÈLES AVEC VRAIES DONNÉES")
print("=" * 70)

# ==================== 1. CHARGEMENT DES VRAIES DONNÉES ====================
print("\n1. 📊 CHARGEMENT DES DONNÉES RÉELLES")
print("-" * 40)

try:
    # Charger Adelaide
    df = pd.read_csv('data/Adelaide_Data.csv', nrows=10000)
    print(f"✅ Dataset chargé: {df.shape}")
    
    # Définir target (ADAPTER SELON VOS COLONNES!)
    # Si vous savez quelle colonne est la production d'énergie:
    # target_col = 'production_energy'  # À remplacer par le vrai nom
    
    # Sinon, prenez la dernière colonne comme exemple
    target_col = df.columns[-1]
    feature_cols = df.columns[:-1]
    
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"🎯 Target: {target_col}")
    print(f"🔢 Features: {X.shape[1]} variables")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"📊 Train: {X_train.shape[0]} samples")
    print(f"📊 Test: {X_test.shape[0]} samples")
    
    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Sauvegarder le scaler
    joblib.dump(scaler, 'data/processed/scaler_reel.pkl')
    print("✅ Normalisation terminée")
    
except Exception as e:
    print(f"❌ Erreur: {e}")
    exit()

# ==================== 2. CONFIGURATION MLFLOW ====================
print("\n2. 🔧 CONFIGURATION MLFLOW")
print("-" * 40)

os.makedirs('mlruns', exist_ok=True)
experiment_name = f"Modeles_Reels_Energie_{datetime.now().strftime('%Y%m%d_%H%M')}"
mlflow.set_experiment(experiment_name)
print(f"📁 Expérience: {experiment_name}")

# ==================== 3. DÉFINITION DES MODÈLES ====================
print("\n3. 🤖 LISTE DES MODÈLES")
print("-" * 40)

models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'DecisionTree': DecisionTreeRegressor(max_depth=10, random_state=42),
    'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1),
    'SVR': SVR(kernel='rbf', C=100, gamma=0.1)
}

print(f"📋 {len(models)} modèles à entraîner")

# ==================== 4. ENTRAÎNEMENT ====================
print("\n4. 🚀 ENTRAÎNEMENT")
print("-" * 40)

results = []

for model_name, model in models.items():
    print(f"\n🔵 {model_name}...")
    
    with mlflow.start_run(run_name=model_name):
        
        # Paramètres
        mlflow.log_params(model.get_params())
        
        # Entraînement
        model.fit(X_train_scaled, y_train)
        
        # Prédictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Métriques
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        # Validation croisée (optionnel, peut être lent)
        try:
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='r2')
            cv_mean = cv_scores.mean()
        except:
            cv_mean = None
        
        # Loguer
        metrics = {
            "train_r2": train_r2,
            "test_r2": test_r2,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse
        }
        mlflow.log_metrics(metrics)
        
        # Sauvegarder
        mlflow.sklearn.log_model(model, f"{model_name}_model")
        
        # Résultats
        results.append({
            'Modèle': model_name,
            'Train R²': round(train_r2, 4),
            'Test R²': round(test_r2, 4),
            'Train RMSE': round(train_rmse, 2),
            'Test RMSE': round(test_rmse, 2)
        })
        
        print(f"   ✅ Test R²: {test_r2:.4f} | Test RMSE: {test_rmse:.2f}")

# ==================== 5. RÉSULTATS ====================
print("\n" + "=" * 70)
print("📋 RÉSULTATS AVEC VRAIES DONNÉES")
print("=" * 70)

df_results = pd.DataFrame(results)
df_results = df_results.sort_values('Test R²', ascending=False)

print("\n🏆 CLASSEMENT:")
print(df_results.to_string(index=False))

# Sauvegarder
df_results.to_csv('data/reports/comparaison_modeles_reels.csv', index=False)
print(f"\n💾 Résultats: data/reports/comparaison_modeles_reels.csv")

# Meilleur modèle
best = df_results.iloc[0]
print(f"\n🥇 MEILLEUR MODÈLE: {best['Modèle']}")
print(f"   Test R²: {best['Test R²']}")
print(f"   Test RMSE: {best['Test RMSE']}")

# Sauvegarder meilleur modèle
best_model = models[best['Modèle']]
best_model.fit(X_train_scaled, y_train)
joblib.dump(best_model, 'data/processed/best_model_reel.pkl')
print(f"✅ Modèle sauvegardé: data/processed/best_model_reel.pkl")