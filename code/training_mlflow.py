"""
ENTRAÎNEMENT MLflow - 1 MODÈLE
Phase 4 CRISP-DM: Modeling
"""
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
import joblib
import os
from datetime import datetime

print("=" * 70)
print("PHASE 4 CRISP-DM: MODELING AVEC MLFLOW")
print("=" * 70)

# ==================== 1. CHARGER DONNÉES PRÉPARÉES ====================
print("\n1. 📊 CHARGEMENT DES DONNÉES PREPROCESSÉES")
print("-" * 40)

# Charger les données préparées par preprocessing
X_train = pd.read_csv('data/processed/X_train_scaled.csv')
X_test = pd.read_csv('data/processed/X_test_scaled.csv')

# Charger les targets (supposons qu'elles existent)
# Si vous ne les avez pas, créez-les
try:
    y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
    y_test = pd.read_csv('data/processed/y_test.csv').squeeze()
except:
    # Simulation pour démo (à remplacer par vos vraies données)
    print("⚠️  Targets non trouvées, simulation pour démo")
    y_train = np.random.randn(len(X_train))
    y_test = np.random.randn(len(X_test))

print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test: {y_test.shape}")

# ==================== 2. CONFIGURATION MLFLOW ====================
print("\n2. 🔧 CONFIGURATION MLFLOW")
print("-" * 40)

# Créer dossier pour les logs
os.makedirs('mlruns', exist_ok=True)

# Définir l'expérience
experiment_name = f"Modele_Energie_Vagues_{datetime.now().strftime('%Y%m%d')}"
mlflow.set_experiment(experiment_name)

print(f"Expérience MLflow: {experiment_name}")
print(f"📁 Logs sauvegardés dans: mlruns/")

# ==================== 3. ENTRAÎNEMENT AVEC MLFLOW ====================
print("\n3. 🤖 ENTRAÎNEMENT DU MODÈLE (Random Forest)")
print("-" * 40)

# Définir les paramètres du modèle
params = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
    "model_type": "RandomForestRegressor",
    "features_count": X_train.shape[1],
    "train_samples": X_train.shape[0],
    "test_samples": X_test.shape[0]
}

print("Paramètres du modèle:")
for key, value in params.items():
    print(f"  • {key}: {value}")

# Démarrer le tracking MLflow
with mlflow.start_run(run_name="RandomForest_1"):
    
    # 1. Loguer les paramètres
    mlflow.log_params(params)
    
    # 2. Créer et entraîner le modèle
    print("\n🔄 Entraînement en cours...")
    model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        random_state=params["random_state"]
    )
    
    model.fit(X_train, y_train)
    print("✅ Modèle entraîné!")
    
    # 3. Prédictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # 4. Calculer les métriques
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    # Validation croisée
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    # 5. Loguer les métriques
    metrics = {
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "cv_mean_r2": cv_scores.mean(),
        "cv_std_r2": cv_scores.std()
    }
    
    mlflow.log_metrics(metrics)
    
    # 6. Loguer le modèle
    mlflow.sklearn.log_model(model, "random_forest_model")
    
    # 7. Loguer des tags
    mlflow.set_tags({
        "phase_crisp_dm": "Modeling",
        "preprocessing": "StandardScaler",
        "dataset": "Adelaide_Energy",
        "author": "Sourour Ben Salha",
        "team": "Mariem Abida, Mariem Werhani, Sourour Ben Salha, Hedyl Ben Taher"
    })
    
    # 8. Sauvegarder localement aussi
    joblib.dump(model, 'data/processed/model_rf.pkl')
    
    # 9. Afficher les résultats
    print("\n📊 RÉSULTATS DU MODÈLE:")
    print(f"  • Train RMSE: {train_rmse:.4f}")
    print(f"  • Test RMSE: {test_rmse:.4f}")
    print(f"  • Train R²: {train_r2:.4f}")
    print(f"  • Test R²: {test_r2:.4f}")
    print(f"  • MAE Train: {train_mae:.4f}")
    print(f"  • MAE Test: {test_mae:.4f}")
    print(f"  • CV R² moyenne: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # 10. Interprétation
    print("\n🔍 INTERPRÉTATION:")
    if test_r2 > 0.7:
        print("  ✅ Excellent modèle (R² > 0.7)")
    elif test_r2 > 0.5:
        print("  👍 Bon modèle (0.5 < R² < 0.7)")
    elif test_r2 > 0.3:
        print("  🤔 Modèle moyen (0.3 < R² < 0.5)")
    else:
        print("  ⚠️ Modèle à améliorer (R² < 0.3)")
    
    # Écart train/test
    overfitting = abs(train_r2 - test_r2)
    if overfitting > 0.2:
        print("  ⚠️ Suroptimisation détectée (train > test)")
    else:
        print("  ✅ Pas de suroptimisation majeure")
    
    # Sauvegarder run_id
    run_id = mlflow.active_run().info.run_id
    print(f"\n💾 Run ID: {run_id}")

# ==================== 4. RAPPORT FINAL ====================
print("\n" + "=" * 70)
print("📋 RAPPORT PHASE 4 CRISP-DM")
print("=" * 70)

print(f"""
📊 RÉSUMÉ DE L'EXPÉRIENCE MLflow:

1. 📁 EXPÉRIENCE: {experiment_name}
   • Run ID: {run_id if 'run_id' in locals() else 'N/A'}

2. 📈 MÉTRIQUES PRINCIPALES:
   • Test R²: {test_r2:.4f}
   • Test RMSE: {test_rmse:.4f}
   • Test MAE: {test_mae:.4f}

3. 🤖 MODÈLE: Random Forest
   • n_estimators: {params['n_estimators']}
   • max_depth: {params['max_depth']}
   • Features: {params['features_count']}

4. 📁 FICHIERS GÉNÉRÉS:
   • Modèle: data/processed/model_rf.pkl
   • MLflow logs: mlruns/

5. 🎯 PROCHAINES ÉTAPES (PHASE 5 CRISP-DM):
   • Évaluation approfondie
   • Optimisation hyperparamètres
   • Test d'autres modèles
   • Analyse des erreurs

6. 🔍 COMMANDE POUR VOIR MLFLOW:
   mlflow ui
   Puis ouvrir http://localhost:5000

👥 ÉQUIPE:
Mariem Abida, Mariem Werhani, Sourour Ben Salha, Hedyl Ben Taher
""")