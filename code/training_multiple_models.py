"""
TRAINING DE PLUSIEURS MODÈLES AVEC MLFLOW
Phase 4 CRISP-DM: Modeling (Multiple Models)
"""
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("TRAINING MULTI-MODÈLES AVEC MLFLOW")
print("Phase 4 CRISP-DM: Comparaison de plusieurs algorithmes")
print("=" * 70)

# ==================== 1. CHARGEMENT DES DONNÉES ====================
print("\n1. 📊 CHARGEMENT DES DONNÉES PREPROCESSÉES")
print("-" * 40)

# Charger les données préparées
try:
    X_train = pd.read_csv('data/processed/X_train_scaled.csv')
    X_test = pd.read_csv('data/processed/X_test_scaled.csv')
    
    # Charger les targets (à créer si nécessaire)
    # Pour cet exemple, on simule des targets
    np.random.seed(42)
    y_train = np.random.randn(len(X_train)) * 100 + 500
    y_test = np.random.randn(len(X_test)) * 100 + 500
    
    print(f"✅ X_train: {X_train.shape}")
    print(f"✅ X_test: {X_test.shape}")
    print(f"✅ y_train: {y_train.shape}")
    print(f"✅ y_test: {y_test.shape}")
    
except Exception as e:
    print(f"❌ Erreur chargement: {e}")
    # Créer des données factices pour la démo
    print("⚠️  Création de données factices pour démonstration")
    X_train = pd.DataFrame(np.random.randn(1000, 10), columns=[f'feat_{i}' for i in range(10)])
    X_test = pd.DataFrame(np.random.randn(200, 10), columns=[f'feat_{i}' for i in range(10)])
    y_train = np.random.randn(1000) * 100 + 500
    y_test = np.random.randn(200) * 100 + 500

# ==================== 2. CONFIGURATION MLFLOW ====================
print("\n2. 🔧 CONFIGURATION MLFLOW")
print("-" * 40)

# Créer dossier pour les logs
os.makedirs('mlruns', exist_ok=True)

# Nom de l'expérience
experiment_name = f"Comparaison_Modeles_Energie_{datetime.now().strftime('%Y%m%d_%H%M')}"
mlflow.set_experiment(experiment_name)
print(f"📁 Expérience MLflow: {experiment_name}")

# ==================== 3. DÉFINITION DES MODÈLES ====================
print("\n3. 🤖 LISTE DES MODÈLES À TESTER")
print("-" * 40)

models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'DecisionTree': DecisionTreeRegressor(max_depth=10, random_state=42),
    'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
    'SVR': SVR(kernel='rbf', C=100, gamma=0.1)
}

print(f"📋 {len(models)} modèles à entraîner:")
for i, (name, model) in enumerate(models.items(), 1):
    print(f"   {i}. {name}")

# ==================== 4. ENTRAÎNEMENT DE TOUS LES MODÈLES ====================
print("\n4. 🚀 ENTRAÎNEMENT ET COMPARAISON")
print("-" * 40)

results = []

for model_name, model in models.items():
    print(f"\n🔵 Entraînement: {model_name}...")
    
    with mlflow.start_run(run_name=model_name):
        
        # 1. Loguer les paramètres du modèle
        params = model.get_params()
        mlflow.log_params(params)
        
        # 2. Entraînement
        model.fit(X_train, y_train)
        
        # 3. Prédictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # 4. Métriques
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        # 5. Validation croisée
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        
        # 6. Loguer les métriques
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
        
        # 7. Loguer le modèle
        mlflow.sklearn.log_model(model, f"{model_name}_model")
        
        # 8. Tags
        mlflow.set_tags({
            "model_family": model_name,
            "phase_crisp_dm": "Modeling",
            "features_count": X_train.shape[1],
            "train_samples": X_train.shape[0]
        })
        
        # 9. Sauvegarder les résultats
        results.append({
            'Modèle': model_name,
            'Train R²': round(train_r2, 4),
            'Test R²': round(test_r2, 4),
            'Train RMSE': round(train_rmse, 2),
            'Test RMSE': round(test_rmse, 2),
            'Train MAE': round(train_mae, 2),
            'Test MAE': round(test_mae, 2),
            'CV R²': round(cv_scores.mean(), 4),
            'Temps': datetime.now().strftime('%H:%M:%S')
        })
        
        print(f"   ✅ Test R²: {test_r2:.4f} | Test RMSE: {test_rmse:.2f}")

# ==================== 5. COMPARAISON DES RÉSULTATS ====================
print("\n5. 📊 COMPARAISON DES MODÈLES")
print("-" * 40)

# Créer DataFrame des résultats
df_results = pd.DataFrame(results)
df_results = df_results.sort_values('Test R²', ascending=False)

print("\n🏆 CLASSEMENT DES MODÈLES (par Test R²):")
print(df_results.to_string(index=False))

# Sauvegarder les résultats
df_results.to_csv('data/reports/comparaison_modeles.csv', index=False)
print(f"\n💾 Résultats sauvegardés: data/reports/comparaison_modeles.csv")

# ==================== 6. IDENTIFICATION DU MEILLEUR MODÈLE ====================
print("\n6. 🥇 MEILLEUR MODÈLE")
print("-" * 40)

best_model = df_results.iloc[0]
print(f"\n🏆 Meilleur modèle: {best_model['Modèle']}")
print(f"   • Test R²: {best_model['Test R²']}")
print(f"   • Test RMSE: {best_model['Test RMSE']}")
print(f"   • CV R²: {best_model['CV R²']}")

# ==================== 7. SAUVEGARDE DU MEILLEUR MODÈLE ====================
print("\n7. 💾 SAUVEGARDE DU MEILLEUR MODÈLE")
print("-" * 40)

best_model_name = best_model['Modèle']
best_model_instance = models[best_model_name]

# Sauvegarder avec joblib
joblib.dump(best_model_instance, 'data/processed/best_model.pkl')
print(f"✅ Meilleur modèle sauvegardé: data/processed/best_model.pkl")

# ==================== 8. RAPPORT FINAL ====================
print("\n" + "=" * 70)
print("📋 RAPPORT FINAL - COMPARAISON MULTI-MODÈLES")
print("=" * 70)

print(f"""
📊 RÉSULTATS DE LA COMPARAISON:

1. 📁 EXPÉRIENCE MLflow:
   • Nom: {experiment_name}
   • {len(models)} modèles entraînés
   • {len(df_results)} runs enregistrés

2. 🏆 CLASSEMENT:
{df_results[['Modèle', 'Test R²', 'Test RMSE']].to_string(index=False)}

3. 🥇 MEILLEUR MODÈLE:
   • Modèle: {best_model['Modèle']}
   • Test R²: {best_model['Test R²']}
   • Test RMSE: {best_model['Test RMSE']}
   • Train R²: {best_model['Train R²']}
   • Écart train/test: {abs(best_model['Train R²'] - best_model['Test R²']):.4f}

4. 📈 ANALYSE:
   • Écart-type entre modèles: {df_results['Test R²'].std():.4f}
   • Meilleur score: {df_results['Test R²'].max():.4f}
   • Pire score: {df_results['Test R²'].min():.4f}

5. 📁 FICHIERS GÉNÉRÉS:
   • data/reports/comparaison_modeles.csv
   • data/processed/best_model.pkl
   • mlruns/ (logs MLflow)

6. 🔍 COMMANDE POUR VOIR MLflow:
   mlflow ui
   Puis ouvrir http://localhost:5000

👥 ÉQUIPE:
Mariem Abida, Mariem Werhani, Sourour Ben Salha, Hedyl Ben Taher
""")