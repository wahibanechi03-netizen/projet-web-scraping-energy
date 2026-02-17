"""
Sauvegarder le vrai meilleur modèle (GradientBoosting)
"""
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("SAUVEGARDE DU VRAI MEILLEUR MODÈLE")
print("=" * 60)

# 1. Charger les données
print("\n1. 📊 CHARGEMENT DES DONNÉES")
df = pd.read_csv('data/Adelaide_Data.csv', nrows=10000)
X = df.iloc[:, :-1]  # Toutes sauf dernière colonne
y = df.iloc[:, -1]   # Dernière colonne = target

print(f"Features: {X.shape}")
print(f"Target: {y.shape}")

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Entraîner GradientBoosting (le meilleur)
print("\n2. 🤖 ENTRAÎNEMENT GRADIENTBOOSTING...")
best_model = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

best_model.fit(X_train_scaled, y_train)

# 5. Évaluer
train_score = best_model.score(X_train_scaled, y_train)
test_score = best_model.score(X_test_scaled, y_test)
print(f"Train R²: {train_score:.4f}")
print(f"Test R²: {test_score:.4f}")

# 6. Sauvegarder avec un nom clair
print("\n3. 💾 SAUVEGARDE")
joblib.dump(best_model, 'data/processed/gradient_boosting_best.pkl')
joblib.dump(scaler, 'data/processed/scaler_gb.pkl')

print("✅ Modèle sauvegardé: data/processed/gradient_boosting_best.pkl")
print("✅ Scaler sauvegardé: data/processed/scaler_gb.pkl")