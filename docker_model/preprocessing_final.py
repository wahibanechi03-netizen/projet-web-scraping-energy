"""
PREPROCESSING COMPLET POUR MACHINE LEARNING
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("PREPROCESSING POUR MACHINE LEARNING")
print("=" * 70)

# ==================== 1. CHARGEMENT ====================
print("\n1. 📊 CHARGEMENT DES DONNÉES")
print("-" * 40)

# Charger Adelaide (supposé avoir une colonne target)
df = pd.read_csv('data/Adelaide_Data.csv', nrows=10000)  # 10000 lignes pour test

print(f"Shape: {df.shape}")
print(f"Colonnes: {df.columns[:10].tolist()}...")

# ==================== 2. IDENTIFICATION TARGET ====================
print("\n2. 🎯 IDENTIFICATION DE LA VARIABLE CIBLE")
print("-" * 40)

# Supposons que la dernière colonne est la target
target_col = df.columns[-1]
print(f"Target sélectionnée: {target_col}")

# Séparation
X = df.drop(columns=[target_col])
y = df[target_col]

print(f"Features (X): {X.shape}")
print(f"Target (y): {y.shape}")

# ==================== 3. SPLIT TRAIN/TEST ====================
print("\n3. 🔀 SPLIT TRAIN/TEST (80/20)")
print("-" * 40)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train: {X_train.shape[0]} samples")
print(f"Test: {X_test.shape[0]} samples")

# ==================== 4. STANDARDSCALER ====================
print("\n4. 🔢 STANDARDSCALER")
print("-" * 40)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("AVANT StandardScaler:")
print(f"  Moyenne: {X_train.mean().mean():.2f}")
print(f"  Écart-type: {X_train.std().mean():.2f}")

print("\nAPRÈS StandardScaler:")
print(f"  Moyenne: {X_train_scaled.mean():.6f}")
print(f"  Écart-type: {X_train_scaled.std():.2f}")

# ==================== 5. PIPELINE ML ====================
print("\n5. 🤖 PIPELINE MACHINE LEARNING")
print("-" * 40)

# Création pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(n_estimators=50, random_state=42))
])

# Entraînement
print("Entraînement du modèle...")
pipeline.fit(X_train, y_train)

# Prédictions
y_pred = pipeline.predict(X_test)

# Évaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 RÉSULTATS:")
print(f"   • MSE: {mse:.2f}")
print(f"   • R²: {r2:.4f}")

# ==================== 6. SAUVEGARDE ====================
print("\n6. 💾 SAUVEGARDE")
print("-" * 40)

# Sauvegarder données transformées
pd.DataFrame(X_train_scaled, columns=X.columns).to_csv(
    'data/processed/X_train_scaled.csv', index=False
)
pd.DataFrame(X_test_scaled, columns=X.columns).to_csv(
    'data/processed/X_test_scaled.csv', index=False
)

# Sauvegarder scaler
import joblib
joblib.dump(scaler, 'data/processed/scaler.pkl')

print("✅ Données sauvegardées:")
print("   • data/processed/X_train_scaled.csv")
print("   • data/processed/X_test_scaled.csv")
print("   • data/processed/scaler.pkl")

# ==================== 7. RAPPORT ====================
print("\n" + "=" * 70)
print("📋 RAPPORT PREPROCESSING")
print("=" * 70)

print(f"""
RÉSULTATS PREPROCESSING:

1. DONNÉES ORIGINALES:
   • Features: {X.shape[1]} variables
   • Samples: {X.shape[0]} observations

2. SPLIT:
   • Train: {X_train.shape[0]} ({X_train.shape[0]/X.shape[0]*100:.1f}%)
   • Test: {X_test.shape[0]} ({X_test.shape[0]/X.shape[0]*100:.1f}%)

3. TRANSFORMATION:
   • StandardScaler appliqué
   • Moyenne après scaling: {X_train_scaled.mean():.6f}
   • Écart-type après scaling: {X_train_scaled.std():.2f}

4. PREMIER MODÈLE:
   • Random Forest (50 arbres)
   • R² Score: {r2:.4f}
   • MSE: {mse:.2f}

5. FICHIERS GÉNÉRÉS:
   • X_train_scaled.csv
   • X_test_scaled.csv
   • scaler.pkl

✅ PRÊT POUR MODÉLISATION AVANCÉE!
""")