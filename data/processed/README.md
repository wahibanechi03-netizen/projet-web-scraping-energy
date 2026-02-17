## 📋 DESCRIPTION
Ce dossier contient tous les fichiers générés par nos scripts de preprocessing et d'analyse EDA.

---

## ✅ FICHIERS SUR GITHUB (versionnés)

Ces fichiers sont **sur GitHub** car ils sont petits et importants pour la documentation:

| Fichier | Description |
|---------|-------------|
| `analyse_types_variables.csv` | Types de variables identifiés par dataset |
| `detection_outliers.csv` | Résultats de détection des outliers |
| `recommandations_encodage.csv` | Recommandations pour l'encodage catégoriel |
| `gradient_boosting_best.pkl` | ✅ Meilleur modèle GradientBoosting (R²=0.85) |
| `scaler_gb.pkl` | Scaler correspondant au meilleur modèle |

---

## 🔧 FICHIERS LOCAUX SEULEMENT (non versionnés)

Ces fichiers sont **UNIQUEMENT sur votre PC** (exclus par `.gitignore`):

### 📊 Données transformées (fichiers volumineux)
| Fichier | Description | Comment re-générer |
|---------|-------------|-------------------|
| `adelaide_features_engineered.csv` | Features engineering (ratios, interactions) | `python code/eda_complet.py` |
| `adelaide_final.csv` | Dataset final après preprocessing | `python code/eda_complet.py` |
| `adelaide_numerique_scaled.csv` | Variables numériques normalisées | `python code/eda_complet.py` |
| `adelaide_scaled.csv` | Dataset complet normalisé | `python code/eda_complet.py` |
| `X_train_scaled.csv` | Features d'entraînement (80%) | `python code/preprocessing_final.py` |
| `X_test_scaled.csv` | Features de test (20%) | `python code/preprocessing_final.py` |

### 🤖 Modèles Machine Learning (anciens)
| Fichier | Description | Comment re-générer |
|---------|-------------|-------------------|
| `best_model.pkl` | Ancien modèle Lasso | `python code/training_multiple_models.py` |
| `best_model_reel.pkl` | Ancien modèle LinearRegression | `python code/training_multiple_models_corrige.py` |
| `model_rf.pkl` | Modèle Random Forest | `python code/training_mlflow.py` |
| `scaler.pkl` | Ancien scaler | `python code/preprocessing_final.py` |
| `scaler_reel.pkl` | Ancien scaler | `python code/preprocessing_final.py` |

---

## 🎯 POURQUOI CES FICHIERS NE SONT PAS SUR GITHUB?

| Raison | Explication |
|--------|-------------|
| **Taille** | Certains fichiers CSV font plusieurs MB |
| **Binaires** | Les fichiers `.pkl` volumineux ne sont pas adaptés à Git |
| **Reproductibilité** | Ils peuvent être re-générés par les scripts |
| **Pratique** | Bonne pratique en data science |

---

## 🚀 COMMENT RE-GÉNÉRER CES FICHIERS

```bash
# 1. Pour re-générer tous les fichiers EDA
python code/eda_complet.py

# 2. Pour re-générer les fichiers preprocessing
python code/preprocessing_final.py

# 3. Pour re-générer et comparer tous les modèles
python code/training_multiple_models.py

# 4. Pour sauvegarder le meilleur modèle (GradientBoosting)
python code/save_best_model.py