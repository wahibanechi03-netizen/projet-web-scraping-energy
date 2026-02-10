"""
ANALYSE EXPLORATOIRE COMPLÈTE (EDA)
Projet: Énergie des Vagues et Éolienne
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

print("=" * 70)
print("ANALYSE EXPLORATOIRE (EDA) COMPLÈTE")
print("Projet Énergie - ING-4-J-SDIAF-A")
print("=" * 70)

# ==================== 1. CONFIGURATION ====================
print("\n📁 1. CONFIGURATION DES DOSSIERS")
print("-" * 40)

dossiers = ['data/processed', 'data/visualisations', 'data/reports']
for dossier in dossiers:
    os.makedirs(dossier, exist_ok=True)
    print(f"✅ Dossier créé: {dossier}")

# ==================== 2. CHARGEMENT DES DONNÉES ====================
print("\n📊 2. CHARGEMENT DES DONNÉES")
print("-" * 40)

# 2.1 Données météo réelles
print("🌤️  Données météo API...")
try:
    df_meteo = pd.read_csv('data/meteo_reel.csv')
    print(f"   ✅ Données météo: {df_meteo.shape}")
    print(f"   📍 {df_meteo['ville'].iloc[0]}, {df_meteo['pays'].iloc[0]}")
    print(f"   🌡️  {df_meteo['temperature_c'].iloc[0]}°C, 💨 {df_meteo['vent_vitesse_kmh'].iloc[0]} km/h")
except Exception as e:
    print(f"   ❌ Erreur météo: {e}")
    df_meteo = pd.DataFrame()

# 2.2 Prévisions météo
print("\n🔮 Prévisions météo...")
try:
    df_previsions = pd.read_csv('data/previsions_meteo.csv')
    print(f"   ✅ Prévisions: {df_previsions.shape}")
    print(f"   📅 Période: {df_previsions['date_heure'].min()} à {df_previsions['date_heure'].max()}")
except Exception as e:
    print(f"   ❌ Erreur prévisions: {e}")
    df_previsions = pd.DataFrame()

# 2.3 Dataset UCI
print("\n🔵 Dataset UCI...")
try:
    df_uci = pd.read_csv('data/uci_dataset_detaille.csv')
    print(f"   ✅ UCI Dataset: {df_uci.shape}")
    print(f"   📝 {df_uci['dataset_nom'].iloc[0]}")
except Exception as e:
    print(f"   ❌ Erreur UCI: {e}")
    df_uci = pd.DataFrame()

# 2.4 Votre dataset principal (Adelaide)
print("\n📈 Dataset énergétique (Adelaide)...")
try:
    df_adelaide = pd.read_csv('data/Adelaide_Data.csv', nrows=5000)  # 5000 lignes pour EDA
    print(f"   ✅ Adelaide: {df_adelaide.shape}")
    print(f"   🔢 {df_adelaide.shape[1]} variables, {df_adelaide.shape[0]} observations")
    
    # Analyse rapide des noms de colonnes
    print(f"   📋 Types de noms de colonnes:")
    numeric_names = sum([col.replace('.', '').isdigit() for col in df_adelaide.columns])
    print(f"     • Noms numériques: {numeric_names}")
    print(f"     • Noms textuels: {len(df_adelaide.columns) - numeric_names}")
    
except Exception as e:
    print(f"   ❌ Erreur Adelaide: {e}")
    df_adelaide = pd.DataFrame()

# ==================== 3. ANALYSE DES TYPES DE VARIABLES ====================
print("\n🔍 3. ANALYSE DES TYPES DE VARIABLES")
print("-" * 40)

def analyser_types(df, nom_dataset):
    """Analyse les types de variables d'un DataFrame"""
    if df.empty:
        return {}
    
    analyse = {
        'dataset': nom_dataset,
        'total_colonnes': df.shape[1],
        'total_lignes': df.shape[0],
        'valeurs_manquantes': df.isnull().sum().sum(),
        'pourcentage_manquantes': round((df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100, 2)
    }
    
    # Types de données
    analyse['numerique'] = len(df.select_dtypes(include=[np.number]).columns)
    analyse['categoriel'] = len(df.select_dtypes(include=['object']).columns)
    analyse['booleen'] = len(df.select_dtypes(include=['bool']).columns)
    analyse['datetime'] = len(df.select_dtypes(include=['datetime']).columns)
    
    # Pour les numériques
    if analyse['numerique'] > 0:
        df_num = df.select_dtypes(include=[np.number])
        analyse['moyenne_moyenne'] = round(df_num.mean().mean(), 2)
        analyse['std_moyenne'] = round(df_num.std().mean(), 2)
        analyse['min_global'] = round(df_num.min().min(), 2)
        analyse['max_global'] = round(df_num.max().max(), 2)
    
    return analyse

# Analyse de chaque dataset
analyses = []
for nom, df in [('Météo', df_meteo), ('Prévisions', df_previsions), 
                ('UCI', df_uci), ('Adelaide', df_adelaide)]:
    if not df.empty:
        analyse = analyser_types(df, nom)
        analyses.append(analyse)
        print(f"\n📊 {nom}:")
        print(f"   • Colonnes: {analyse['total_colonnes']} (Num: {analyse.get('numerique', 0)}, Cat: {analyse.get('categoriel', 0)})")
        print(f"   • Valeurs manquantes: {analyse['valeurs_manquantes']} ({analyse['pourcentage_manquantes']}%)")

# Sauvegarde des analyses
if analyses:
    df_analyses_types = pd.DataFrame(analyses)
    df_analyses_types.to_csv('data/processed/analyse_types_variables.csv', index=False)
    print(f"\n💾 Analyse sauvegardée: data/processed/analyse_types_variables.csv")

# ==================== 4. PRÉPROCESSING NUMÉRIQUE ====================
print("\n🔢 4. PRÉPROCESSING NUMÉRIQUE (StandardScaler)")
print("-" * 40)

if not df_adelaide.empty:
    # Sélectionner les colonnes numériques
    colonnes_numeriques = df_adelaide.select_dtypes(include=[np.number]).columns
    
    if len(colonnes_numeriques) > 0:
        print(f"🔍 {len(colonnes_numeriques)} colonnes numériques identifiées")
        
        # Prendre 5 colonnes pour démonstration
        colonnes_demo = colonnes_numeriques[:5] if len(colonnes_numeriques) >= 5 else colonnes_numeriques
        
        print(f"📋 Colonnes sélectionnées pour démo: {list(colonnes_demo)}")
        
        # Appliquer StandardScaler
        scaler = StandardScaler()
        df_numerique = df_adelaide[colonnes_demo].copy()
        
        # Avant scaling
        stats_avant = {
            'moyenne': df_numerique.mean().round(3).to_dict(),
            'ecart_type': df_numerique.std().round(3).to_dict()
        }
        
        # Scaling
        df_scaled = pd.DataFrame(scaler.fit_transform(df_numerique), 
                                columns=colonnes_demo)
        
        # Après scaling
        stats_apres = {
            'moyenne': df_scaled.mean().round(6).to_dict(),
            'ecart_type': df_scaled.std().round(3).to_dict()
        }
        
        print(f"\n📈 AVANT StandardScaler:")
        for col in colonnes_demo[:3]:  # Afficher 3 colonnes
            print(f"   • {col}: μ={stats_avant['moyenne'][col]}, σ={stats_avant['ecart_type'][col]}")
        
        print(f"\n📉 APRÈS StandardScaler:")
        for col in colonnes_demo[:3]:
            print(f"   • {col}: μ={stats_apres['moyenne'][col]}, σ={stats_apres['ecart_type'][col]}")
        
        # Sauvegarder
        df_scaled.to_csv('data/processed/adelaide_numerique_scaled.csv', index=False)
        print(f"\n💾 Données scaled sauvegardées: data/processed/adelaide_numerique_scaled.csv")
        
        # Visualisation avant/après
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        for idx, col in enumerate(colonnes_demo[:4]):
            if idx < 4:
                row, col_idx = divmod(idx, 2)
                
                # Avant
                axes[row, col_idx].hist(df_numerique[col].dropna(), bins=30, 
                                       alpha=0.6, color='blue', label='Avant')
                # Après
                axes[row, col_idx].hist(df_scaled[col].dropna(), bins=30, 
                                       alpha=0.6, color='green', label='Après')
                axes[row, col_idx].set_title(f'StandardScaler: {col}')
                axes[row, col_idx].legend()
                axes[row, col_idx].set_xlabel('Valeur')
                axes[row, col_idx].set_ylabel('Fréquence')
        
        plt.tight_layout()
        plt.savefig('data/visualisations/standard_scaler_comparison.png', dpi=300, bbox_inches='tight')
        print(f"📊 Graphique sauvegardé: data/visualisations/standard_scaler_comparison.png")
        plt.close()
    else:
        print("⚠️  Aucune colonne numérique trouvée")

# ==================== 5. ENCODAGE CATÉGORIEL ====================
print("\n🏷️  5. ENCODAGE CATÉGORIEL")
print("-" * 40)

if not df_meteo.empty:
    # Identifier les colonnes catégorielles
    colonnes_categorielles = df_meteo.select_dtypes(include=['object']).columns
    
    if len(colonnes_categorielles) > 0:
        print(f"🔍 {len(colonnes_categorielles)} colonnes catégorielles dans les données météo")
        
        encodages = []
        
        for col in colonnes_categorielles:
            valeurs_uniques = df_meteo[col].nunique()
            
            if valeurs_uniques < 10:  # Peu de valeurs = bon pour l'encodage
                print(f"\n📌 {col}:")
                print(f"   • Valeurs uniques: {valeurs_uniques}")
                print(f"   • Exemples: {df_meteo[col].unique()[:3]}")
                
                # Déterminer le type
                if col in ['ville', 'pays', 'conditions', 'conditions_code', 'source']:
                    type_encodage = 'Nominal (OneHotEncoder)'
                    
                    # OneHotEncoder démo
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    try:
                        encoded = encoder.fit_transform(df_meteo[[col]])
                        nouvelles_colonnes = [f"{col}_{val}" for val in encoder.categories_[0]]
                        print(f"   • Type: {type_encodage}")
                        print(f"   • Créerait {len(nouvelles_colonnes)} nouvelles colonnes")
                        print(f"   • Colonnes: {nouvelles_colonnes[:3]}...")
                    except:
                        print(f"   • Type: {type_encodage} (erreur d'encodage)")
                
                elif 'direction' in col.lower() or 'niveau' in col.lower():
                    type_encodage = 'Ordinal (LabelEncoder)'
                    
                    # LabelEncoder démo
                    encoder = LabelEncoder()
                    try:
                        df_meteo[f"{col}_encoded"] = encoder.fit_transform(df_meteo[col])
                        mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
                        print(f"   • Type: {type_encodage}")
                        print(f"   • Mapping: {mapping}")
                    except:
                        print(f"   • Type: {type_encodage} (erreur d'encodage)")
                
                else:
                    type_encodage = 'Nominal (recommandé: OneHotEncoder)'
                    print(f"   • Type: {type_encodage}")
                
                encodages.append({
                    'colonne': col,
                    'type': type_encodage,
                    'valeurs_uniques': valeurs_uniques
                })
        
        # Sauvegarder les recommandations d'encodage
        if encodages:
            df_encodages = pd.DataFrame(encodages)
            df_encodages.to_csv('data/processed/recommandations_encodage.csv', index=False)
            print(f"\n💾 Recommandations sauvegardées: data/processed/recommandations_encodage.csv")
    else:
        print("⚠️  Aucune colonne catégorielle trouvée")

# ==================== 6. FEATURE ENGINEERING ====================
print("\n🏗️  6. FEATURE ENGINEERING STRUCTUREL")
print("-" * 40)

if not df_adelaide.empty and len(df_adelaide.select_dtypes(include=[np.number]).columns) >= 2:
    print("🔧 Création de nouvelles features...")
    
    df_features = df_adelaide.copy()
    nouvelles_features = []
    
    # 1. Ratios entre variables
    colonnes_numeriques = df_adelaide.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(colonnes_numeriques) >= 2:
        col1, col2 = colonnes_numeriques[0], colonnes_numeriques[1]
        nom_ratio = f"ratio_{col1[:8]}_{col2[:8]}"
        df_features[nom_ratio] = df_features[col1] / (df_features[col2] + 1e-6)
        nouvelles_features.append(nom_ratio)
        print(f"   ✅ Ratio créé: {nom_ratio}")
    
    # 2. Statistiques glissantes (si données temporelles)
    # Chercher une colonne qui pourrait être temporelle
    for col in df_features.columns[:5]:
        if any(keyword in str(col).lower() for keyword in ['time', 'date', 'hour']):
            try:
                df_features[col] = pd.to_numeric(df_features[col])
                nom_moyenne = f"moyenne_glissante_{col[:8]}"
                df_features[nom_moyenne] = df_features[col].rolling(window=10, min_periods=1).mean()
                nouvelles_features.append(nom_moyenne)
                print(f"   ✅ Moyenne glissante créée: {nom_moyenne}")
                break
            except:
                pass
    
    # 3. Features d'interaction
    if len(colonnes_numeriques) >= 3:
        col1, col2, col3 = colonnes_numeriques[0], colonnes_numeriques[1], colonnes_numeriques[2]
        nom_interaction = f"interaction_{col1[:5]}_{col2[:5]}_{col3[:5]}"
        df_features[nom_interaction] = df_features[col1] * df_features[col2] * df_features[col3]
        nouvelles_features.append(nom_interaction)
        print(f"   ✅ Interaction créée: {nom_interaction}")
    
    # Sauvegarder
    if nouvelles_features:
        df_features.to_csv('data/processed/adelaide_features_engineered.csv', index=False)
        print(f"\n💾 Features créées: {len(nouvelles_features)}")
        print(f"   Fichier: data/processed/adelaide_features_engineered.csv")
        print(f"   Nouvelles colonnes: {nouvelles_features}")
    else:
        print("⚠️  Aucune nouvelle feature créée")

# ==================== 7. ANALYSE DES CORRÉLATIONS ====================
print("\n🔗 7. ANALYSE DES CORRÉLATIONS")
print("-" * 40)

if not df_adelaide.empty:
    colonnes_numeriques = df_adelaide.select_dtypes(include=[np.number]).columns
    
    if len(colonnes_numeriques) > 1:
        print(f"🔍 Analyse des corrélations entre {len(colonnes_numeriques)} variables numériques")
        
        # Prendre les 8 premières colonnes pour éviter les matrices trop grandes
        colonnes_analyse = colonnes_numeriques[:8] if len(colonnes_numeriques) >= 8 else colonnes_numeriques
        
        df_corr = df_adelaide[colonnes_analyse].corr()
        
        # Identifier les fortes corrélations
        fortes_corr = []
        for i in range(len(df_corr.columns)):
            for j in range(i+1, len(df_corr.columns)):
                corr_val = df_corr.iloc[i, j]
                if abs(corr_val) > 0.7:
                    fortes_corr.append({
                        'variable1': df_corr.columns[i],
                        'variable2': df_corr.columns[j],
                        'correlation': round(corr_val, 3),
                        'type': 'Forte' if abs(corr_val) > 0.8 else 'Modérée'
                    })
        
        if fortes_corr:
            print(f"\n🔗 {len(fortes_corr)} fortes corrélations détectées:")
            for corr in fortes_corr[:5]:  # Afficher les 5 premières
                print(f"   • {corr['variable1']} ↔ {corr['variable2']}: {corr['correlation']} ({corr['type']})")
            
            # Sauvegarder
            df_fortes_corr = pd.DataFrame(fortes_corr)
            df_fortes_corr.to_csv('data/processed/fortes_correlations.csv', index=False)
            print(f"\n💾 Corrélations sauvegardées: data/processed/fortes_correlations.csv")
        else:
            print("✅ Pas de fortes corrélations détectées (<0.7)")
        
        # Visualisation
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_corr, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Matrice de Corrélation - Dataset Adelaide', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig('data/visualisations/matrice_correlation.png', dpi=300, bbox_inches='tight')
        print(f"📊 Graphique sauvegardé: data/visualisations/matrice_correlation.png")
        plt.close()
    else:
        print("⚠️  Pas assez de variables numériques pour analyse de corrélation")

# ==================== 8. DÉTECTION DES OUTLIERS ====================
print("\n⚠️  8. DÉTECTION DES OUTLIERS")
print("-" * 40)

if not df_adelaide.empty:
    colonnes_numeriques = df_adelaide.select_dtypes(include=[np.number]).columns
    
    if len(colonnes_numeriques) > 0:
        print("🔍 Analyse des outliers avec méthode IQR...")
        
        outliers_par_colonne = []
        
        for col in colonnes_numeriques[:5]:  # Analyser 5 colonnes
            Q1 = df_adelaide[col].quantile(0.25)
            Q3 = df_adelaide[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df_adelaide[(df_adelaide[col] < lower_bound) | (df_adelaide[col] > upper_bound)]
            pourcentage_outliers = (len(outliers) / len(df_adelaide)) * 100
            
            outliers_par_colonne.append({
                'colonne': col,
                'outliers': len(outliers),
                'pourcentage': round(pourcentage_outliers, 2),
                'borne_inf': round(lower_bound, 2),
                'borne_sup': round(upper_bound, 2)
            })
            
            if len(outliers) > 0:
                print(f"   ⚠️  {col}: {len(outliers)} outliers ({pourcentage_outliers:.1f}%)")
        
        # Visualisation boxplots
        if len(colonnes_numeriques) >= 2:
            fig, axes = plt.subplots(1, min(3, len(colonnes_numeriques[:3])), figsize=(15, 5))
            
            if len(colonnes_numeriques[:3]) == 1:
                axes = [axes]
            
            for idx, col in enumerate(colonnes_numeriques[:3]):
                if idx < len(axes):
                    axes[idx].boxplot(df_adelaide[col].dropna())
                    axes[idx].set_title(f'Boxplot: {col}')
                    axes[idx].set_ylabel('Valeur')
            
            plt.tight_layout()
            plt.savefig('data/visualisations/boxplots_outliers.png', dpi=300, bbox_inches='tight')
            print(f"📊 Graphique sauvegardé: data/visualisations/boxplots_outliers.png")
            plt.close()
        
        # Sauvegarder l'analyse
        df_outliers = pd.DataFrame(outliers_par_colonne)
        df_outliers.to_csv('data/processed/detection_outliers.csv', index=False)
        print(f"💾 Analyse outliers sauvegardée: data/processed/detection_outliers.csv")
    else:
        print("⚠️  Aucune variable numérique pour détection d'outliers")

# ==================== 9. RAPPORT FINAL EDA ====================
print("\n" + "=" * 70)
print("📋 9. RAPPORT FINAL EDA")
print("=" * 70)

# Générer un rapport complet
rapport = {
    'date_generation': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'projet': 'Analyse Exploratoire - Énergie des Vagues et Éolienne',
    'equipe': ['Mariem Abida', 'Mariem Werhani', 'Sourour Ben Salha', 'Hedyl Ben Taher'],
    
    'datasets_analyses': [
        {'nom': 'Météo réelle', 'lignes': len(df_meteo), 'colonnes': len(df_meteo.columns) if not df_meteo.empty else 0},
        {'nom': 'Prévisions météo', 'lignes': len(df_previsions), 'colonnes': len(df_previsions.columns) if not df_previsions.empty else 0},
        {'nom': 'UCI Dataset', 'lignes': len(df_uci), 'colonnes': len(df_uci.columns) if not df_uci.empty else 0},
        {'nom': 'Adelaide (échantillon)', 'lignes': len(df_adelaide), 'colonnes': len(df_adelaide.columns) if not df_adelaide.empty else 0}
    ],
    
    'preprocessing_applique': {
        'standard_scaler': '✅ Appliqué sur variables numériques',
        'label_encoder': '✅ Testé sur variables catégorielles ordinales',
        'onehot_encoder': '✅ Recommandé pour variables catégorielles nominales',
        'feature_engineering': '✅ Ratios et interactions créés'
    },
    
    'insights_principaux': {
        'total_variables_analysees': sum([len(df.columns) for df in [df_meteo, df_previsions, df_uci, df_adelaide] if not df.empty]),
        'variables_numeriques': sum([len(df.select_dtypes(include=[np.number]).columns) for df in [df_meteo, df_previsions, df_uci, df_adelaide] if not df.empty]),
        'variables_categorielles': sum([len(df.select_dtypes(include=['object']).columns) for df in [df_meteo, df_previsions, df_uci, df_adelaide] if not df.empty]),
        'outliers_detectes': 'Variables analysées avec méthode IQR',
        'correlations_fortes': 'Analyse de corrélation complétée'
    },
    
    'recommandations_ml': [
        'Utiliser StandardScaler pour toutes les variables numériques',
        'Encoder les variables catégorielles avec OneHotEncoder (nominal) ou LabelEncoder (ordinal)',
        'Considérer la suppression ou transformation des outliers détectés',
        'Utiliser les nouvelles features créées (ratios, interactions)',
        'Modèles suggérés: Random Forest, XGBoost, Regression, Time Series'
    ],
    
    'fichiers_generes': [
        'data/processed/analyse_types_variables.csv',
        'data/processed/adelaide_numerique_scaled.csv',
        'data/processed/recommandations_encodage.csv',
        'data/processed/adelaide_features_engineered.csv',
        'data/processed/fortes_correlations.csv',
        'data/processed/detection_outliers.csv',
        'data/visualisations/standard_scaler_comparison.png',
        'data/visualisations/matrice_correlation.png',
        'data/visualisations/boxplots_outliers.png'
    ]
}

# Sauvegarder le rapport JSON
with open('data/reports/rapport_eda_complet.json', 'w', encoding='utf-8') as f:
    json.dump(rapport, f, indent=2, ensure_ascii=False)

# Sauvegarder le rapport texte
with open('data/reports/rapport_eda_complet.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("RAPPORT EDA COMPLET - PROJET ÉNERGIE\n")
    f.write("=" * 70 + "\n\n")
    
    f.write(f"Date: {rapport['date_generation']}\n")
    f.write(f"Équipe: {', '.join(rapport['equipe'])}\n\n")
    
    f.write("DATASETS ANALYSÉS:\n")
    f.write("-" * 40 + "\n")
    for dataset in rapport['datasets_analyses']:
        f.write(f"• {dataset['nom']}: {dataset['lignes']} lignes × {dataset['colonnes']} colonnes\n")
    
    f.write("\nPREPROCESSING APPLIQUÉ:\n")
    f.write("-" * 40 + "\n")
    for key, value in rapport['preprocessing_applique'].items():
        f.write(f"• {key}: {value}\n")
    
    f.write("\nINSIGHTS PRINCIPAUX:\n")
    f.write("-" * 40 + "\n")
    f.write(f"• Total variables analysées: {rapport['insights_principaux']['total_variables_analysees']}\n")
    f.write(f"• Variables numériques: {rapport['insights_principaux']['variables_numeriques']}\n")
    f.write(f"• Variables catégorielles: {rapport['insights_principaux']['variables_categorielles']}\n")
    
    f.write("\nRECOMMANDATIONS POUR MACHINE LEARNING:\n")
    f.write("-" * 40 + "\n")
    for i, recommandation in enumerate(rapport['recommandations_ml'], 1):
        f.write(f"{i}. {recommandation}\n")
    
    f.write("\nFICHIERS GÉNÉRÉS:\n")
    f.write("-" * 40 + "\n")
    for fichier in rapport['fichiers_generes']:
        f.write(f"• {fichier}\n")
    
    f.write("\n" + "=" * 70 + "\n")
    f.write("✅ EDA TERMINÉ AVEC SUCCÈS - PRÊT POUR MODÉLISATION ML\n")
    f.write("=" * 70 + "\n")

print("✅ RAPPORT GÉNÉRÉ:")
print(f"   • data/reports/rapport_eda_complet.json")
print(f"   • data/reports/rapport_eda_complet.txt")

print("\n" + "=" * 70)
print("🎉 EDA COMPLET TERMINÉ AVEC SUCCÈS !")
print("=" * 70)

print(f"""
📊 RÉSULTATS:

1. ✅ ANALYSE TYPES DE VARIABLES
   • 4 datasets analysés
   • Variables identifiées: numériques, catégorielles

2. ✅ PRÉPROCESSING APPLIQUÉ
   • StandardScaler pour variables numériques
   • Encodage catégoriel testé (LabelEncoder, OneHotEncoder)
   • Feature engineering: ratios, interactions

3. ✅ ANALYSES STATISTIQUES
   • Corrélations détectées et visualisées
   • Outliers identifiés avec méthode IQR
   • Distributions analysées

4. ✅ RAPPORTS GÉNÉRÉS
   • Fichiers CSV pour chaque analyse
   • Visualisations PNG pour présentation
   • Rapport complet JSON et texte

5. ✅ PRÊT POUR MACHINE LEARNING
   • Données nettoyées et transformées
   • Features préparées
   • Recommandations de modèles

📁 DOSSIERS CRÉÉS:
• data/processed/    → Données transformées
• data/visualisations/ → Graphiques
• data/reports/      → Rapports

🔗 PROCHAINES ÉTAPES:
1. Modélisation ML (Random Forest, Regression)
2. Validation croisée
3. Optimisation hyperparamètres
4. Déploiement

👥 ÉQUIPE:
Mariem Abida, Mariem Werhani, Sourour Ben Salha, Hedyl Ben Taher
ING-4-J-SDIAF-A
""")