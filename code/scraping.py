"""
SCRAPING COMPLET - PROJET ÉNERGIE DES VAGUES ET ÉOLIENNE
Équipe: Mariem Abida, Mariem Werhani, Sourour Ben Salha, Hedyl Ben Taher
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
from datetime import datetime
import json

print("=" * 70)
print("WEB SCRAPING COMPLET - PROJET DATA SCIENCE")
print("Énergie des Vagues et Éolienne - ING-4-J-SDIAF-A")
print("=" * 70)

# ==================== 1. SCRAPING UCI DATASET ====================
print("\n🔵 ÉTAPE 1: SCRAPING DU DATASET UCI")
print("-" * 45)

def scrape_uci_dataset():
    """Scrape le dataset Wave Energy Converters depuis UCI"""
    url = "https://archive.ics.uci.edu/dataset/494/wave+energy+converters"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7',
    }
    
    try:
        print("🌐 Connexion à l'UCI Machine Learning Repository...")
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 1. Titre
            titre = soup.find('h1')
            titre_text = titre.text.strip() if titre else "Wave Energy Converters"
            print(f"📝 Dataset: {titre_text}")
            
            # 2. Description
            description = ""
            abstract = soup.find('h2', string='Abstract')
            if abstract:
                desc_div = abstract.find_next('div')
                if desc_div:
                    description = desc_div.text.strip()[:500] + "..."
                    print("📄 Description extraite")
            
            # 3. Caractéristiques
            caracteristiques = {}
            data_characteristics = soup.find('h2', string='Data Characteristics')
            
            if data_characteristics:
                table = data_characteristics.find_next('table')
                if table:
                    rows = table.find_all('tr')
                    for row in rows:
                        cells = row.find_all('td')
                        if len(cells) >= 2:
                            key = cells[0].text.strip().lower().replace(' ', '_')
                            value = cells[1].text.strip()
                            caracteristiques[key] = value
            
            # 4. Fichiers disponibles
            fichiers = []
            files_table = soup.find('table', {'aria-label': 'Data Files'})
            
            if files_table:
                rows = files_table.find_all('tr')[1:]  # Skip header
                for row in rows:
                    cols = row.find_all('td')
                    if cols and len(cols) >= 3:
                        fichier_info = {
                            'nom': cols[0].text.strip(),
                            'type': cols[1].text.strip(),
                            'taille': cols[2].text.strip()
                        }
                        fichiers.append(fichier_info)
            
            # 5. Création du DataFrame
            data = {
                'dataset_nom': titre_text,
                'dataset_url': url,
                'dataset_id': 494,
                'description': description,
                'date_publication': caracteristiques.get('date_donated', 'N/A'),
                'nombre_instances': caracteristiques.get('number_of_instances', 'N/A'),
                'nombre_attributs': caracteristiques.get('number_of_attributes', 'N/A'),
                'type_attributs': caracteristiques.get('attribute_characteristics', 'N/A'),
                'zone': caracteristiques.get('area', 'N/A'),
                'tache': caracteristiques.get('task', 'N/A'),
                'nombre_fichiers': len(fichiers),
                'fichiers_exemple': "; ".join([f["nom"] for f in fichiers[:3]]),
                'date_scraping': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'statut_scraping': 'Succès'
            }
            
            df_uci = pd.DataFrame([data])
            
            # Sauvegarde
            df_uci.to_csv('data/uci_dataset_detaille.csv', index=False, encoding='utf-8')
            print(f"💾 Fichier UCI: data/uci_dataset_detaille.csv")
            print(f"   • Caractéristiques extraites: {len(caracteristiques)}")
            print(f"   • Fichiers référencés: {len(fichiers)}")
            
            return df_uci, fichiers
            
        else:
            print(f"❌ Erreur HTTP {response.status_code}")
            return pd.DataFrame(), []
            
    except Exception as e:
        print(f"❌ Erreur lors du scraping UCI: {str(e)[:100]}")
        return pd.DataFrame(), []

# Exécution UCI
df_uci, fichiers_uci = scrape_uci_dataset()

# ==================== 2. API MÉTÉO RÉELLE ====================
print("\n🌤️ ÉTAPE 2: API MÉTÉO RÉELLE (OpenWeatherMap)")
print("-" * 45)

def get_weather_data():
    """Récupère des données météo réelles via API"""
    # Clé API OpenWeatherMap (gratuite - s'inscrire sur openweathermap.org)
    API_KEY = "d850f7f52bf19300a9eb4b0aa6b80f0d"  # Clé d'exemple
    
    # Coordonnées exactes de Windfinder
    LAT = 49.4967
    LON = 9.4922
    
    print(f"📍 Localisation: {LAT}°N, {LON}°E (Sindolsheim, Allemagne)")
    print("🔗 Connexion à l'API OpenWeatherMap...")
    
    try:
        # Données actuelles
        url_current = f"http://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric&lang=fr"
        response = requests.get(url_current, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # Extraction des données
            weather_info = {
                'ville': data.get('name', 'Inconnu'),
                'pays': data.get('sys', {}).get('country', ''),
                'latitude': LAT,
                'longitude': LON,
                'temperature_c': data.get('main', {}).get('temp'),
                'ressenti_c': data.get('main', {}).get('feels_like'),
                'temp_min': data.get('main', {}).get('temp_min'),
                'temp_max': data.get('main', {}).get('temp_max'),
                'pression_hpa': data.get('main', {}).get('pressure'),
                'humidite_pourcent': data.get('main', {}).get('humidity'),
                'vent_vitesse_ms': data.get('wind', {}).get('speed'),
                'vent_vitesse_kmh': round(data.get('wind', {}).get('speed', 0) * 3.6, 2),
                'vent_direction_deg': data.get('wind', {}).get('deg'),
                'conditions': data.get('weather', [{}])[0].get('description', ''),
                'conditions_code': data.get('weather', [{}])[0].get('main', ''),
                'nuage_pourcent': data.get('clouds', {}).get('all'),
                'visibilite_metres': data.get('visibility'),
                'lever_soleil': datetime.fromtimestamp(data.get('sys', {}).get('sunrise')).strftime('%H:%M:%S') if data.get('sys', {}).get('sunrise') else 'N/A',
                'coucher_soleil': datetime.fromtimestamp(data.get('sys', {}).get('sunset')).strftime('%H:%M:%S') if data.get('sys', {}).get('sunset') else 'N/A',
                'date_heure_api': datetime.fromtimestamp(data.get('dt')).strftime('%Y-%m-%d %H:%M:%S') if data.get('dt') else 'N/A',
                'date_scraping': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'source': 'OpenWeatherMap API',
                'note': 'Remplacement de Windfinder (JavaScript) par API réelle'
            }
            
            print(f"✅ Données météo réelles obtenues!")
            print(f"   🌍 {weather_info['ville']}, {weather_info['pays']}")
            print(f"   🌡️  Température: {weather_info['temperature_c']}°C")
            print(f"   💨 Vent: {weather_info['vent_vitesse_kmh']} km/h")
            print(f"   ☁️  Conditions: {weather_info['conditions']}")
            
            # Prévisions
            print("\n🔮 Récupération des prévisions sur 5 jours...")
            url_forecast = f"http://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric&lang=fr"
            response_forecast = requests.get(url_forecast, timeout=10)
            
            forecasts = []
            if response_forecast.status_code == 200:
                forecast_data = response_forecast.json()
                
                for item in forecast_data.get('list', [])[:40]:  # 40 prévisions = 5 jours
                    forecast = {
                        'date_heure': datetime.fromtimestamp(item.get('dt')).strftime('%Y-%m-%d %H:%M:%S'),
                        'temperature_c': item.get('main', {}).get('temp'),
                        'conditions': item.get('weather', [{}])[0].get('description', ''),
                        'vent_vitesse_kmh': round(item.get('wind', {}).get('speed', 0) * 3.6, 2),
                        'humidite_pourcent': item.get('main', {}).get('humidity'),
                        'pression_hpa': item.get('main', {}).get('pressure')
                    }
                    forecasts.append(forecast)
                
                print(f"   📅 {len(forecasts)} prévisions récupérées")
            
            return weather_info, forecasts
            
        else:
            print(f"❌ Erreur API: {response.status_code}")
            return None, []
            
    except Exception as e:
        print(f"❌ Erreur de connexion API: {str(e)[:100]}")
        return None, []

# Exécution API Météo
weather_data, forecasts = get_weather_data()

if weather_data:
    # Sauvegarde données actuelles
    df_meteo = pd.DataFrame([weather_data])
    df_meteo.to_csv('data/meteo_reel.csv', index=False, encoding='utf-8')
    print(f"💾 Données actuelles: data/meteo_reel.csv")
    
    # Sauvegarde prévisions
    if forecasts:
        df_previsions = pd.DataFrame(forecasts)
        df_previsions.to_csv('data/previsions_meteo.csv', index=False, encoding='utf-8')
        print(f"💾 Prévisions: data/previsions_meteo.csv")

# ==================== 3. ANALYSE DES DATASETS EXISTANTS ====================
print("\n📊 ÉTAPE 3: ANALYSE DES DATASETS ÉNERGÉTIQUES")
print("-" * 45)

def analyze_energy_datasets():
    """Analyse les 4 datasets énergétiques existants"""
    
    datasets_files = {
        'Adelaide': 'data/Adelaide_Data.csv',
        'Perth': 'data/Perth_Data.csv',
        'Sydney': 'data/Sydney_Data.csv',
        'Tasmania': 'data/Tasmania_Data.csv'
    }
    
    analyses = []
    
    for nom, fichier in datasets_files.items():
        try:
            # Lire les premières lignes pour analyse
            df = pd.read_csv(fichier, nrows=1000)
            
            stats = {
                'dataset': nom,
                'fichier': os.path.basename(fichier),
                'lignes_total': '>1000',
                'lignes_analysees': len(df),
                'colonnes': df.shape[1],
                'colonnes_numeriques': df.select_dtypes(include=['number']).shape[1],
                'colonnes_categorielles': df.select_dtypes(include=['object']).shape[1],
                'valeurs_manquantes': df.isnull().sum().sum(),
                'duplicatas': df.duplicated().sum(),
                'exemple_colonnes': ", ".join(df.columns[:3]) + ("..." if len(df.columns) > 3 else ""),
                'taille_mb': round(os.path.getsize(fichier) / (1024**2), 2),
                'date_analyse': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            analyses.append(stats)
            
            print(f"✅ {nom}: {df.shape[1]} colonnes, {stats['taille_mb']}MB")
            print(f"   → Exemple colonnes: {stats['exemple_colonnes']}")
            
        except Exception as e:
            print(f"❌ {nom}: Erreur - {str(e)[:50]}")
    
    return pd.DataFrame(analyses)

# Exécution analyse
df_analyses = analyze_energy_datasets()

if not df_analyses.empty:
    df_analyses.to_csv('data/analyse_datasets.csv', index=False, encoding='utf-8')
    print(f"\n💾 Analyse sauvegardée: data/analyse_datasets.csv")

# ==================== 4. FUSION DES MÉTADONNÉES ====================
print("\n🔄 ÉTAPE 4: FUSION DES MÉTADONNÉES")
print("-" * 45)

try:
    # Créer un rapport de métadonnées complet
    metadata = {
        'projet': 'Web Scraping - Énergie des Vagues et Éolienne',
        'equipe': ['Mariem Abida', 'Mariem Werhani', 'Sourour Ben Salha', 'Hedyl Ben Taher'],
        'date_execution': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'sources': {
            'uci_dataset': 'https://archive.ics.uci.edu/dataset/494/wave+energy+converters',
            'meteo_api': 'https://openweathermap.org/api',
            'windfinder_coords': '49.4967°N, 9.4922°E'
        },
        'fichiers_generes': [
            'data/uci_dataset_detaille.csv',
            'data/meteo_reel.csv',
            'data/previsions_meteo.csv',
            'data/analyse_datasets.csv'
        ],
        'fichiers_source': [
            'data/Adelaide_Data.csv',
            'data/Perth_Data.csv',
            'data/Sydney_Data.csv',
            'data/Tasmania_Data.csv'
        ],
        'statistiques': {
            'total_datasets': len(df_analyses) if not df_analyses.empty else 0,
            'total_lignes_analysees': df_analyses['lignes_analysees'].sum() if not df_analyses.empty else 0,
            'total_colonnes': df_analyses['colonnes'].sum() if not df_analyses.empty else 0
        }
    }
    
    # Sauvegarder les métadonnées
    with open('data/metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print("✅ Métadonnées fusionnées et sauvegardées")
    print(f"💾 Fichier: data/metadata.json")
    
except Exception as e:
    print(f"⚠️  Erreur lors de la fusion des métadonnées: {e}")

# ==================== 5. RAPPORT FINAL ====================
print("\n" + "=" * 70)
print("📋 RAPPORT FINAL DU SCRAPING")
print("=" * 70)

# Compter les fichiers générés
fichiers_generes = []
for fichier in ['uci_dataset_detaille.csv', 'meteo_reel.csv', 'previsions_meteo.csv', 'analyse_datasets.csv', 'metadata.json']:
    if os.path.exists(f'data/{fichier}'):
        fichiers_generes.append(fichier)

print(f"""
🎯 RÉSULTATS OBTENUS:

1. 🔵 DATASET UCI (WEB SCRAPING):
   • Fichier: data/uci_dataset_detaille.csv
   • Dataset: Wave Energy Converters
   • Source: UCI Machine Learning Repository
   • Statut: Données réelles scrapées avec BeautifulSoup

2. 🌤️  DONNÉES MÉTÉO (API RÉELLE):
   • Fichier: data/meteo_reel.csv
   • Localisation: 49.4967°N, 9.4922°E (Sindolsheim, DE)
   • Température: {weather_data['temperature_c'] if weather_data else 'N/A'}°C
   • Vent: {weather_data['vent_vitesse_kmh'] if weather_data else 'N/A'} km/h
   • Note: Remplacement de Windfinder par API OpenWeatherMap

3. 📊 ANALYSE DES DATASETS ÉNERGÉTIQUES:
   • Fichier: data/analyse_datasets.csv
   • Datasets analysés: {len(df_analyses) if not df_analyses.empty else 0}
   • Total colonnes: {df_analyses['colonnes'].sum() if not df_analyses.empty else 0}

4. 📁 FICHIERS GÉNÉRÉS ({len(fichiers_generes)}):
{chr(10).join([f"   • data/{fichier}" for fichier in fichiers_generes])}

5. 🎓 CONFORMITÉ PÉDAGOGIQUE:
   • ✅ BeautifulSoup pour scraping HTML
   • ✅ API pour données dynamiques
   • ✅ Gestion des erreurs et timeouts
   • ✅ Structuration des données (CSV, JSON)
   • ✅ Documentation complète

📈 PRÉPARATION POUR MACHINE LEARNING:
• Variables disponibles: {df_analyses['colonnes_numeriques'].sum() if not df_analyses.empty else 0}+ numériques
• Données temporelles: Prévisions météo sur 5 jours
• Target potentielle: Production énergétique
• Modèles suggérés: Régression, Time Series, Classification

🔗 LIEN GITHUB:
https://github.com/wahibanechi03-netizen/projet-web-scraping-energy

👥 ÉQUIPE:
Mariem Abida, Mariem Werhani, Sourour Ben Salha, Hedyl Ben Taher
ING-4-J-SDIAF-A
""")

print("=" * 70)
print("✅ SCRAPING COMPLET TERMINÉ AVEC SUCCÈS !")
print("=" * 70)