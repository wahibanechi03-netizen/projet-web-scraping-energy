# Projet Web Scraping - Énergie des Vagues

##  Équipe
- **Sourour Ben Salha** (wahibanechi03@gmail.com)
- **Mariem Abida**
- **Mariem Werhani**
- **Hedyl Ben Taher**

**Groupe:** ING-4-J-SDIAF-A

---

##  LIENS DU PROJET
- **GitHub Repository:** https://github.com/wahibanechi03-netizen/projet-web-scraping-energy
- **Dataset UCI:** https://archive.ics.uci.edu/dataset/494/wave+energy+converters
- **OpenWeatherMap API:** https://openweathermap.org/api
- **Documentation BeautifulSoup:** https://www.crummy.com/software/BeautifulSoup/bs4/doc/
- **Pandas Documentation:** https://pandas.pydata.org/docs/
- **Scikit-learn:** https://scikit-learn.org/stable/
- **MLflow Documentation:** https://mlflow.org/docs/latest/index.html
- **Docker Documentation:** https://docs.docker.com/

---

## 🎯 Objectif du Projet
Collecte et analyse de données sur l'énergie des vagues et éolienne pour des applications de Machine Learning, en utilisant des techniques de web scraping et d'API.

---

## Sources de Données

### 1. UCI Machine Learning Repository
- **Dataset:** Wave Energy Converters
- **URL:** https://archive.ics.uci.edu/dataset/494/wave+energy+converters
- **Méthode:** Web scraping avec BeautifulSoup
- **Fichier généré:** `data/uci_dataset_detaille.csv`

### 2. OpenWeatherMap API (remplace Windfinder)
- **Site original (non scrapé):** https://www.windfinder.com/#3/49.4967/9.4922/spot
- **API utilisée:** https://openweathermap.org/api
- **Documentation API:** https://openweathermap.org/current
- **Localisation:** 49.4967°N, 9.4922°E (Sindolsheim, Allemagne)
- **Données:** Température, vent, humidité, pression en temps réel
- **Prévisions:** 5 jours (toutes les 3 heures)
- **Fichiers générés:** `data/meteo_reel.csv`, `data/previsions_meteo.csv`
- **Pourquoi API?** Windfinder utilise JavaScript → BeautifulSoup insuffisant

### 3. Données Énergétiques (4 datasets)
- `Adelaide_Data.csv` (source locale)
- `Perth_Data.csv` (source locale)
- `Sydney_Data.csv` (source locale)
- `Tasmania_Data.csv` (source locale)

---

## Technologies Utilisées
- **Python 3.x** - https://www.python.org/
- **BeautifulSoup4** - https://pypi.org/project/beautifulsoup4/
- **Requests** - https://pypi.org/project/requests/
- **Pandas** - https://pandas.pydata.org/
- **NumPy** - https://numpy.org/
- **Scikit-learn** - https://scikit-learn.org/
- **Matplotlib** - https://matplotlib.org/
- **Seaborn** - https://seaborn.pydata.org/
- **MLflow** - https://mlflow.org/
- **Docker** - https://www.docker.com/
- **Git/GitHub** - https://git-scm.com/

---

## Structure du Projet
projet-web-scraping-energy/
├── README.md
├── requirements.txt
├── .gitignore
│
├── code/
│ ├── scraping.py # Script principal scraping UCI
│ ├── scraping_meteo_api.py # API météo OpenWeatherMap
│ ├── create_samples.py # Échantillons pour GitHub
│ ├── eda_complet.py # Analyse EDA complète
│ ├── preprocessing_final.py # Preprocessing (StandardScaler)
│ ├── training_mlflow.py # Entraînement avec MLflow
│ ├── training_multiple_models.py # Comparaison de 8 modèles
│ ├── verification_api.py # Vérification API météo
│ ├── voir_resultats_eda.py # Visualisation résultats EDA
│ └── montrer_points_importants.py # Points clés pour le prof
│
├── data/
│ ├── Adelaide_Data.csv # Dataset énergétique (local)
│ ├── Perth_Data.csv # Dataset énergétique (local)
│ ├── Sydney_Data.csv # Dataset énergétique (local)
│ ├── Tasmania_Data.csv # Dataset énergétique (local)
│ ├── meteo_reel.csv # Données météo réelles (API)
│ ├── previsions_meteo.csv # Prévisions 5 jours
│ ├── uci_dataset_detaille.csv # Métadonnées UCI scrapées
│ ├── analyse_datasets.csv # Analyse des datasets
│ ├── metadata.json # Métadonnées projet
│ │
│ ├── processed/ # Données transformées
│ │ ├── analyse_types_variables.csv
│ │ ├── recommandations_encodage.csv
│ │ ├── fortes_correlations.csv
│ │ ├── detection_outliers.csv
│ │ ├── best_model_reel.pkl # Meilleur modèle (GradientBoosting)
│ │ ├── scaler_reel.pkl # Scaler sauvegardé
│ │ └── README.md
│ │
│ ├── visualisations/ # Graphiques PNG
│ │ ├── standard_scaler_comparison.png
│ │ ├── matrice_correlation.png
│ │ └── boxplots_outliers.png
│ │
│ └── reports/ # Rapports
│ ├── rapport_eda_complet.json
│ ├── rapport_eda_complet.txt
│ └── comparaison_modeles_reels.csv # Résultats des 8 modèles
│
├── docker_model/ # Déploiement Docker
│ ├── Dockerfile
│ ├── predict_api.py # API Flask
│ ├── requirements.txt
│ ├── requirements_api.txt
│ └── README.md
│
└── mlruns/ # Logs MLflow (local)
