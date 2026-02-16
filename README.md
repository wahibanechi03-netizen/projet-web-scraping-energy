# Projet Web Scraping - Énergie des Vagues

##  Équipe
- **Sourour Ben Salha** (wahibanechi03@gmail.com)
- **Mariem Abida**
- **Mariem Werhani**
- **Hedyl Ben Taher**

**Groupe:** ING-4-J-SDIAF-A

##  LIENS DU PROJET
- **GitHub Repository:** https://github.com/wahibanechi03-netizen/projet-web-scraping-energy
- **Dataset UCI:** https://archive.ics.uci.edu/dataset/494/wave+energy+converters
- **OpenWeatherMap API:** https://openweathermap.org/api
- **Documentation BeautifulSoup:** https://www.crummy.com/software/BeautifulSoup/bs4/doc/
- **Pandas Documentation:** https://pandas.pydata.org/docs/
- **Scikit-learn:** https://scikit-learn.org/stable/

## Objectif du Projet
Collecte et analyse de données sur l'énergie des vagues et éolienne pour des applications de Machine Learning, en utilisant des techniques de web scraping et d'API.

##  Sources de Données

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
- Adelaide_Data.csv (source locale)
- Perth_Data.csv (source locale)
- Sydney_Data.csv (source locale)
- Tasmania_Data.csv (source locale)

##  Technologies Utilisées
- **Python 3.x** - https://www.python.org/
- **BeautifulSoup4** - https://pypi.org/project/beautifulsoup4/
- **Requests** - https://pypi.org/project/requests/
- **Pandas** - https://pandas.pydata.org/
- **NumPy** - https://numpy.org/
- **Scikit-learn** - https://scikit-learn.org/
- **Matplotlib** - https://matplotlib.org/
- **Seaborn** - https://seaborn.pydata.org/
- **Git/GitHub** - https://git-scm.com/

## 📁 Structure du Projet
