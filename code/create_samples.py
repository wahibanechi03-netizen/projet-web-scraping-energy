import pandas as pd
import os

print("Création des échantillons pour GitHub...")

# Créer le dossier sample_data
os.makedirs('data/sample_data', exist_ok=True)

# Fichiers à sampler
files = ['Adelaide_Data.csv', 'Perth_Data.csv']

for file in files:
    try:
        # Lire 100 premières lignes
        df = pd.read_csv(f'data/{file}', nrows=100)
        # Sauvegarder l'échantillon
        df.to_csv(f'data/sample_data/{file.replace(".csv", "_sample.csv")}', index=False)
        print(f"✅ Échantillon créé pour {file}")
    except:
        print(f"❌ Erreur avec {file}")

# Copier le fichier scrapé complet
import shutil
shutil.copy('data/uci_dataset_scrape.csv', 'data/sample_data/')
print("✅ Fichier UCI copié")

print("\n🎉 Échantillons prêts pour GitHub!")