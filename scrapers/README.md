# Architecture de Scraping pour AI-Recruiter

Ce répertoire contient l'infrastructure de scraping d'offres d'emploi pour le système AI-Recruiter. L'architecture a été conçue pour être modulaire, robuste et facile à maintenir.

## Structure du projet

```
scrapers/
├── scraper_utils.py       # Utilitaires communs pour tous les scrapers
├── indeed_scraper.py      # Scraper spécifique pour Indeed
├── job_scraper_morocco.py # Scraper pour les offres au Maroc
├── sample_job_scraper.py  # Exemple de scraper utilisant les modules
└── README.md              # Documentation

infrastructure/
├── proxies.py             # Gestion des proxies pour éviter le blocage
├── scrape_scheduler.py    # Orchestration et planification des tâches
└── airflow_job_scraper.py # DAG Airflow pour l'exécution planifiée

tests/
├── test_scraper_modules.py # Tests des modules de scraping
```

## Modules principaux

### 1. Utilitaires de Scraping (`scraper_utils.py`)

Module central contenant des fonctions et utilitaires communs utilisés par tous les scrapers:

- Gestion des headers HTTP et User-Agents
- Requêtes HTTP robustes avec retry automatique
- Extraction de texte et de données depuis le HTML
- Calcul d'ID unique pour les offres d'emploi
- Détection de doublons
- Normalisation des données (ex: emplacement, salaire)
- Sauvegarde des offres au format JSON

### 2. Gestion des Proxies (`proxies.py`)

Module de gestion intelligente des proxies pour éviter le blocage lors des opérations de scraping:

- Récupération de proxies depuis différentes sources
- Test automatique des proxies pour vérifier leur fonctionnement
- Cache des proxies fonctionnels
- Rotation automatique des proxies
- Gestion asynchrone pour optimiser les performances

### 3. Orchestration des Tâches (`scrape_scheduler.py`)

Module d'orchestration avec Airflow pour planifier et exécuter automatiquement les tâches de scraping:

- Configuration des DAGs Airflow pour chaque type de scraping
- Gestion de l'espace disque et nettoyage automatique
- Archivage des données historiques
- Logs détaillés et génération de rapports

## Utilisation

### Installation des dépendances

```bash
pip install -r requirements.txt
```

### Exécution d'un scraper simple

```bash
python sample_job_scraper.py --domain informatique_reseaux --country fr --max-results 10
```

### Tests des modules

```bash
python test_scraper_modules.py
```

### Options disponibles

- `--domain`: Domaine professionnel (informatique_reseaux, finance, genie_civil_btp, etc.)
- `--location`: Localisation spécifique pour la recherche
- `--country`: Code pays (fr pour France, ma pour Maroc)
- `--max-results`: Nombre maximum de résultats à récupérer
- `--verbose`: Mode verbeux pour plus de détails dans les logs

## Intégration avec Airflow

Les tâches de scraping sont orchestrées via Apache Airflow. Pour exécuter les DAGs:

1. Assurez-vous qu'Airflow est installé et configuré
2. Placez les fichiers DAG dans le dossier `airflow/dags/` 
3. Redémarrez le scheduler Airflow pour détecter les nouveaux DAGs

## Sécurité et bonnes pratiques

- Utilisez des délais aléatoires entre les requêtes pour simuler un comportement humain
- Respectez les conditions d'utilisation des sites web que vous scrappez
- Pensez à consulter le fichier robots.txt des sites cibles
- Utilisez des proxies différents pour les requêtes à haute fréquence
- Évitez de surcharger les serveurs des sites cibles

## Extension

Pour ajouter un nouveau scraper:

1. Créez un nouveau fichier dans le dossier `scrapers/`
2. Importez les fonctions utilitaires depuis `scraper_utils.py`
3. Utilisez la classe `ProxyManager` de `proxies.py` si nécessaire
4. Ajoutez la configuration correspondante dans `scrape_scheduler.py`

## Contribution

Pour contribuer à l'amélioration des scrapers:

1. Suivez les conventions de codage PEP 8
2. Documentez vos fonctions avec des docstrings
3. Ajoutez des tests pour vos nouvelles fonctionnalités
4. Soumettez une pull request avec une description détaillée

## Licence

Ce code est destiné à un usage interne uniquement. 