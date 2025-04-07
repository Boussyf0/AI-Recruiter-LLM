# Architecture de Scraping Améliorée

Ce document détaille l'architecture modulaire de scraping mise en place pour collecter des offres d'emploi de manière robuste, efficace et maintenable.

## Structure du Projet

```
ai-recruiter-llm/
├── scrapers/
│   ├── __init__.py
│   ├── scraper_utils.py   # Utilitaires communs pour tous les scrapers
│   ├── indeed_scraper.py  # Scraper spécifique à Indeed
│   ├── job_scraper.py     # Scraper générique pour la France
│   └── job_scraper_morocco.py  # Scraper pour le Maroc
├── infrastructure/
│   ├── __init__.py
│   ├── proxies.py         # Gestion des proxies
│   ├── scrape_scheduler.py  # Orchestration avec Airflow
│   └── airflow_job_scraper.py  # Configuration DAGs Airflow
├── data/
│   └── scraped_jobs/      # Stockage des données scrapées
└── logs/                  # Journaux des opérations de scraping
```

## Modules Principaux

### 1. Utilitaires de Scraping (`scraper_utils.py`)

Ce module fournit des fonctions communes à tous les scrapers:

- **Gestion des User-Agents**: Rotation automatique des User-Agents pour éviter la détection
- **Requêtes HTTP robustes**: Automatisation des retries, gestion des erreurs et des timeouts
- **Extraction et nettoyage**: Fonctions pour extraire et normaliser les données
- **Déduplication**: Génération d'IDs uniques pour éviter les doublons
- **Normalisation**: Formatage des emplacements, salaires et autres informations

Usage:
```python
from scrapers.scraper_utils import (
    setup_logger, get_random_user_agent, safe_request,
    calculate_job_id, normalize_job_location
)
```

### 2. Gestion des Proxies (`proxies.py`)

Module dédié à la gestion des proxies:

- **Multi-sources**: Récupération automatique de proxies depuis diverses API
- **Cache local**: Stockage des proxies fonctionnels pour une utilisation future
- **Rotation intelligente**: Attribution dynamique basée sur la disponibilité et la performance
- **Gestion asynchrone**: Utilisation de l'asyncio pour plus d'efficacité

Usage:
```python
from infrastructure.proxies import get_random_proxy, ProxyManager

# Utilisation simple
proxy = get_random_proxy()

# Utilisation avancée
proxy_manager = ProxyManager()
await proxy_manager.initialize()
proxy = await proxy_manager.get_proxy()
```

### 3. Orchestration des Tâches (`scrape_scheduler.py`)

Intégration avec Apache Airflow pour l'orchestration:

- **DAGs configurables**: Workflows séparés pour différents pays
- **Gestion de l'espace disque**: Vérification et nettoyage automatique
- **Archivage intelligent**: Conservation des données selon une politique définie
- **Logging détaillé**: Enregistrement des performances et des erreurs

## Scrapers Spécifiques

### Indeed Scraper (`indeed_scraper.py`)

Collecte d'offres d'emploi depuis Indeed:
- Support multi-pays (France, Maroc)
- Extraction structurée de toutes les informations importantes
- Fallback sur données locales en cas d'échec de scraping

### Scraper Générique (`job_scraper.py`) 

Scraper configurable pour différentes sources:
- Interface par ligne de commande complète
- Intégration avec la base vectorielle
- Support pour différents formats de sortie (JSON, CSV)

### Scraper Maroc (`job_scraper_morocco.py`)

Version spécialisée pour le marché marocain:
- Scraping par domaine professionnel
- Adaptation aux spécificités de localisation
- Support multilingue (français/arabe)

## Utilisation

### Installation

```bash
# Installer les dépendances
pip install -r requirements.txt
```

### Exécution d'un scraper

```bash
# Scraper générique (France)
python scrapers/job_scraper.py --query "développeur python" --location "Paris" --max-results 20

# Scraper Maroc
python scrapers/job_scraper_morocco.py --domain informatique_reseaux --max-results 15

# Scraper Indeed
python scrapers/indeed_scraper.py --query "data scientist" --location "Lyon" --country fr
```

### Configuration d'Airflow

Pour configurer Airflow:

1. Assurez-vous qu'Apache Airflow est installé
2. Copiez les fichiers DAG dans votre dossier `$AIRFLOW_HOME/dags/`
3. Redémarrez le scheduler Airflow

```bash
airflow db init  # Si premier démarrage
airflow scheduler
```

## Bonnes Pratiques

1. **Respect des sites**: Ajoutez des délais entre les requêtes et respectez robots.txt
2. **Rotation des identités**: Utilisez différents User-Agents et proxies
3. **Gestion des erreurs**: Implémentez des fallbacks et retries appropriés
4. **Économie de ressources**: Limitez la fréquence et le volume de scraping
5. **Nettoyage des données**: Assurez-vous que les données sont normalisées avant stockage

## Dépannage

Problèmes courants et solutions:

1. **Blocage par le site**: Augmentez les délais entre requêtes et utilisez plus de proxies
2. **Erreurs de parsing**: Vérifiez si la structure du site a changé et mettez à jour les sélecteurs
3. **Proxies non fonctionnels**: Augmentez le timeout ou utilisez d'autres sources de proxies
4. **Performances lentes**: Utilisez le scraping asynchrone ou distribuez les tâches

## Contribution

Pour étendre l'architecture:

1. Respectez la structure modulaire existante
2. Utilisez les utilitaires communs de `scraper_utils.py`
3. Ajoutez des tests pour les nouvelles fonctionnalités
4. Documentez le code et mettez à jour ce README

## Licence

Ce code est distribué sous licence MIT.
