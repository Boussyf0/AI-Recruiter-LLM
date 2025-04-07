#!/usr/bin/env python
# Script d'orchestration Airflow pour scraper les offres d'emploi au Maroc

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import sys
import subprocess
import logging

# Configuration du DAG
default_args = {
    'owner': 'ai_recruiter',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2023, 1, 1),
}

# Chemin vers les scripts de scraping
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRAPER_DIR = os.path.join(PROJECT_ROOT, 'scrapers')
MOROCCO_SCRIPT = os.path.join(SCRAPER_DIR, 'job_scraper_morocco.py')
FRANCE_SCRIPT = os.path.join(SCRAPER_DIR, 'job_scraper.py')

# Fonction pour exécuter un script de scraping
def run_job_scraper(script_path, country_code, **kwargs):
    try:
        logging.info(f"Exécution du script de scraping pour {country_code}: {script_path}")
        
        # Vérifier si le script existe
        if not os.path.exists(script_path):
            logging.error(f"Script non trouvé: {script_path}")
            raise FileNotFoundError(f"Script non trouvé: {script_path}")
        
        # Préparer les arguments selon le script
        cmd = [sys.executable, script_path]
        
        if country_code == 'ma':
            # Arguments spécifiques pour le Maroc
            cmd.extend(['--max-results', '20'])
        else:
            # Arguments spécifiques pour la France
            cmd.extend(['--country', 'fr', '--max-results', '20'])
        
        # Exécuter le script
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        logging.info(f"Scraping pour {country_code} terminé avec succès")
        logging.info(f"Sortie: {result.stdout}")
        
        return result.stdout
    
    except subprocess.CalledProcessError as e:
        logging.error(f"Erreur lors de l'exécution du script pour {country_code}: {e}")
        logging.error(f"Sortie d'erreur: {e.stderr}")
        raise
    except Exception as e:
        logging.error(f"Erreur inattendue pour {country_code}: {str(e)}")
        raise

# Fonction pour le scraping au Maroc
def run_morocco_scraper(**kwargs):
    return run_job_scraper(MOROCCO_SCRIPT, 'ma', **kwargs)

# Fonction pour le scraping en France
def run_france_scraper(**kwargs):
    return run_job_scraper(FRANCE_SCRIPT, 'fr', **kwargs)

# Définition du DAG pour le Maroc
with DAG(
    'morocco_job_scraper',
    default_args=default_args,
    description='Scrape des offres d\'emploi au Maroc toutes les 12 heures',
    schedule_interval='0 */12 * * *',  # Exécution toutes les 12 heures
    catchup=False,
    tags=['scraping', 'jobs', 'morocco'],
) as morocco_dag:

    # Tâche pour vérifier l'espace disque
    check_disk_space_ma = BashOperator(
        task_id='check_disk_space',
        bash_command='df -h | grep -v "tmpfs\|cdrom" | tail -n +2',
    )
    
    # Tâche pour exécuter le script de scraping Maroc
    scrape_morocco_jobs = PythonOperator(
        task_id='scrape_morocco_jobs',
        python_callable=run_morocco_scraper,
        provide_context=True,
    )
    
    # Tâche pour nettoyer les fichiers JSON trop anciens (1 semaine)
    cleanup_old_files_ma = BashOperator(
        task_id='cleanup_old_files',
        bash_command='find data/scraped_jobs -name "*_ma_*.json" -type f -mtime +7 -delete || true',
    )
    
    # Définition de l'ordre d'exécution des tâches
    check_disk_space_ma >> scrape_morocco_jobs >> cleanup_old_files_ma

# Définition du DAG pour la France
with DAG(
    'france_job_scraper',
    default_args=default_args,
    description='Scrape des offres d\'emploi en France tous les jours',
    schedule_interval='0 0 * * *',  # Exécution quotidienne à minuit
    catchup=False,
    tags=['scraping', 'jobs', 'france'],
) as france_dag:

    # Tâche pour vérifier l'espace disque
    check_disk_space_fr = BashOperator(
        task_id='check_disk_space',
        bash_command='df -h | grep -v "tmpfs\|cdrom" | tail -n +2',
    )
    
    # Tâche pour exécuter le script de scraping France
    scrape_france_jobs = PythonOperator(
        task_id='scrape_france_jobs',
        python_callable=run_france_scraper,
        provide_context=True,
    )
    
    # Tâche pour nettoyer les fichiers JSON trop anciens (1 semaine)
    cleanup_old_files_fr = BashOperator(
        task_id='cleanup_old_files',
        bash_command='find data/scraped_jobs -name "*_fr_*.json" -type f -mtime +7 -delete || true',
    )
    
    # Définition de l'ordre d'exécution des tâches
    check_disk_space_fr >> scrape_france_jobs >> cleanup_old_files_fr 