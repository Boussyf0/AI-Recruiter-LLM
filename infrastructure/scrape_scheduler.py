#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
scrape_scheduler.py - Orchestration avancée des tâches de scraping avec Airflow
Module qui définit les DAGs Airflow pour l'exécution planifiée des scrapers
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable
from airflow.hooks.base import BaseHook

import os
import sys
import json
import glob
import logging
import subprocess
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# Chemin vers le répertoire du projet
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRAPERS_DIR = os.path.join(PROJECT_DIR, "scrapers")
LOG_DIR = os.path.join(PROJECT_DIR, "logs")
DATA_DIR = os.path.join(PROJECT_DIR, "data")

# S'assurer que les répertoires existent
for dir_path in [LOG_DIR, DATA_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Configuration des arguments par défaut pour les DAGs
default_args = {
    'owner': 'ai_recruiter',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'start_date': days_ago(1),
}

# Configuration des scrapers
SCRAPER_CONFIG = {
    'france': {
        'script': os.path.join(SCRAPERS_DIR, 'job_scraper.py'),
        'schedule': '0 */6 * * *',  # Toutes les 6 heures
        'max_offers': 50,
        'country': 'fr',
        'domains': ['informatique_reseaux', 'automatismes_info_industrielle', 'finance', 'genie_civil_btp', 'genie_industriel'],
        'api_url': 'http://localhost:8000/api/v1'
    },
    'morocco': {
        'script': os.path.join(SCRAPERS_DIR, 'job_scraper_morocco.py'),
        'schedule': '0 */8 * * *',  # Toutes les 8 heures
        'max_offers': 50,
        'country': 'ma',
        'domains': ['informatique_reseaux', 'automatismes_info_industrielle', 'finance', 'genie_civil_btp', 'genie_industriel'],
        'api_url': 'http://localhost:8000/api/v1'
    }
}

def check_disk_space(min_space_gb=1.0, path=DATA_DIR):
    """
    Vérifie l'espace disque disponible
    
    Args:
        min_space_gb: Espace minimum requis en GB
        path: Chemin du répertoire à vérifier
    
    Returns:
        True si l'espace est suffisant, False sinon
    """
    import shutil
    
    # Obtenir l'espace disque disponible en bytes
    total, used, free = shutil.disk_usage(path)
    
    # Convertir en GB
    free_gb = free / (1024 ** 3)
    
    logging.info(f"Espace disque disponible: {free_gb:.2f} GB")
    
    if free_gb < min_space_gb:
        logging.warning(f"Espace disque insuffisant: {free_gb:.2f} GB (minimum requis: {min_space_gb} GB)")
        return False
    
    return True

def clean_old_files(directory=os.path.join(DATA_DIR, 'scraped_jobs'), days=7, pattern="*.json"):
    """
    Nettoie les fichiers plus anciens qu'un certain nombre de jours
    
    Args:
        directory: Répertoire à nettoyer
        days: Âge maximum des fichiers en jours
        pattern: Pattern de fichiers à supprimer
    
    Returns:
        Nombre de fichiers supprimés
    """
    if not os.path.exists(directory):
        logging.warning(f"Répertoire {directory} introuvable")
        return 0
    
    # Calculer la date limite
    cutoff_date = datetime.now() - timedelta(days=days)
    count = 0
    
    # Parcourir les fichiers
    for filepath in glob.glob(os.path.join(directory, pattern)):
        if os.path.isfile(filepath):
            file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            if file_time < cutoff_date:
                try:
                    os.remove(filepath)
                    count += 1
                    logging.info(f"Suppression de {filepath}")
                except Exception as e:
                    logging.error(f"Erreur lors de la suppression de {filepath}: {str(e)}")
    
    logging.info(f"Nettoyage terminé: {count} fichiers supprimés")
    return count

def archive_old_files(directory=os.path.join(DATA_DIR, 'scraped_jobs'), archive_dir=None, days=30, pattern="*.json"):
    """
    Archive les fichiers plus anciens qu'un certain nombre de jours
    
    Args:
        directory: Répertoire contenant les fichiers
        archive_dir: Répertoire d'archivage (par défaut: directory/archive)
        days: Âge maximum des fichiers en jours
        pattern: Pattern de fichiers à archiver
    
    Returns:
        Nombre de fichiers archivés
    """
    if not os.path.exists(directory):
        logging.warning(f"Répertoire {directory} introuvable")
        return 0
    
    # Déterminer le répertoire d'archivage
    if archive_dir is None:
        archive_dir = os.path.join(directory, 'archive')
    
    # Créer le répertoire d'archivage si nécessaire
    os.makedirs(archive_dir, exist_ok=True)
    
    # Calculer la date limite
    cutoff_date = datetime.now() - timedelta(days=days)
    count = 0
    
    # Parcourir les fichiers
    for filepath in glob.glob(os.path.join(directory, pattern)):
        if os.path.isfile(filepath):
            file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            if file_time < cutoff_date:
                try:
                    filename = os.path.basename(filepath)
                    archive_path = os.path.join(archive_dir, filename)
                    
                    # Déplacer le fichier
                    shutil.move(filepath, archive_path)
                    count += 1
                    logging.info(f"Archivage de {filepath} vers {archive_path}")
                except Exception as e:
                    logging.error(f"Erreur lors de l'archivage de {filepath}: {str(e)}")
    
    logging.info(f"Archivage terminé: {count} fichiers archivés")
    return count

def run_scraper(config_name, **kwargs):
    """
    Exécute un scraper en fonction de sa configuration
    
    Args:
        config_name: Nom de la configuration du scraper
        **kwargs: Arguments Airflow
    
    Returns:
        Sortie du script de scraping
    """
    if config_name not in SCRAPER_CONFIG:
        raise ValueError(f"Configuration inconnue: {config_name}")
    
    config = SCRAPER_CONFIG[config_name]
    script_path = config['script']
    
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script de scraping introuvable: {script_path}")
    
    # Construire la commande
    cmd = [sys.executable, script_path]
    
    # Ajouter les paramètres spécifiques
    if 'max_offers' in config:
        cmd.extend(['--max-offers', str(config['max_offers'])])
    
    if 'country' in config:
        cmd.extend(['--country', config['country']])
    
    if 'api_url' in config:
        cmd.extend(['--api-url', config['api_url']])
    
    if 'domains' in config:
        domains_str = ','.join(config['domains'])
        cmd.extend(['--domains', domains_str])
    
    # Exécuter la commande
    logging.info(f"Exécution du scraper {config_name}: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        logging.info(f"Scraping {config_name} terminé avec succès")
        logging.debug(f"Sortie: {result.stdout}")
        
        # Enregistrer la sortie dans un fichier de log
        log_path = os.path.join(LOG_DIR, f"scraper_{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"COMMANDE: {' '.join(cmd)}\n")
            f.write(f"CODE: {result.returncode}\n")
            f.write(f"SORTIE:\n{result.stdout}\n")
            if result.stderr:
                f.write(f"ERREURS:\n{result.stderr}\n")
        
        return result.stdout
    
    except subprocess.CalledProcessError as e:
        logging.error(f"Erreur lors de l'exécution du scraper {config_name}: code {e.returncode}")
        logging.error(f"Sortie d'erreur: {e.stderr}")
        
        # Enregistrer l'erreur dans un fichier de log
        log_path = os.path.join(LOG_DIR, f"scraper_{config_name}_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"COMMANDE: {' '.join(cmd)}\n")
            f.write(f"CODE: {e.returncode}\n")
            if e.stdout:
                f.write(f"SORTIE:\n{e.stdout}\n")
            f.write(f"ERREURS:\n{e.stderr}\n")
        
        raise

def send_stats_report(**kwargs):
    """
    Génère et envoie un rapport de statistiques sur le scraping
    
    Args:
        **kwargs: Arguments Airflow
    """
    # TODO: Implémenter l'envoi d'un rapport par email ou API
    pass

# Création des DAGs pour chaque scraper
for config_name, config in SCRAPER_CONFIG.items():
    # Créer un DAG spécifique pour ce scraper
    dag = DAG(
        f'{config_name}_job_scraper',
        default_args=default_args,
        description=f'Scrape des offres d\'emploi pour {config_name}',
        schedule_interval=config['schedule'],
        catchup=False,
        tags=['scraping', 'jobs', config_name],
    )
    
    # Tâche pour vérifier l'espace disque
    disk_check = PythonOperator(
        task_id='check_disk_space',
        python_callable=check_disk_space,
        op_kwargs={'min_space_gb': 0.5},  # 500 MB minimum
        dag=dag,
    )
    
    # Tâche pour nettoyer les anciens fichiers
    cleanup = PythonOperator(
        task_id='cleanup_old_files',
        python_callable=clean_old_files,
        op_kwargs={'days': 7},  # Fichiers > 7 jours
        dag=dag,
    )
    
    # Tâche pour archiver les fichiers plus anciens
    archive = PythonOperator(
        task_id='archive_old_files',
        python_callable=archive_old_files,
        op_kwargs={'days': 30},  # Fichiers > 30 jours
        dag=dag,
    )
    
    # Tâche pour exécuter le scraper
    scrape = PythonOperator(
        task_id=f'scrape_{config_name}_jobs',
        python_callable=run_scraper,
        op_kwargs={'config_name': config_name},
        dag=dag,
    )
    
    # Tâche pour afficher les statistiques
    stats = BashOperator(
        task_id='show_stats',
        bash_command=f'find {DATA_DIR}/scraped_jobs -name "*.json" -type f -mtime -1 | wc -l',
        dag=dag,
    )
    
    # Définir les dépendances entre les tâches
    disk_check >> cleanup >> archive >> scrape >> stats
    
    # Exposer le DAG
    globals()[f'{config_name}_dag'] = dag 