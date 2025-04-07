#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
indeed_scraper.py - Module de scraping d'offres d'emploi sur Indeed
Ce module permet de récupérer des offres d'emploi d'Indeed qui correspondent 
aux compétences et au domaine d'un CV analysé.

Fonctionnalités:
- Requêtes de recherche basées sur les résultats d'analyse de CV
- Scraping des résultats avec rotation d'User-Agents
- Extraction d'informations clés (titre, entreprise, lieu, description)
- Calcul de score de correspondance avec le CV
"""

import os
import sys
import re
import json
import time
import random
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

import requests
from bs4 import BeautifulSoup
import pandas as pd

# Importer les utilitaires de scraping communs
from scrapers.scraper_utils import (
    setup_logger, get_random_user_agent, get_request_headers,
    safe_request, extract_text_from_html, calculate_job_id,
    is_duplicate_job, normalize_job_location, save_job_offers,
    extract_salary_from_text, DOMAINS, CITIES_BY_COUNTRY
)

# Importer le gestionnaire de proxies (optionnel)
try:
    from infrastructure.proxies import get_random_proxy
    HAS_PROXIES = True
except ImportError:
    HAS_PROXIES = False

# Configuration
DEFAULT_LOCATION = "France"
DEFAULT_OUTPUT_DIR = "data/indeed_jobs"
DEFAULT_COUNTRY = "fr"  # fr pour France, ma pour Maroc

def parse_arguments():
    """Gère les arguments en ligne de commande pour le script."""
    parser = argparse.ArgumentParser(description="Scraper d'offres d'emploi Indeed")
    
    parser.add_argument("--query", required=True, help="Requête de recherche Indeed")
    parser.add_argument("--location", default=DEFAULT_LOCATION, help=f"Localisation (défaut: {DEFAULT_LOCATION})")
    parser.add_argument("--max-results", type=int, default=10, help="Nombre maximum de résultats (défaut: 10)")
    parser.add_argument("--output", help="Chemin du fichier de sortie JSON")
    parser.add_argument("--cv-results", help="Fichier JSON contenant les résultats d'analyse du CV")
    parser.add_argument("--verbose", action="store_true", help="Afficher des informations détaillées")
    parser.add_argument("--country", default=DEFAULT_COUNTRY, choices=["fr", "ma"], 
                      help=f"Code pays - fr: France, ma: Maroc (défaut: {DEFAULT_COUNTRY})")
    
    return parser.parse_args()

def format_query_for_url(query):
    """Formate une requête pour l'URL Indeed."""
    return query.replace(' ', '+')

def generate_indeed_query(cv_results):
    """Génère une requête de recherche Indeed basée sur l'analyse d'un CV."""
    cv_analysis = cv_results.get("cv_analysis", {})
    
    # Obtenir les informations principales
    domain = cv_analysis.get("main_domain", "")
    skills = cv_analysis.get("skills", [])
    
    # Mapper les domaines aux intitulés de poste appropriés
    domain_to_job_title = {
        "informatique_reseaux": "développeur OR devops OR ingénieur",
        "automatismes_info_industrielle": "automaticien OR robotique OR programmation",
        "finance": "finance OR comptable OR audit",
        "genie_civil_btp": "génie civil OR btp OR construction",
        "genie_industriel": "ingénieur OR production OR logistique"
    }
    
    # Construire la requête
    job_title_part = domain_to_job_title.get(domain, "")
    
    # Ajouter des compétences clés (max 3 pour éviter une requête trop longue)
    skill_part = ""
    if skills:
        # Prioriser les compétences techniques
        top_skills = skills[:3]
        skill_part = " ".join([f'"{skill}"' for skill in top_skills])
    
    # Construire la requête finale
    query_parts = []
    
    if job_title_part:
        query_parts.append(f"({job_title_part})")
    
    if skill_part:
        query_parts.append(skill_part)
    
    # Requête par défaut si rien n'a été trouvé
    if not query_parts:
        return "développeur OR ingénieur"
    
    return " ".join(query_parts)

def load_local_indeed_jobs(domain=None, max_results=5, country_code=None, logger=None):
    """
    Charge les offres d'emploi Indeed sauvegardées localement.
    Utilisé comme source alternative en cas d'échec du scraping Indeed.
    
    Args:
        domain: Domaine professionnel pour filtrer les offres
        max_results: Nombre max de résultats à retourner
        country_code: Code du pays (fr ou ma) pour filtrer les offres
        logger: Logger pour les messages
        
    Returns:
        Liste d'offres d'emploi au format de scrape_indeed_jobs
    """
    jobs = []
    
    try:
        # Répertoire de stockage des offres
        jobs_dir = os.path.join("data", "indeed_jobs")
        
        if not os.path.exists(jobs_dir):
            if logger:
                logger.warning(f"Répertoire des offres locales introuvable: {jobs_dir}")
            else:
            print(f"Répertoire des offres locales introuvable: {jobs_dir}")
            return []
        
        # Parcourir les fichiers JSON dans le répertoire
        json_files = [f for f in os.listdir(jobs_dir) if f.endswith('.json')]
        
        # Trier par date de modification (plus récent d'abord)
        json_files.sort(key=lambda x: os.path.getmtime(os.path.join(jobs_dir, x)), reverse=True)
        
        # Priorité aux fichiers de lot (batch)
        batch_files = [f for f in json_files if f.startswith('batch_')]
        individual_files = [f for f in json_files if not f.startswith('batch_') and not f.startswith('emergency_backup_')]
        emergency_files = [f for f in json_files if f.startswith('emergency_backup_')]
        
        sorted_files = batch_files + individual_files + emergency_files
        
        # Ensemble pour suivre les offres déjà ajoutées (éviter les doublons)
        added_jobs = set()
        
        # Charger des offres jusqu'à atteindre max_results
        for filename in sorted_files:
            filepath = os.path.join(jobs_dir, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Traiter différemment selon le type de fichier
                if filename.startswith('batch_') or filename.startswith('emergency_backup_'):
                    # Fichier de lot
                    batch_jobs = data.get('jobs', [])
                    for job in batch_jobs:
                        # Vérifier si l'offre est déjà dans notre liste
                        job_id = job.get('id')
                        if not job_id:
                            job_id = calculate_job_id(job)
                            job['id'] = job_id
                            
                        if job_id in added_jobs:
                            continue
                        
                        # Filtrer par domaine si spécifié
                        if domain and job.get('domain') != domain:
                            continue
                            
                        # Filtrer par pays si spécifié
                        job_country = job.get('country', '')
                        if country_code:
                            if country_code == 'fr' and job_country != 'France':
                                continue
                            elif country_code == 'ma' and job_country != 'Maroc':
                                continue
                        
                        # Ajouter l'offre à notre liste
                            added_jobs.add(job_id)
                        jobs.append(job)
                        
                        # Arrêter si on a assez d'offres
                        if len(jobs) >= max_results:
                            break
                else:
                    # Fichier individuel
                    job = data
                    
                    # Vérifier si l'offre est déjà dans notre liste
                    job_id = job.get('id')
                    if not job_id:
                        job_id = calculate_job_id(job)
                        job['id'] = job_id
                        
                    if job_id in added_jobs:
                        continue
                    
                    # Filtrer par domaine si spécifié
                    if domain and job.get('domain') != domain:
                        continue
                        
                    # Filtrer par pays si spécifié
                    job_country = job.get('country', '')
                    if country_code:
                        if country_code == 'fr' and job_country != 'France':
                            continue
                        elif country_code == 'ma' and job_country != 'Maroc':
                            continue
                    
                    # Ajouter l'offre à notre liste
                        added_jobs.add(job_id)
                    jobs.append(job)
            
            except Exception as e:
                if logger:
                    logger.error(f"Erreur lors de la lecture de {filepath}: {str(e)}")
                else:
                    print(f"Erreur lors de la lecture de {filepath}: {str(e)}")
                continue
            
            # Arrêter si on a assez d'offres
            if len(jobs) >= max_results:
                break
        
        if logger:
            logger.info(f"Chargé {len(jobs)} offres locales")
        else:
            print(f"Chargé {len(jobs)} offres locales")
        
        return jobs
    
    except Exception as e:
        if logger:
            logger.error(f"Erreur lors du chargement des offres locales: {str(e)}")
        else:
        print(f"Erreur lors du chargement des offres locales: {str(e)}")
        return []

def scrape_indeed_jobs(query, location, max_results=10, country_code="fr", logger=None):
    """
    Scrape les résultats de recherche Indeed pour une requête donnée.
    
    Args:
        query: Requête de recherche
        location: Localisation (ville, région, pays)
        max_results: Nombre maximum de résultats à retourner
        country_code: Code pays (fr, ma)
        logger: Logger pour les messages
        
    Returns:
        Liste d'offres d'emploi
    """
    # Utiliser le logger fourni ou créer un logger par défaut
    if logger is None:
        logger = setup_logger("indeed_scraper")
    
    jobs = []
    
    # Formater la requête pour l'URL
    query_formatted = format_query_for_url(query)
    location_formatted = format_query_for_url(location)
    
    # Déterminer l'URL de base selon le pays
    if country_code == "ma":
        base_url = "https://ma.indeed.com"
    else:
        base_url = "https://fr.indeed.com"
    
    # Construire l'URL de recherche
    search_url = f"{base_url}/jobs?q={query_formatted}&l={location_formatted}"
    
    logger.info(f"Scraping d'Indeed pour la recherche: {query}")
    logger.info(f"URL: {search_url}")
    
    try:
        # Obtenir les headers et proxy
        headers = get_request_headers(referer=base_url)
        proxies = get_random_proxy() if HAS_PROXIES else None
        
        # Faire la requête HTTP avec gestion des erreurs
        response = safe_request(
            search_url, 
            headers=headers, 
            proxies=proxies,
            logger=logger
        )
        
        if not response:
            logger.error("Échec de la requête vers Indeed")
            return []
        
        # Parser le HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Trouver les cartes d'offres d'emploi
        job_cards = soup.select("div.job_seen_beacon") or soup.select("div.jobsearch-SerpJobCard")
        
        if not job_cards:
            logger.warning("Aucune offre trouvée. La structure HTML a peut-être changé.")
            return []
        
        logger.info(f"Trouvé {len(job_cards)} offres")
        
        # Limiter au nombre max de résultats
        job_cards = job_cards[:max_results]
        
        # Extraire les informations de chaque offre
        for card in job_cards:
            try:
                # Extraire les éléments clés
                title_elem = card.select_one("h2.jobTitle") or card.select_one("a.jobtitle")
                company_elem = card.select_one("span.companyName") or card.select_one("span.company")
                location_elem = card.select_one("div.companyLocation") or card.select_one("div.location")
                description_elem = card.select_one("div.job-snippet") or card.select_one("div.summary")
                
                # Extraire le texte
                title = title_elem.get_text(strip=True) if title_elem else "N/A"
                company = company_elem.get_text(strip=True) if company_elem else "N/A"
                location_raw = location_elem.get_text(strip=True) if location_elem else location
                description = description_elem.get_text(strip=True) if description_elem else "N/A"
                
                # Normaliser l'emplacement
                location_normalized = normalize_job_location(location_raw, country_code)
                
                # Essayer d'extraire le salaire de la description
                salary = extract_salary_from_text(description)
                
                # Essayer d'obtenir l'URL de l'offre
                job_url = ""
                url_elem = card.select_one("a.jcs-JobTitle") or card.select_one("a.jobtitle")
                if url_elem and 'href' in url_elem.attrs:
                    job_url = base_url + url_elem['href']
                
                # Créer l'objet d'offre d'emploi
                job = {
                    "title": title,
                    "company": company,
                    "location": location_normalized,
                    "description": description,
                    "salary": salary,
                    "url": job_url,
                    "source": "Indeed",
                    "country": "France" if country_code == "fr" else "Maroc",
                    "date_collected": datetime.now().isoformat()
                }
                
                # Générer un ID unique
                job["id"] = calculate_job_id(job)
                
                # Vérifier s'il s'agit d'un doublon
                if not is_duplicate_job(job, jobs):
                    jobs.append(job)
                    logger.debug(f"Ajouté: {title} - {company}")
            
    except Exception as e:
                logger.error(f"Erreur lors de l'extraction d'une offre: {str(e)}")
                continue
        
        return jobs
        
    except Exception as e:
        logger.error(f"Erreur lors du scraping Indeed: {str(e)}")
        return []

def match_job_with_cv(job, cv_results):
    """
    Calcule un score de correspondance entre une offre d'emploi et un CV.
    
    Args:
        job: Offre d'emploi
        cv_results: Résultats d'analyse du CV
        
    Returns:
        Score de correspondance entre 0 et 1
    """
    cv_analysis = cv_results.get("cv_analysis", {})
    
    # Extraire les informations du CV
    cv_domain = cv_analysis.get("main_domain", "")
    cv_skills = cv_analysis.get("skills", [])
    
    # Le score commence à 0.1 pour éviter les scores nuls
    score = 0.1
    
    # Vérifier la correspondance du domaine
    if cv_domain and job.get("domain") == cv_domain:
        score += 0.3
    
    # Vérifier la correspondance des compétences
    job_description = job.get("description", "").lower()
    job_title = job.get("title", "").lower()
    
    skills_found = 0
    for skill in cv_skills:
        skill_lower = skill.lower()
        if skill_lower in job_description or skill_lower in job_title:
            skills_found += 1
    
    if cv_skills:
        skill_score = min(0.6, (skills_found / len(cv_skills)) * 0.6)
        score += skill_score
    
    return min(1.0, score)

def main():
    """Fonction principale pour exécuter le scraper Indeed."""
    # Parser les arguments
    args = parse_arguments()
    
    # Configurer le logger
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger("indeed_scraper", log_level=log_level)
    
    # Charger les résultats du CV si spécifiés
    cv_results = None
    if args.cv_results:
        try:
            with open(args.cv_results, 'r', encoding='utf-8') as f:
                cv_results = json.load(f)
        except Exception as e:
            logger.error(f"Erreur lors de la lecture des résultats du CV: {str(e)}")
            return 1
    
    # Déterminer la requête
    query = args.query
    if cv_results:
        query = generate_indeed_query(cv_results)
        logger.info(f"Requête générée à partir du CV: {query}")
    
    # Scraper les offres d'emploi
    jobs = scrape_indeed_jobs(
        query=query,
        location=args.location,
        max_results=args.max_results,
        country_code=args.country,
        logger=logger
    )
    
    # Si aucune offre n'a été trouvée, essayer de charger les offres locales
    if not jobs:
        logger.warning("Aucune offre trouvée en ligne, tentative de chargement local")
        domain = cv_results.get("cv_analysis", {}).get("main_domain") if cv_results else None
        jobs = load_local_indeed_jobs(
            domain=domain, 
            max_results=args.max_results,
            country_code=args.country,
            logger=logger
        )
    
    # Calculer les scores de correspondance si les résultats du CV sont disponibles
    if cv_results:
        for job in jobs:
            job["match_score"] = match_job_with_cv(job, cv_results)
        
        # Trier par score de correspondance décroissant
        jobs.sort(key=lambda x: x.get("match_score", 0), reverse=True)
    
    # Déterminer le chemin de sortie
    if args.output:
    output_path = args.output
    else:
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(DEFAULT_OUTPUT_DIR, f"indeed_jobs_{timestamp}.json")
    
    # Sauvegarder les offres
    if jobs:
        save_job_offers(jobs, output_path, logger)
        logger.info(f"Sauvegardé {len(jobs)} offres dans {output_path}")
    else:
        logger.warning("Aucune offre à sauvegarder")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 