#!/usr/bin/env python
# Script pour scraper les offres d'emploi au Maroc par domaine

import requests
from bs4 import BeautifulSoup
import httpx
import asyncio
import json
import os
import time
import random
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

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

# Configuration du scraper
DEFAULT_API_URL = "http://localhost:8000/api/v1"
DEFAULT_LLM_URL = "http://localhost:8001"
INDEED_URL = "https://ma.indeed.com/jobs"
MAX_OFFERS_PER_DOMAIN = 15
OUTPUT_DIR = "data/scraped_jobs"
LOG_FILE = "logs/job_scraper.log"

def parse_arguments():
    """Gère les arguments en ligne de commande pour le script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Scraper d'offres d'emploi au Maroc")
    
    parser.add_argument("--domain", choices=list(DOMAINS.keys()), 
                      help="Domaine spécifique à cibler")
    parser.add_argument("--max-results", type=int, default=MAX_OFFERS_PER_DOMAIN, 
                      help=f"Nombre maximum de résultats par domaine (défaut: {MAX_OFFERS_PER_DOMAIN})")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, 
                      help=f"Répertoire de sortie (défaut: {OUTPUT_DIR})")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, 
                      help="URL de l'API backend pour indexation")
    parser.add_argument("--add-to-db", action="store_true", 
                      help="Ajouter les offres à la base vectorielle")
    parser.add_argument("--verbose", action="store_true", 
                      help="Afficher des informations détaillées")
    
    return parser.parse_args()

def scrape_jobs_for_domain(domain: str, keywords: List[str], max_offers: int = MAX_OFFERS_PER_DOMAIN, logger=None) -> List[Dict[str, Any]]:
    """Scrape les offres d'emploi pour un domaine spécifique au Maroc"""
    # Utiliser le logger fourni ou créer un logger par défaut
    if logger is None:
        logger = setup_logger("job_scraper_morocco")
    
    all_offers = []
    
    # Utiliser quelques mots-clés pour ce domaine (pas tous pour éviter des requêtes trop spécifiques)
    search_keywords = random.sample(keywords, min(3, len(keywords)))
    search_query = " OR ".join(search_keywords)
    
    logger.info(f"Scraping offres pour le domaine '{domain}' avec les mots-clés: {search_query}")
    
    # Construire l'URL de recherche
    query_formatted = search_query.replace(' ', '+')
    url = f"{INDEED_URL}?q={query_formatted}&l=Maroc"
    
    try:
        # Obtenir les headers et proxy
        headers = get_request_headers(referer="https://ma.indeed.com/")
        proxies = get_random_proxy() if HAS_PROXIES else None
        
        # Faire la requête HTTP avec gestion des erreurs
        response = safe_request(
            url, 
            headers=headers, 
            proxies=proxies,
            logger=logger
        )
        
        if not response:
            logger.error(f"Échec de la requête vers {url}")
            return []
        
        # Parser le HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Trouver les cartes d'offres d'emploi
        job_cards = soup.select("div.job_seen_beacon") or soup.select("div.jobsearch-SerpJobCard")
        
        if not job_cards:
            logger.warning(f"Aucune offre trouvée pour {domain} avec les mots-clés {search_query}. Vérifiez la structure HTML.")
            return []
        
        logger.info(f"Trouvé {len(job_cards)} offres pour {domain}")
        
        # Limiter au nombre maximum demandé
        job_cards = job_cards[:max_offers]
        
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
                location_raw = location_elem.get_text(strip=True) if location_elem else "Maroc"
                description = description_elem.get_text(strip=True) if description_elem else "N/A"
                
                # Normaliser l'emplacement
                location_normalized = normalize_job_location(location_raw, "ma")
                
                # Essayer d'extraire le salaire de la description
                salary = extract_salary_from_text(description)
                
                # Essayer d'obtenir l'URL de l'offre
                job_url = ""
                url_elem = card.select_one("a.jcs-JobTitle") or card.select_one("a.jobtitle")
                if url_elem and 'href' in url_elem.attrs:
                    job_url = "https://ma.indeed.com" + url_elem['href']
                
                # Créer l'objet offre d'emploi
                job_offer = {
                    "title": title,
                    "company": company,
                    "location": location_normalized,
                    "description": description,
                    "domain": domain,
                    "salary": salary,
                    "url": job_url,
                    "source": "Indeed Maroc",
                    "country": "Maroc", 
                    "date_collected": datetime.now().isoformat()
                }
                
                # Générer un ID unique
                job_offer["id"] = calculate_job_id(job_offer)
                
                # Vérifier s'il s'agit d'un doublon
                if not is_duplicate_job(job_offer, all_offers):
                    all_offers.append(job_offer)
                    logger.debug(f"Ajouté: {title} - {company}")
                
            except Exception as e:
                logger.error(f"Erreur lors de l'extraction d'une offre: {str(e)}")
                continue
        
        # Ajout d'un délai aléatoire pour éviter d'être bloqué
        delay = random.uniform(2, 5)
        time.sleep(delay)
        
        return all_offers
        
    except Exception as e:
        logger.error(f"Erreur lors du scraping pour le domaine {domain}: {str(e)}")
        return []

async def add_job_to_vector_db(api_url: str, job: Dict[str, Any], logger=None) -> bool:
    """Ajouter une offre d'emploi à la base vectorielle"""
    if logger is None:
        logger = setup_logger("job_scraper_morocco")
    
    try:
        job_data = {
            "content": job["description"],
            "metadata": {
                "title": job["title"],
                "company": job["company"],
                "location": job["location"],
                "domain": job["domain"],
                "url": job.get("url", ""),
                "source": job["source"],
                "date_collected": job["date_collected"]
            }
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{api_url}/upload/job",
                json=job_data
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ Offre ajoutée: {job['title']} (ID: {result.get('id', 'N/A')})")
                return True
            else:
                logger.error(f"❌ Erreur lors de l'ajout de l'offre {job['title']}: {response.status_code}")
                logger.error(f"Détails: {response.text}")
                return False
    except Exception as e:
        logger.error(f"Exception lors de l'ajout de l'offre {job['title']}: {str(e)}")
        return False

async def add_jobs_to_db(api_url: str, jobs: List[Dict[str, Any]], logger=None):
    """Ajouter toutes les offres d'emploi à la base vectorielle"""
    if logger is None:
        logger = setup_logger("job_scraper_morocco")
    
    logger.info(f"Ajout de {len(jobs)} offres à la base vectorielle...")
    
    success_count = 0
    for job in jobs:
        if await add_job_to_vector_db(api_url, job, logger):
            success_count += 1
    
    logger.info(f"{success_count}/{len(jobs)} offres ajoutées avec succès.")
    return success_count

async def main_async(args=None, logger=None):
    """Fonction principale asynchrone"""
    # Parser les arguments si non fournis
    if args is None:
        args = parse_arguments()
    
    # Configurer le logger si non fourni
    if logger is None:
        log_level = logging.DEBUG if args.verbose else logging.INFO
        logger = setup_logger("job_scraper_morocco", log_level=log_level)
    
    # S'assurer que le répertoire de sortie existe
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Si un domaine spécifique est demandé, ne scraper que celui-ci
    if args.domain:
        domains_to_scrape = {args.domain: DOMAINS[args.domain]}
        logger.info(f"Scraping uniquement pour le domaine: {args.domain}")
    else:
        # Sinon, scraper tous les domaines
        domains_to_scrape = DOMAINS
        logger.info(f"Scraping pour tous les domaines: {list(domains_to_scrape.keys())}")
    
    all_jobs = []
    
    # Scraper chaque domaine
    for domain, keywords in domains_to_scrape.items():
        jobs = scrape_jobs_for_domain(domain, keywords, args.max_results, logger)
        logger.info(f"Trouvé {len(jobs)} offres pour le domaine {domain}")
        all_jobs.extend(jobs)
    
    # S'il y a des offres, les sauvegarder
    if all_jobs:
        # Créer un nom de fichier basé sur la date/heure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.domain:
            output_file = os.path.join(args.output_dir, f"jobs_{args.domain}_ma_{timestamp}.json")
        else:
            output_file = os.path.join(args.output_dir, f"jobs_all_domains_ma_{timestamp}.json")
        
        # Sauvegarder les offres
        save_job_offers(all_jobs, output_file, logger)
        
        # Ajouter à la base vectorielle si demandé
        if args.add_to_db:
            await add_jobs_to_db(args.api_url, all_jobs, logger)
    else:
        logger.warning("Aucune offre d'emploi trouvée.")
    
    return all_jobs

def main():
    """Fonction principale"""
    args = parse_arguments()
    
    # Configurer le logger
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger("job_scraper_morocco", log_level=log_level)
    
    logger.info("Démarrage du scraper d'offres d'emploi au Maroc")
    
    # Exécuter la fonction asynchrone
    try:
        all_jobs = asyncio.run(main_async(args, logger))
        logger.info(f"Scraping terminé, {len(all_jobs)} offres collectées au total")
        return 0
    except KeyboardInterrupt:
        logger.info("Scraping interrompu par l'utilisateur")
        return 1
    except Exception as e:
        logger.error(f"Erreur lors du scraping: {str(e)}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main()) 