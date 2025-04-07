#!/usr/bin/env python
# Script pour collecter des offres d'emploi depuis le web et les indexer dans notre base vectorielle

import requests
import asyncio
import httpx
import json
import pandas as pd
import argparse
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from bs4 import BeautifulSoup
import logging

# Importer les utilitaires de scraping communs
from scrapers.scraper_utils import (
    setup_logger, get_random_user_agent, get_request_headers,
    safe_request, extract_text_from_html, calculate_job_id,
    is_duplicate_job, normalize_job_location, save_job_offers,
    extract_salary_from_text, DOMAINS
)

# Importer le gestionnaire de proxies (optionnel)
try:
    from infrastructure.proxies import get_random_proxy
    HAS_PROXIES = True
except ImportError:
    HAS_PROXIES = False

# URL de recherche simulée (exemple avec Indeed pour simplifier)
BASE_URL = "https://www.indeed.com/jobs?q={query}&l={location}"

# Configuration par défaut
DEFAULT_API_URL = "http://localhost:8000/api/v1"
DEFAULT_LLM_URL = "http://localhost:8001"

def parse_args():
    parser = argparse.ArgumentParser(description="Collecter et stocker des offres d'emploi dans notre base vectorielle")
    parser.add_argument("--query", type=str, default="devops", help="Requête de recherche")
    parser.add_argument("--location", type=str, default="", help="Localisation (ex. 'Remote', 'Paris')")
    parser.add_argument("--domain", type=str, choices=list(DOMAINS.keys()), 
                        help="Domaine spécifique à cibler")
    parser.add_argument("--output", type=str, default="data/job_offers.json", 
                        help="Fichier de sortie (JSON ou CSV)")
    parser.add_argument("--max_results", type=int, default=10, 
                        help="Nombre maximum d'offres à collecter")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="URL de l'API backend")
    parser.add_argument("--llm-url", default=DEFAULT_LLM_URL, help="URL du service LLM")
    parser.add_argument("--add-to-db", action="store_true", help="Ajouter les offres à la base vectorielle")
    parser.add_argument("--verbose", action="store_true", help="Afficher des informations détaillées")
    parser.add_argument("--country", default="fr", choices=["fr", "ma"], help="Pays cible (fr: France, ma: Maroc)")
    return parser.parse_args()

def guess_domain_from_keywords(job_title: str, job_description: str) -> Optional[str]:
    """Détermine le domaine probable d'une offre d'emploi en fonction de ses mots-clés"""
    # Convertir en minuscules pour une recherche insensible à la casse
    text = (job_title + " " + job_description).lower()
    
    # Compter les occurrences de mots-clés pour chaque domaine
    domain_scores = {domain: 0 for domain in DOMAINS}
    
    for domain, keywords in DOMAINS.items():
        for keyword in keywords:
            if keyword.lower() in text:
                # Augmenter le score en fonction du nombre d'occurrences
                occurrences = text.count(keyword.lower())
                domain_scores[domain] += occurrences
    
    # Trouver le domaine avec le plus grand nombre de correspondances
    best_domain = max(domain_scores.items(), key=lambda x: x[1])
    
    # Si aucun mot-clé n'a été trouvé ou si le score est trop faible
    if best_domain[1] == 0:
        return None
    
    return best_domain[0]

def fetch_job_offers(query: str, location: str, max_results: int, country_code="fr", logger=None) -> List[Dict[str, Any]]:
    """Récupère les offres d'emploi depuis une source web"""
    # Utiliser le logger fourni ou créer un logger par défaut
    if logger is None:
        logger = setup_logger("job_scraper")
    
    # Formater la requête pour l'URL
    query_formatted = query.replace(' ', '+')
    location_formatted = location.replace(' ', '+')
    
    # Déterminer l'URL de base selon le pays
    if country_code == "ma":
        base_url = "https://ma.indeed.com"
    else:
        base_url = "https://fr.indeed.com"
    
    # Construire l'URL de recherche
    url = f"{base_url}/jobs?q={query_formatted}&l={location_formatted}"
    
    logger.info(f"Récupération des offres depuis {url}")
    
    try:
        # Obtenir les headers et proxy
        headers = get_request_headers(referer=base_url)
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
            logger.warning("Aucune offre trouvée. La structure HTML a peut-être changé.")
            return []
        
        job_cards = job_cards[:max_results]
        logger.info(f"Nombre d'offres trouvées: {len(job_cards)}")
        
        job_offers = []
        for card in job_cards:
            try:
                # Extraire les éléments clés
                title_elem = card.select_one("h2.jobTitle") or card.select_one("a.jobtitle")
                company_elem = card.select_one("span.companyName") or card.select_one("span.company")
                location_elem = card.select_one("div.companyLocation") or card.select_one("div.location")
                description_elem = card.select_one("div.job-snippet") or card.select_one("div.summary")
                
                # Extraire le texte
                job_title = title_elem.get_text(strip=True) if title_elem else "N/A"
                company = company_elem.get_text(strip=True) if company_elem else "N/A"
                location_raw = location_elem.get_text(strip=True) if location_elem else location
                job_description = description_elem.get_text(strip=True) if description_elem else "N/A"
                
                # Normaliser l'emplacement
                location_normalized = normalize_job_location(location_raw, country_code)
                
                # Essayer d'extraire le salaire de la description
                salary = extract_salary_from_text(job_description)
                
                # Essayer d'obtenir l'URL de l'offre
                job_url = ""
                url_elem = card.select_one("a.jcs-JobTitle") or card.select_one("a.jobtitle")
                if url_elem and 'href' in url_elem.attrs:
                    job_url = base_url + url_elem['href']
                
                # Détecter le domaine à partir des mots-clés
                domain = guess_domain_from_keywords(job_title, job_description)
                
                # Créer l'objet d'offre d'emploi
                offer = {
                    "title": job_title,
                    "company": company,
                    "location": location_normalized,
                    "description": job_description,
                    "salary": salary,
                    "domain": domain,
                    "url": job_url,
                    "source": "Indeed",
                    "country": "France" if country_code == "fr" else "Maroc",
                    "date_collected": datetime.now().isoformat()
                }
                
                # Générer un ID unique
                offer["id"] = calculate_job_id(offer)
                
                # Vérifier s'il s'agit d'un doublon
                if not is_duplicate_job(offer, job_offers):
                    job_offers.append(offer)
                    logger.debug(f"Ajouté: {job_title} - {company}")
                
            except Exception as e:
                logger.error(f"Erreur lors de l'extraction d'une offre: {str(e)}")
                continue
        
        logger.info(f"{len(job_offers)} offres collectées.")
        return job_offers
        
    except Exception as e:
        logger.error(f"Erreur lors du scraping: {str(e)}")
        return []

def save_offers_to_file(job_offers: List[Dict[str, Any]], output_path: str, logger=None):
    """Stocke les offres dans un fichier JSON ou CSV"""
    if logger is None:
        logger = setup_logger("job_scraper")
    
    # Utiliser la fonction commune pour JSON
    if output_path.endswith('.json'):
        save_job_offers(job_offers, output_path, logger)
    
    # Spécifique pour CSV
    elif output_path.endswith('.csv'):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df = pd.DataFrame(job_offers)
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Offres sauvegardées dans {output_path} (CSV)")
    
    else:
        logger.warning("Format de fichier non supporté. Utilisez .json ou .csv")

async def add_job_to_vector_db(api_url: str, llm_url: str, job: Dict[str, Any], logger=None) -> bool:
    """Ajouter une offre d'emploi à la base de vecteurs"""
    if logger is None:
        logger = setup_logger("job_scraper")
    
    try:
        # Préparer les données de l'offre
        job_data = {
            "content": job["description"],
            "metadata": {
                "title": job["title"],
                "company": job["company"],
                "location": job["location"],
                "domain": job["domain"] or "unknown",
                "source": job["source"],
                "date_collected": job["date_collected"]
            }
        }
        
        # Envoyer l'offre au backend
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{api_url}/upload/job",
                json=job_data
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ Offre ajoutée avec succès: {job['title']} (ID: {result.get('id', 'N/A')})")
                return True
            else:
                logger.error(f"❌ Erreur lors de l'ajout de l'offre {job['title']}: {response.status_code}")
                logger.error(f"Détails: {response.text}")
                return False
    except Exception as e:
        logger.error(f"Exception lors de l'ajout de l'offre {job['title']}: {str(e)}")
        return False

async def add_jobs_to_db(api_url: str, llm_url: str, job_offers: List[Dict[str, Any]], logger=None):
    """Ajouter toutes les offres d'emploi à la base de vecteurs"""
    if logger is None:
        logger = setup_logger("job_scraper")
    
    logger.info(f"Ajout de {len(job_offers)} offres d'emploi à la base de vecteurs...")
    logger.info(f"URL API: {api_url}")
    logger.info(f"URL LLM: {llm_url}")
    
    success_count = 0
    for job in job_offers:
        if await add_job_to_vector_db(api_url, llm_url, job, logger):
            success_count += 1
    
    logger.info(f"{success_count}/{len(job_offers)} offres ajoutées avec succès.")
    return success_count

async def main_async(args, logger=None):
    """Fonction principale asynchrone"""
    if logger is None:
        log_level = logging.DEBUG if args.verbose else logging.INFO
        logger = setup_logger("job_scraper", log_level=log_level)
    
    # Si un domaine est spécifié, utiliser les mots-clés correspondants
    if args.domain:
        domain_keywords = DOMAINS.get(args.domain, [])
        if domain_keywords:
            # Utiliser 3 mots-clés aléatoires du domaine
            import random
            keywords = random.sample(domain_keywords, min(3, len(domain_keywords)))
            query = " OR ".join(keywords)
            logger.info(f"Utilisation de la requête générée à partir du domaine {args.domain}: {query}")
            args.query = query
    
    # Récupérer les offres d'emploi
    job_offers = fetch_job_offers(
        query=args.query,
        location=args.location,
        max_results=args.max_results,
        country_code=args.country,
        logger=logger
    )
    
    if job_offers:
        # Sauvegarder les offres dans un fichier
        save_offers_to_file(job_offers, args.output, logger)
        
        # Ajouter les offres à la base de vecteurs si demandé
        if args.add_to_db:
            await add_jobs_to_db(args.api_url, args.llm_url, job_offers, logger)
    else:
        logger.warning("Aucune offre d'emploi trouvée.")
    
    return job_offers

def main():
    """Fonction principale"""
    args = parse_args()
    
    # Configurer le logger
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger("job_scraper", log_level=log_level)
    
    logger.info("Démarrage du scraper d'offres d'emploi")
    logger.info(f"Recherche pour: {args.query}")
    logger.info(f"Localisation: {args.location}")
    logger.info(f"Pays: {args.country}")
    
    # Exécuter la fonction asynchrone
    try:
        job_offers = asyncio.run(main_async(args, logger))
        logger.info(f"Scraping terminé, {len(job_offers)} offres collectées")
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