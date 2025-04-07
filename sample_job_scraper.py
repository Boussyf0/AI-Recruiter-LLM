#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
sample_job_scraper.py - Exemple de scraper utilisant les modules créés
Script d'exemple pour montrer comment utiliser les modules de scraping
"""

import os
import sys
import json
import argparse
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importer les modules
from scrapers.scraper_utils import (
    setup_logger, get_request_headers, safe_request, 
    extract_text_from_html, calculate_job_id,
    is_duplicate_job, extract_salary_from_text, 
    normalize_job_location, save_job_offers, 
    DOMAINS, CITIES_BY_COUNTRY
)
from infrastructure.proxies import get_random_proxy

# Configuration
DEFAULT_OUTPUT_DIR = "data/scraped_jobs"
DEFAULT_MAX_RESULTS = 5
DEFAULT_COUNTRY = "ma"

def parse_arguments():
    """Gère les arguments en ligne de commande"""
    parser = argparse.ArgumentParser(description="Exemple de scraper d'offres d'emploi")

    parser.add_argument("--domain", type=str, choices=DOMAINS.keys(), 
                      help="Domaine professionnel à scraper")
    parser.add_argument("--location", type=str, default=None,
                      help="Localisation pour la recherche (ville)")
    parser.add_argument("--max-results", type=int, default=DEFAULT_MAX_RESULTS,
                      help=f"Nombre maximum de résultats (défaut: {DEFAULT_MAX_RESULTS})")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                      help=f"Répertoire de sortie (défaut: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--country", type=str, default=DEFAULT_COUNTRY, choices=["fr", "ma"],
                      help=f"Pays de recherche (défaut: {DEFAULT_COUNTRY})")
    parser.add_argument("--verbose", action="store_true", help="Mode verbeux")

    return parser.parse_args()

def setup_environment(args):
    """Configure l'environnement d'exécution"""
    # Configurer le logger
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger("sample_scraper", log_level=log_level)
    
    # Créer le répertoire de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    
    return logger

def get_search_keywords(domain):
    """Obtient des mots-clés de recherche pour un domaine"""
    if domain and domain in DOMAINS:
        # Prendre jusqu'à 3 mots-clés du domaine
        import random
        keywords = DOMAINS[domain]
        if len(keywords) > 3:
            return random.sample(keywords, 3)
        return keywords
    
    # Mots-clés généraux si aucun domaine n'est spécifié
    return ["emploi", "job", "recrutement"]

def get_search_location(args):
    """Détermine la localisation de recherche"""
    if args.location:
        return args.location
    
    # Choisir une ville aléatoire du pays spécifié
    import random
    country_code = args.country
    cities = CITIES_BY_COUNTRY.get(country_code, [])
    
    if cities:
        return random.choice(cities)
    
    # Valeur par défaut
    return "France" if country_code == "fr" else "Maroc"

def build_search_url(keywords, location, country_code):
    """Construit l'URL de recherche pour le site de jobs"""
    # Exemple avec une URL fictive - à adapter selon le site cible
    base_url = "https://example-jobs.com/search"
    
    # Encoder les paramètres
    import urllib.parse
    query = " ".join(keywords)
    query_encoded = urllib.parse.quote(query)
    location_encoded = urllib.parse.quote(location)
    
    # Construire l'URL complète
    url = f"{base_url}?q={query_encoded}&l={location_encoded}"
    
    # Ajouter des paramètres spécifiques au pays
    if country_code == "fr":
        url += "&country=france"
    elif country_code == "ma":
        url += "&country=morocco"
    
    return url

def parse_job_listings(html_content, domain, country_code, logger):
    """Parse le HTML pour extraire les offres d'emploi"""
    from bs4 import BeautifulSoup
    
    job_offers = []
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Note: ceci est un exemple fictif - adapter selon la structure du site
    job_cards = soup.select("div.job-card")  # Sélecteur CSS fictif
    logger.info(f"Trouvé {len(job_cards)} offres d'emploi")
    
    for card in job_cards:
        try:
            # Extraire les informations (adapter selon la structure réelle)
            title_elem = card.select_one("h2.job-title")
            company_elem = card.select_one("div.company-name")
            location_elem = card.select_one("div.job-location")
            description_elem = card.select_one("div.job-description")
            
            title = title_elem.get_text(strip=True) if title_elem else "N/A"
            company = company_elem.get_text(strip=True) if company_elem else "N/A"
            location = location_elem.get_text(strip=True) if location_elem else "N/A"
            description = description_elem.get_text(strip=True) if description_elem else "N/A"
            
            # Normaliser l'emplacement
            location = normalize_job_location(location, country_code)
            
            # Essayer d'extraire le salaire
            salary = None
            if description:
                salary = extract_salary_from_text(description)
            
            # Créer l'objet offre
            job = {
                "title": title,
                "company": company,
                "location": location,
                "description": description,
                "domain": domain,
                "salary": salary,
                "country": "France" if country_code == "fr" else "Maroc",
                "date_collected": datetime.now().isoformat(),
                "source": "Exemple Scraper"
            }
            
            # Générer un ID unique
            job["id"] = calculate_job_id(job)
            
            # Ajouter à la liste
            job_offers.append(job)
            logger.debug(f"Offre extraite: {title} - {company}")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction d'une offre: {str(e)}")
            continue
    
    return job_offers

def mock_html_content():
    """Génère un contenu HTML fictif pour les tests"""
    html = """
    <html>
        <body>
            <div class="search-results">
                <div class="job-card">
                    <h2 class="job-title">Développeur Python Senior</h2>
                    <div class="company-name">Tech Solutions</div>
                    <div class="job-location">Paris, 75001</div>
                    <div class="job-description">
                        Nous recherchons un développeur Python expérimenté pour rejoindre notre équipe.
                        Salaire: entre 45K€ et 55K€ selon expérience.
                    </div>
                </div>
                <div class="job-card">
                    <h2 class="job-title">Data Engineer</h2>
                    <div class="company-name">Data Insights</div>
                    <div class="job-location">Lyon</div>
                    <div class="job-description">
                        Poste de Data Engineer pour travailler sur des projets Big Data.
                        Compétences: Python, Spark, Hadoop.
                    </div>
                </div>
                <div class="job-card">
                    <h2 class="job-title">DevOps Engineer</h2>
                    <div class="company-name">Cloud Systems</div>
                    <div class="job-location">Nantes</div>
                    <div class="job-description">
                        Ingénieur DevOps pour gérer notre infrastructure cloud.
                        Technologies: Docker, Kubernetes, AWS.
                        Rémunération: 50K€
                    </div>
                </div>
            </div>
        </body>
    </html>
    """
    return html

def scrape_jobs(args, logger):
    """Fonction principale de scraping"""
    # Déterminer les paramètres de recherche
    domain = args.domain
    keywords = get_search_keywords(domain)
    location = get_search_location(args)
    country_code = args.country
    
    logger.info(f"Recherche pour le domaine: {domain or 'tous'}")
    logger.info(f"Mots-clés: {keywords}")
    logger.info(f"Localisation: {location}")
    
    # Construire l'URL de recherche
    search_url = build_search_url(keywords, location, country_code)
    logger.info(f"URL de recherche: {search_url}")
    
    # En production, nous ferions une vraie requête HTTP:
    # headers = get_request_headers()
    # proxies = get_random_proxy()
    # response = safe_request(search_url, headers=headers, proxies=proxies)
    # if response:
    #     html_content = response.text
    
    # Pour ce test, utilisons un contenu HTML fictif
    logger.info("Utilisation d'un contenu HTML fictif pour le test")
    html_content = mock_html_content()
    
    # Parser les offres d'emploi
    job_offers = parse_job_listings(html_content, domain, country_code, logger)
    
    # Limiter au nombre max de résultats
    job_offers = job_offers[:args.max_results]
    logger.info(f"Nombre d'offres extraites: {len(job_offers)}")
    
    # Sauvegarder les offres
    if job_offers:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        domain_str = domain or "all"
        filename = f"jobs_{domain_str}_{country_code}_{timestamp}.json"
        output_path = os.path.join(args.output_dir, filename)
        
        save_job_offers(job_offers, output_path, logger)
        logger.info(f"Offres sauvegardées dans: {output_path}")
    else:
        logger.warning("Aucune offre d'emploi trouvée")
    
    return job_offers

def main():
    """Fonction principale"""
    # Parser les arguments
    args = parse_arguments()
    
    # Configurer l'environnement
    logger = setup_environment(args)
    
    try:
        # Exécuter le scraping
        logger.info("Démarrage du scraping d'offres d'emploi")
        job_offers = scrape_jobs(args, logger)
        
        # Afficher un résumé
        logger.info("=== Résumé du scraping ===")
        logger.info(f"Domaine: {args.domain or 'tous'}")
        logger.info(f"Pays: {args.country}")
        logger.info(f"Nombre d'offres: {len(job_offers)}")
        
        # Afficher des détails sur les offres trouvées
        for i, job in enumerate(job_offers, 1):
            logger.info(f"{i}. {job['title']} - {job['company']} ({job['location']})")
            if job.get('salary'):
                logger.info(f"   Salaire: {job['salary']}")
        
        logger.info("Scraping terminé avec succès")
        return 0
        
    except Exception as e:
        logger.error(f"Erreur lors du scraping: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 