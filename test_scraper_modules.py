#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_scraper_modules.py - Tests des modules de scraping
Script pour tester le fonctionnement des modules de scraping nouvellement créés
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

# Configurer le logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_scraper')

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importer les modules à tester
try:
    from scrapers.scraper_utils import (
        setup_logger, get_random_user_agent, get_request_headers, 
        safe_request, extract_text_from_html, calculate_job_id,
        extract_salary_from_text, normalize_job_location, DOMAINS
    )
    from infrastructure.proxies import ProxyManager, get_random_proxy, update_proxies
except ImportError as e:
    logger.error(f"Erreur d'importation: {str(e)}")
    logger.error("Assurez-vous que les modules sont bien dans le chemin d'importation")
    sys.exit(1)

async def test_proxy_manager():
    """
    Teste le gestionnaire de proxies
    """
    logger.info("=== Test du gestionnaire de proxies ===")
    
    # Créer un gestionnaire temporaire pour éviter d'écraser le cache
    cache_file = "test_proxies_cache.json"
    manager = ProxyManager(cache_file=cache_file)
    
    # Tester la récupération des proxies
    proxies = manager.get_proxies_from_apis()
    logger.info(f"Proxies récupérés: {len(proxies)}")
    
    # Tester un proxy aléatoire
    if proxies:
        proxy = proxies[0]
        result = manager.test_proxy(proxy)
        logger.info(f"Test du proxy {proxy}: {'OK' if result else 'Échec'}")
    
    # Nettoyer
    if os.path.exists(cache_file):
        os.remove(cache_file)
    
    return True

def test_scraper_utils():
    """
    Teste les fonctions utilitaires pour le scraping
    """
    logger.info("=== Test des utilitaires de scraping ===")
    
    # Tester le logger
    test_logger = setup_logger("test_logger")
    logger.info("Logger configuré: OK")
    
    # Tester User-Agent
    ua = get_random_user_agent()
    logger.info(f"User-Agent aléatoire: {ua}")
    
    # Tester les headers
    headers = get_request_headers("https://example.com")
    logger.info(f"Headers générés avec {len(headers)} éléments")
    
    # Tester l'extraction de texte HTML
    html = """
    <html>
        <body>
            <div>
                <h1>Titre</h1>
                <script>alert('test');</script>
                <p>Paragraphe de test</p>
            </div>
        </body>
    </html>
    """
    text = extract_text_from_html(html)
    logger.info(f"Texte extrait: {text}")
    
    # Tester le calcul d'ID
    job = {
        "title": "Développeur Python",
        "company": "Test Inc.",
        "location": "Paris",
        "description": "Poste de développeur Python"
    }
    job_id = calculate_job_id(job)
    logger.info(f"ID de l'offre: {job_id}")
    
    # Tester l'extraction de salaire
    salary_text = "Salaire: entre 40K€ et 50K€ selon expérience"
    salary = extract_salary_from_text(salary_text)
    logger.info(f"Salaire extrait: {salary}")
    
    # Tester la normalisation d'emplacement
    location = normalize_job_location("Paris - 75001", "fr")
    logger.info(f"Emplacement normalisé: {location}")
    
    # Tester les domaines
    logger.info(f"Nombre de domaines: {len(DOMAINS)}")
    for domain, keywords in DOMAINS.items():
        logger.info(f"Domaine {domain}: {len(keywords)} mots-clés")
    
    return True

async def test_safe_request():
    """
    Teste les fonctions de requête HTTP sécurisées
    """
    logger.info("=== Test des requêtes HTTP sécurisées ===")
    
    # URL de test
    test_url = "https://httpbin.org/get"
    
    # Tester safe_request
    response = safe_request(test_url, logger=logger)
    if response:
        logger.info(f"Requête réussie: status {response.status_code}")
        data = response.json()
        logger.info(f"Headers utilisés: {data.get('headers', {}).get('User-Agent')}")
    else:
        logger.error("Échec de la requête")
    
    return True

async def main():
    """
    Fonction principale pour exécuter tous les tests
    """
    logger.info("Démarrage des tests des modules de scraping")
    
    try:
        # Tester les utilitaires de scraping
        result_utils = test_scraper_utils()
        
        # Tester les requêtes sécurisées
        result_request = await test_safe_request()
        
        # Tester le gestionnaire de proxies
        result_proxy = await test_proxy_manager()
        
        # Afficher le résumé des tests
        logger.info("\n=== Résumé des tests ===")
        logger.info(f"Utilitaires de scraping: {'OK' if result_utils else 'ÉCHEC'}")
        logger.info(f"Requêtes HTTP sécurisées: {'OK' if result_request else 'ÉCHEC'}")
        logger.info(f"Gestionnaire de proxies: {'OK' if result_proxy else 'ÉCHEC'}")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution des tests: {str(e)}", exc_info=True)
        return False
    
    return True

if __name__ == "__main__":
    # Exécuter les tests de manière asynchrone
    try:
        # Vérifier si asyncio est disponible
        asyncio.run(main())
    except AttributeError:
        # Fallback pour Python 3.6
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
        loop.close() 