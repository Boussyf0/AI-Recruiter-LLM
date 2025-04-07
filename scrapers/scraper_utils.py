#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
scraper_utils.py - Utilitaires communs pour tous les scrapers
Module partagé contenant des fonctions réutilisables pour le scraping
"""

import os
import random
import time
import logging
import json
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import httpx

# Configuration des user agents
DEFAULT_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPad; CPU OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
]

# Configuration des proxies (à remplir avec vos propres proxies si nécessaire)
DEFAULT_PROXIES = []

# Configuration des domaines professionnels
DOMAINS = {
    "informatique_reseaux": ["informatique", "développeur", "software", "web", "python", "java", "javascript", "devops", "cloud", "réseau", "cybersécurité", "IT"],
    "automatismes_info_industrielle": ["automatisme", "automaticien", "automate", "plc", "scada", "robot", "industrie 4.0"],
    "finance": ["finance", "comptabilité", "comptable", "audit", "analyste financier", "contrôle de gestion"],
    "genie_civil_btp": ["génie civil", "btp", "bâtiment", "construction", "structure", "architecte"],
    "genie_industriel": ["production", "lean", "qualité", "maintenance", "supply chain", "logistique"]
}

# Liste des villes importantes par pays
CITIES_BY_COUNTRY = {
    "fr": ["Paris", "Lyon", "Marseille", "Toulouse", "Nice", "Nantes", "Strasbourg", "Montpellier", "Bordeaux", "Lille"],
    "ma": ["Casablanca", "Rabat", "Marrakech", "Tanger", "Fès", "Meknès", "Agadir", "Tétouan", "Oujda", "Kénitra", "El Jadida", "Mohammedia"]
}

def setup_logger(name: str, log_file: str = None, console: bool = True, log_level: int = logging.INFO) -> logging.Logger:
    """Configurer un logger avec les paramètres spécifiés
    
    Args:
        name: Nom du logger
        log_file: Chemin vers le fichier de log (optionnel)
        console: Activer la sortie console
        log_level: Niveau de log
        
    Returns:
        Logger configuré
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Supprimer les handlers existants
    if logger.handlers:
        logger.handlers = []
    
    # Ajouter un handler de fichier si spécifié
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Ajouter un handler de console si demandé
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

def get_random_user_agent() -> str:
    """Retourne un User-Agent aléatoire
    
    Returns:
        User-Agent aléatoire
    """
    try:
        # Essayer d'utiliser fake_useragent si disponible
        from fake_useragent import UserAgent
        ua = UserAgent()
        return ua.random
    except (ImportError, Exception):
        # Fallback sur la liste prédéfinie
        return random.choice(DEFAULT_USER_AGENTS)

def get_random_proxy() -> Optional[Dict[str, str]]:
    """Retourne un proxy aléatoire ou None si aucun proxy n'est disponible
    
    Returns:
        Proxy au format dictionnaire ou None
    """
    if not DEFAULT_PROXIES:
        return None
    
    proxy = random.choice(DEFAULT_PROXIES)
    return {"http": proxy, "https": proxy}

def get_request_headers(referer: str = None) -> Dict[str, str]:
    """Génère des headers HTTP aléatoires pour les requêtes
    
    Args:
        referer: URL de référence (optionnel)
        
    Returns:
        Dictionnaire de headers HTTP
    """
    headers = {
        "User-Agent": get_random_user_agent(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "fr,fr-FR;q=0.8,en-US;q=0.5,en;q=0.3",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0"
    }
    
    if referer:
        headers["Referer"] = referer
    
    return headers

def safe_request(url: str, method: str = "GET", logger: logging.Logger = None, 
                max_retries: int = 3, retry_delay: float = 2.0, timeout: float = 30.0,
                headers: Dict[str, str] = None, proxies: Dict[str, str] = None,
                cookies: Dict[str, str] = None, **kwargs) -> Optional[requests.Response]:
    """Effectue une requête HTTP avec gestion des erreurs et retry
    
    Args:
        url: URL à requêter
        method: Méthode HTTP (GET, POST, etc.)
        logger: Logger pour les messages (optionnel)
        max_retries: Nombre maximum de tentatives
        retry_delay: Délai entre les tentatives (secondes)
        timeout: Timeout de la requête (secondes)
        headers: Headers HTTP (optionnels)
        proxies: Proxies à utiliser (optionnels)
        cookies: Cookies à utiliser (optionnels)
        **kwargs: Arguments supplémentaires pour requests
        
    Returns:
        Objet Response ou None en cas d'échec
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if headers is None:
        headers = get_request_headers()
    
    if proxies is None:
        proxies = get_random_proxy()
    
    # Paramètres de la requête
    request_kwargs = {
        "headers": headers,
        "timeout": timeout,
        **kwargs
    }
    
    if proxies:
        request_kwargs["proxies"] = proxies
    
    if cookies:
        request_kwargs["cookies"] = cookies
    
    # Tentatives avec retry
    for attempt in range(max_retries):
        try:
            response = requests.request(method, url, **request_kwargs)
            
            # Vérifier si la requête a été bloquée
            if "captcha" in response.text.lower() or response.status_code in (403, 429):
                logger.warning(f"Requête probablement bloquée: {url} (statut {response.status_code})")
                
                # Attendre plus longtemps avant la prochaine tentative
                time.sleep(retry_delay * (attempt + 1) * 2)
                
                # Changer de User-Agent et de proxy
                request_kwargs["headers"]["User-Agent"] = get_random_user_agent()
                if DEFAULT_PROXIES:
                    request_kwargs["proxies"] = get_random_proxy()
                
                continue
            
            # Vérifier le code de statut
            response.raise_for_status()
            
            # Attente aléatoire pour simuler un comportement humain
            time.sleep(random.uniform(1.0, 3.0))
            
            return response
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur de requête ({attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
            else:
                logger.error(f"Échec après {max_retries} tentatives: {url}")
                return None
    
    return None

async def async_safe_request(url: str, method: str = "GET", logger: logging.Logger = None,
                           max_retries: int = 3, retry_delay: float = 2.0, timeout: float = 30.0,
                           headers: Dict[str, str] = None, **kwargs) -> Optional[httpx.Response]:
    """Version asynchrone de safe_request
    
    Args:
        (Voir safe_request pour la description des paramètres)
        
    Returns:
        Objet Response ou None en cas d'échec
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if headers is None:
        headers = get_request_headers()
    
    # Paramètres de la requête
    request_kwargs = {
        "headers": headers,
        "timeout": timeout,
        **kwargs
    }
    
    # Tentatives avec retry
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.request(method, url, **request_kwargs)
                
                # Vérifier si la requête a été bloquée
                if "captcha" in response.text.lower() or response.status_code in (403, 429):
                    logger.warning(f"Requête probablement bloquée: {url} (statut {response.status_code})")
                    
                    # Attendre plus longtemps avant la prochaine tentative
                    await asyncio.sleep(retry_delay * (attempt + 1) * 2)
                    
                    # Changer de User-Agent
                    request_kwargs["headers"]["User-Agent"] = get_random_user_agent()
                    continue
                
                # Vérifier le code de statut
                response.raise_for_status()
                
                # Attente aléatoire pour simuler un comportement humain
                await asyncio.sleep(random.uniform(1.0, 3.0))
                
                return response
                
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logger.error(f"Erreur de requête ({attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (attempt + 1))
            else:
                logger.error(f"Échec après {max_retries} tentatives: {url}")
                return None
    
    return None

def save_job_offers(job_offers: List[Dict[str, Any]], output_path: str, logger: logging.Logger = None):
    """Sauvegarde les offres d'emploi au format JSON
    
    Args:
        job_offers: Liste des offres d'emploi
        output_path: Chemin vers le fichier de sortie
        logger: Logger pour les messages (optionnel)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Créer le répertoire parent si nécessaire
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "count": len(job_offers),
                "jobs": job_offers
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Sauvegarde de {len(job_offers)} offres dans {output_path}")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde des offres: {str(e)}")

def extract_text_from_html(html_content: str, selector: str = None) -> str:
    """Extrait le texte depuis du contenu HTML
    
    Args:
        html_content: Contenu HTML
        selector: Sélecteur CSS pour cibler un élément spécifique (optionnel)
        
    Returns:
        Texte extrait, nettoyé des balises HTML
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Supprimer les scripts et styles
    for script in soup(["script", "style"]):
        script.decompose()
    
    # Extraire le texte de l'élément sélectionné ou de toute la page
    if selector:
        elements = soup.select(selector)
        if elements:
            text = ' '.join([elem.get_text(strip=True, separator=' ') for elem in elements])
        else:
            text = ""
    else:
        text = soup.get_text(separator=' ')
    
    # Nettoyer le texte
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = ' '.join(chunk for chunk in chunks if chunk)
    
    return text

def calculate_job_id(job: Dict[str, Any]) -> str:
    """Calcule un ID unique pour une offre d'emploi
    
    Args:
        job: Offre d'emploi
        
    Returns:
        ID unique basé sur le contenu de l'offre
    """
    # Construire une chaîne avec les éléments clés de l'offre
    key_parts = [
        job.get('title', ''),
        job.get('company', ''),
        job.get('location', ''),
        job.get('description', '')[:100]  # Utiliser seulement le début de la description
    ]
    
    # Créer une chaîne unique pour le hachage
    key_string = '|'.join([str(part) for part in key_parts if part])
    
    # Générer un hash MD5
    return hashlib.md5(key_string.encode('utf-8')).hexdigest()

def clean_old_files(directory: str, pattern: str = "*.json", days: int = 7, logger: logging.Logger = None):
    """Supprime les fichiers plus anciens qu'un certain nombre de jours
    
    Args:
        directory: Répertoire contenant les fichiers
        pattern: Pattern glob pour les fichiers
        days: Nombre de jours (les fichiers plus anciens seront supprimés)
        logger: Logger pour les messages (optionnel)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if not os.path.exists(directory):
        logger.warning(f"Le répertoire {directory} n'existe pas")
        return
    
    cutoff_time = datetime.now() - timedelta(days=days)
    path = Path(directory)
    
    try:
        count = 0
        for file_path in path.glob(pattern):
            if file_path.is_file():
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if mtime < cutoff_time:
                    file_path.unlink()
                    count += 1
        
        if count > 0:
            logger.info(f"Suppression de {count} fichiers plus anciens que {days} jours dans {directory}")
    except Exception as e:
        logger.error(f"Erreur lors du nettoyage des anciens fichiers: {str(e)}")

def is_duplicate_job(job: Dict[str, Any], existing_jobs: List[Dict[str, Any]]) -> bool:
    """Vérifie si une offre d'emploi est un doublon
    
    Args:
        job: Offre d'emploi à vérifier
        existing_jobs: Liste des offres existantes
        
    Returns:
        True si l'offre est un doublon, False sinon
    """
    # Générer l'ID de l'offre
    job_id = calculate_job_id(job)
    
    # Vérifier si l'ID existe déjà
    for existing_job in existing_jobs:
        existing_id = existing_job.get('id') or calculate_job_id(existing_job)
        if job_id == existing_id:
            return True
        
        # Vérification supplémentaire basée sur le titre et l'entreprise
        if (job.get('title') == existing_job.get('title') and 
            job.get('company') == existing_job.get('company')):
            return True
    
    return False

# Fonctions d'aide pour l'analyse des offres d'emploi

def extract_salary_from_text(text: str) -> Optional[str]:
    """Tente d'extraire une information de salaire depuis un texte
    
    Args:
        text: Texte à analyser
        
    Returns:
        Chaîne contenant l'information de salaire, ou None si non trouvée
    """
    import re
    
    # Patterns de recherche pour différents formats de salaire
    patterns = [
        r'(\d+\s*[kK€][\s\-àa]*\d+\s*[kK€])',  # 30K€-45K€, 30k à 45k€
        r'(\d+\s*[kK€])',  # 40K€, 40k€
        r'(\d+\s*000[\s\-àa]*\d+\s*000[\s€]*)',  # 30 000 - 45 000 €
        r'(\d+\s*000[\s€]*)',  # 40 000 €
        r'(\d+[\s\-àa]*\d+[\s]*[€KDHdh]*)',  # 30-45 KDH, 10000-15000dh
        r'(entre[\s]+\d+[\s]*[€KDHdh]*[\s\-àaet]+\d+[\s]*[€KDHdh]*)',  # entre 30K et 45K
    ]
    
    for pattern in patterns:
        matches = re.search(pattern, text, re.IGNORECASE)
        if matches:
            return matches.group(1).strip()
    
    return None

def normalize_job_location(location: str, country_code: str = "fr") -> str:
    """Normalise l'emplacement d'une offre d'emploi
    
    Args:
        location: Emplacement brut
        country_code: Code pays (fr, ma)
        
    Returns:
        Emplacement normalisé
    """
    # Nettoyer l'emplacement
    location = location.strip()
    
    # Remplacer les caractères spéciaux
    location = location.replace('\u2013', '-').replace('\u00a0', ' ')
    
    # Normaliser le pays
    if country_code == "fr":
        for city in CITIES_BY_COUNTRY.get("fr", []):
            if city.lower() in location.lower():
                return city
        return "France"
    elif country_code == "ma":
        for city in CITIES_BY_COUNTRY.get("ma", []):
            if city.lower() in location.lower():
                return city
        return "Maroc"
    
    return location

# Fonctions d'aide pour les API

async def add_job_to_vector_db(api_url: str, job: Dict[str, Any], logger: logging.Logger = None) -> Optional[str]:
    """Ajoute une offre d'emploi à la base vectorielle
    
    Args:
        api_url: URL de l'API
        job: Offre d'emploi
        logger: Logger pour les messages (optionnel)
        
    Returns:
        ID de l'offre ajoutée, ou None en cas d'échec
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # S'assurer que l'offre a un ID
    if 'id' not in job:
        job['id'] = calculate_job_id(job)
    
    # Préparer les données pour l'API
    job_data = {
        "content": job["description"],
        "metadata": {
            "id": job["id"],
            "title": job["title"],
            "company": job["company"],
            "location": job["location"],
            "domain": job.get("domain", ""),
            "url": job.get("url", ""),
            "salary": job.get("salary", ""),
            "source": job.get("source", "scraping"),
            "country": job.get("country", ""),
            "date_collected": job.get("date_collected", datetime.now().isoformat())
        }
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{api_url}/upload/job",
                json=job_data
            )
            
            if response.status_code == 200:
                result = response.json()
                job_id = result.get('id', job['id'])
                logger.info(f"✅ Offre ajoutée: {job['title']} (ID: {job_id})")
                return job_id
            else:
                logger.error(f"❌ Erreur {response.status_code} lors de l'ajout de l'offre {job['title']}")
                logger.error(f"Détails: {response.text}")
                return None
    except Exception as e:
        logger.error(f"Exception lors de l'ajout de l'offre {job['title']}: {str(e)}")
        return None

if __name__ == "__main__":
    # Test des fonctions si le script est exécuté directement
    logger = setup_logger("scraper_utils_test", console=True)
    logger.info("Test des fonctions de scraper_utils.py")
    
    # Tester l'extraction de texte
    html = "<html><body><div>Test <script>alert('hello')</script><p>Ceci est un paragraphe</p></div></body></html>"
    text = extract_text_from_html(html)
    logger.info(f"Texte extrait: {text}")
    
    # Tester le calcul d'ID
    job = {
        "title": "Développeur Python",
        "company": "Test Company",
        "location": "Paris, France",
        "description": "Nous recherchons un développeur Python expérimenté."
    }
    job_id = calculate_job_id(job)
    logger.info(f"ID calculé: {job_id}")
    
    # Tester l'extraction de salaire
    salary_text = "Salaire : entre 40K€ et 50K€ selon expérience"
    salary = extract_salary_from_text(salary_text)
    logger.info(f"Salaire extrait: {salary}")
    
    logger.info("Tests terminés") 