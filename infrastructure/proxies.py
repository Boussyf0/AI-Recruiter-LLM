#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
proxies.py - Gestion des proxies pour le scraping
Module pour gérer automatiquement une liste de proxies pour le scraping
"""

import os
import json
import random
import time
import logging
import asyncio
import requests
import httpx
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('proxies')

# Fichier de cache pour les proxies
DEFAULT_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/cache/proxies.json')

# Sources de proxies gratuits (à utiliser avec précaution)
FREE_PROXY_APIS = [
    "https://www.proxyscan.io/api/proxy?format=json&type=http,https&limit=20",
    "https://api.proxyscrape.com/v2/?request=getproxies&protocol=http&timeout=10000&country=all&ssl=all&anonymity=all&simplified=true"
]

# Liste de proxies statiques (à remplir avec vos propres proxies)
STATIC_PROXIES = [
    # Format: "http://username:password@ip:port"
]

class ProxyManager:
    """Gestionnaire de proxies pour le scraping"""
    
    def __init__(self, cache_file: str = DEFAULT_CACHE_FILE, test_url: str = "http://httpbin.org/ip"):
        """
        Initialise le gestionnaire de proxies
        
        Args:
            cache_file: Chemin vers le fichier de cache des proxies
            test_url: URL utilisée pour tester les proxies
        """
        self.cache_file = cache_file
        self.test_url = test_url
        self.proxies = []
        self.last_update = None
        
        # Charger les proxies depuis le cache si disponible
        self.load_from_cache()
    
    def load_from_cache(self) -> bool:
        """
        Charge les proxies depuis le fichier de cache
        
        Returns:
            True si le chargement a réussi, False sinon
        """
        if not os.path.exists(self.cache_file):
            logger.info(f"Fichier de cache {self.cache_file} introuvable")
            return False
        
        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
                
            self.proxies = data.get('proxies', [])
            last_update_str = data.get('last_update')
            
            if last_update_str:
                self.last_update = datetime.fromisoformat(last_update_str)
            
            logger.info(f"Chargé {len(self.proxies)} proxies depuis le cache")
            return len(self.proxies) > 0
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du cache: {str(e)}")
            return False
    
    def save_to_cache(self) -> bool:
        """
        Sauvegarde les proxies dans le fichier de cache
        
        Returns:
            True si la sauvegarde a réussi, False sinon
        """
        if not self.proxies:
            logger.warning("Aucun proxy à sauvegarder")
            return False
        
        try:
            # Créer le répertoire parent si nécessaire
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            
            with open(self.cache_file, 'w') as f:
                json.dump({
                    'proxies': self.proxies,
                    'last_update': datetime.now().isoformat()
                }, f, indent=2)
            
            logger.info(f"Sauvegardé {len(self.proxies)} proxies dans le cache")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du cache: {str(e)}")
            return False
    
    def get_random_proxy(self) -> Optional[Dict[str, str]]:
        """
        Retourne un proxy aléatoire de la liste
        
        Returns:
            Dictionnaire proxy ou None si aucun proxy n'est disponible
        """
        # Si aucun proxy n'est disponible ou si la dernière mise à jour est trop ancienne
        if not self.proxies or (self.last_update and (datetime.now() - self.last_update > timedelta(hours=6))):
            self.update_proxies()
        
        if not self.proxies:
            return None
        
        proxy_url = random.choice(self.proxies)
        return {
            "http": proxy_url,
            "https": proxy_url
        }
    
    def test_proxy(self, proxy_url: str, timeout: float = 5.0) -> bool:
        """
        Teste si un proxy fonctionne
        
        Args:
            proxy_url: URL du proxy à tester
            timeout: Timeout en secondes
            
        Returns:
            True si le proxy fonctionne, False sinon
        """
        proxies = {
            "http": proxy_url,
            "https": proxy_url
        }
        
        try:
            response = requests.get(
                self.test_url,
                proxies=proxies,
                timeout=timeout,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
            )
            
            return response.status_code == 200
            
        except Exception:
            return False
    
    async def test_proxy_async(self, proxy_url: str, timeout: float = 5.0) -> bool:
        """
        Version asynchrone de test_proxy
        
        Args:
            proxy_url: URL du proxy à tester
            timeout: Timeout en secondes
            
        Returns:
            True si le proxy fonctionne, False sinon
        """
        proxies = {
            "http://": proxy_url,
            "https://": proxy_url
        }
        
        try:
            async with httpx.AsyncClient(proxies=proxies, timeout=timeout) as client:
                response = await client.get(self.test_url)
                return response.status_code == 200
                
        except Exception:
            return False
    
    def get_proxies_from_apis(self) -> List[str]:
        """
        Récupère une liste de proxies publics depuis les APIs
        
        Returns:
            Liste d'URLs de proxies
        """
        all_proxies = []
        
        for api_url in FREE_PROXY_APIS:
            try:
                logger.info(f"Récupération des proxies depuis {api_url}")
                response = requests.get(api_url, timeout=10)
                
                if response.status_code != 200:
                    logger.warning(f"Échec de la récupération des proxies depuis {api_url}")
                    continue
                
                # Différents formats de réponse selon l'API
                if "proxyscan.io" in api_url:
                    # Format: [{"Ip":"1.2.3.4","Port":"8080","Type":"http",...}]
                    data = response.json()
                    for item in data:
                        ip = item.get("Ip")
                        port = item.get("Port")
                        if ip and port:
                            all_proxies.append(f"http://{ip}:{port}")
                
                elif "proxyscrape.com" in api_url:
                    # Format: texte brut avec une ligne par proxy (ip:port)
                    lines = response.text.strip().split("\n")
                    for line in lines:
                        if ":" in line:
                            all_proxies.append(f"http://{line.strip()}")
                
                else:
                    # Format inconnu, essayer d'analyser comme texte brut (ip:port)
                    text = response.text
                    if ":" in text:
                        lines = text.strip().split("\n")
                        for line in lines:
                            if ":" in line:
                                all_proxies.append(f"http://{line.strip()}")
            
            except Exception as e:
                logger.error(f"Erreur lors de la récupération des proxies depuis {api_url}: {str(e)}")
        
        logger.info(f"Récupéré {len(all_proxies)} proxies depuis les APIs")
        return all_proxies
    
    async def update_proxies_async(self) -> int:
        """
        Met à jour la liste des proxies de manière asynchrone
        
        Returns:
            Nombre de proxies fonctionnels trouvés
        """
        # Combinaison des proxies statiques et des proxies depuis les APIs
        all_proxies = STATIC_PROXIES.copy()
        all_proxies.extend(self.get_proxies_from_apis())
        
        if not all_proxies:
            logger.warning("Aucun proxy trouvé")
            return 0
        
        logger.info(f"Test de {len(all_proxies)} proxies...")
        
        # Tester tous les proxies en parallèle
        tasks = [self.test_proxy_async(proxy) for proxy in all_proxies]
        results = await asyncio.gather(*tasks)
        
        # Filtrer les proxies fonctionnels
        working_proxies = [proxy for proxy, works in zip(all_proxies, results) if works]
        
        self.proxies = working_proxies
        self.last_update = datetime.now()
        
        # Sauvegarder dans le cache
        self.save_to_cache()
        
        logger.info(f"Mise à jour terminée. {len(working_proxies)}/{len(all_proxies)} proxies fonctionnels.")
        return len(working_proxies)
    
    def update_proxies(self) -> int:
        """
        Met à jour la liste des proxies
        
        Returns:
            Nombre de proxies fonctionnels trouvés
        """
        # Combinaison des proxies statiques et des proxies depuis les APIs
        all_proxies = STATIC_PROXIES.copy()
        all_proxies.extend(self.get_proxies_from_apis())
        
        if not all_proxies:
            logger.warning("Aucun proxy trouvé")
            return 0
        
        logger.info(f"Test de {len(all_proxies)} proxies...")
        
        # Tester tous les proxies
        working_proxies = []
        for proxy in all_proxies:
            if self.test_proxy(proxy):
                working_proxies.append(proxy)
                logger.debug(f"Proxy fonctionnel: {proxy}")
            else:
                logger.debug(f"Proxy non fonctionnel: {proxy}")
        
        self.proxies = working_proxies
        self.last_update = datetime.now()
        
        # Sauvegarder dans le cache
        self.save_to_cache()
        
        logger.info(f"Mise à jour terminée. {len(working_proxies)}/{len(all_proxies)} proxies fonctionnels.")
        return len(working_proxies)

# Instance globale du gestionnaire de proxies
_proxy_manager = None

def get_proxy_manager() -> ProxyManager:
    """
    Retourne l'instance globale du gestionnaire de proxies
    
    Returns:
        Instance du gestionnaire de proxies
    """
    global _proxy_manager
    if _proxy_manager is None:
        _proxy_manager = ProxyManager()
    return _proxy_manager

def get_random_proxy() -> Optional[Dict[str, str]]:
    """
    Fonction utilitaire pour obtenir un proxy aléatoire
    
    Returns:
        Dictionnaire de proxy ou None si aucun proxy n'est disponible
    """
    manager = get_proxy_manager()
    return manager.get_random_proxy()

async def update_proxies() -> int:
    """
    Fonction utilitaire pour mettre à jour les proxies
    
    Returns:
        Nombre de proxies fonctionnels trouvés
    """
    manager = get_proxy_manager()
    return await manager.update_proxies_async()

if __name__ == "__main__":
    # Test du gestionnaire de proxies
    async def main():
        manager = ProxyManager()
        
        print("Mise à jour des proxies...")
        proxy_count = await manager.update_proxies_async()
        
        if proxy_count > 0:
            proxy = manager.get_random_proxy()
            print(f"Proxy aléatoire: {proxy}")
        else:
            print("Aucun proxy disponible")
    
    asyncio.run(main()) 