#!/usr/bin/env python
"""
Module pour gérer la base de vecteurs avec Qdrant
"""
import os
import json
import httpx
from typing import Dict, List, Optional, Any, Union
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
import numpy as np
import uuid
import logging

# Configurer le logger
logger = logging.getLogger(__name__)

try:
    from app.config import QDRANT_HOST, QDRANT_PORT, LLM_URL
except ImportError:
    print("Config module not found, using environment variables")
    QDRANT_HOST = os.getenv("QDRANT_HOST", "vector_db")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
    LLM_URL = os.getenv("LLM_URL", "http://llm:8001")

# Collections dans Qdrant
CV_COLLECTION = "cvs"
JOB_COLLECTION = "jobs"

# Dimensions d'embedding
EMBEDDING_DIM = 768  # Dimension standard pour DistilBERT

class VectorDB:
    """Classe pour gérer la base de vecteurs Qdrant"""
    
    def __init__(self):
        """Initialiser la connexion à Qdrant"""
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.init_collections()
    
    def init_collections(self):
        """Initialiser les collections Qdrant si elles n'existent pas"""
        try:
            # Vérifier et créer la collection CV si nécessaire
            self._ensure_collection_exists(CV_COLLECTION)
            
            # Vérifier et créer la collection Job si nécessaire
            self._ensure_collection_exists(JOB_COLLECTION)
        except Exception as e:
            print(f"Erreur lors de l'initialisation des collections: {e}")

    def _ensure_collection_exists(self, collection_name: str):
        """Vérifier si une collection existe et la créer si nécessaire"""
        # Vérifier si la collection existe
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if collection_name not in collection_names:
            print(f"Collection {collection_name} n'existe pas, création en cours...")
            # Créer la collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=EMBEDDING_DIM,
                    distance=models.Distance.COSINE
                )
            )
            print(f"Collection {collection_name} créée")
        else:
            print(f"Collection {collection_name} existe déjà")
    
    async def get_embeddings(self, text: str, domain: str = None) -> List[float]:
        """Obtenir les embeddings d'un texte depuis le service LLM"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                payload = {"text": text}
                if domain:
                    payload["domain"] = domain
                    
                response = await client.post(
                    f"{LLM_URL}/embeddings",
                    json=payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("embeddings", [])
                else:
                    logger.error(f"Erreur lors de la récupération des embeddings: {response.status_code}")
                    logger.error(f"Détails: {response.text}")
                    raise Exception(f"LLM API error: {response.status_code}")
        except Exception as e:
            logger.error(f"Exception lors de la récupération des embeddings: {str(e)}")
            raise Exception(f"Failed to get embeddings: {str(e)}")
    
    async def add_cv(self, cv_text: str, metadata: Dict[str, Any]) -> str:
        """Ajouter un CV à la base de vecteurs"""
        # Générer un ID unique pour le CV
        cv_id = metadata.get("id", str(uuid.uuid4()))
        
        # Obtenir les embeddings
        embeddings = await self.get_embeddings(cv_text)
        
        # Ajouter à la collection
        self.client.upsert(
            collection_name=CV_COLLECTION,
            points=[
                models.PointStruct(
                    id=cv_id,
                    vector=embeddings,
                    payload={"text": cv_text, "metadata": metadata}
                )
            ]
        )
        
        return cv_id
    
    async def add_job(self, job_text: str, metadata: Dict[str, Any]) -> str:
        """Ajouter une offre d'emploi à la base de vecteurs"""
        # Générer un ID unique pour l'offre
        job_id = metadata.get("id", str(uuid.uuid4()))
        
        # Obtenir les embeddings
        embeddings = await self.get_embeddings(job_text)
        
        # Ajouter à la collection
        self.client.upsert(
            collection_name=JOB_COLLECTION,
            points=[
                models.PointStruct(
                    id=job_id,
                    vector=embeddings,
                    payload={"text": job_text, "metadata": metadata}
                )
            ]
        )
        
        return job_id
    
    async def search_similar_cvs(self, query_text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Rechercher des CV similaires à une requête"""
        # Obtenir les embeddings de la requête
        query_embeddings = await self.get_embeddings(query_text)
        
        # Chercher les CV similaires
        results = self.client.search(
            collection_name=CV_COLLECTION,
            query_vector=query_embeddings,
            limit=limit
        )
        
        # Formater les résultats
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": result.id,
                "score": result.score,
                "text": result.payload.get("text", ""),
                "metadata": result.payload.get("metadata", {})
            })
        
        return formatted_results
    
    async def search_similar_jobs(self, cv_text: str, limit: int = 5, domain: str = None) -> List[Dict[str, Any]]:
        """Rechercher des offres d'emploi adaptées à un CV"""
        # Obtenir les embeddings du CV
        cv_embeddings = await self.get_embeddings(cv_text, domain)
        
        # Préparer des filtres éventuels basés sur le domaine
        search_params = None
        if domain:
            # On peut filtrer par domaine si spécifié, mais aussi inclure des domaines proches
            # Syntaxe mise à jour pour être compatible avec l'API Qdrant
            domain_filter = {
                "must": [
                    {
                        "key": "metadata.domain",
                        "match": {
                            "value": domain
                        }
                    }
                ]
            }
            # Créer un objet SearchParams valide pour Qdrant
            # En utilisant filter = None par défaut
            search_params = models.SearchParams()
        
        # Chercher les offres similaires sans filtre (pour l'instant)
        results = self.client.search(
            collection_name=JOB_COLLECTION,
            query_vector=cv_embeddings,
            limit=limit
        )
        
        # Si un filtre de domaine est demandé, faire un filtrage manuel en post-traitement
        if domain:
            # Filtrer les résultats pour ne garder que ceux du domaine spécifié
            filtered_results = [
                result for result in results 
                if result.payload.get("metadata", {}).get("domain") == domain
            ]
            
            # Si des résultats filtrés existent, les utiliser, sinon garder les résultats originaux
            if filtered_results:
                results = filtered_results[:limit]  # S'assurer de respecter la limite
        
        # Formater les résultats
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": result.id,
                "score": result.score,
                "text": result.payload.get("text", ""),
                "metadata": result.payload.get("metadata", {})
            })
        
        return formatted_results
    
    def get_cv_by_id(self, cv_id: str) -> Optional[Dict[str, Any]]:
        """Récupérer un CV par son ID"""
        try:
            result = self.client.retrieve(
                collection_name=CV_COLLECTION,
                ids=[cv_id]
            )
            
            if not result:
                return None
            
            return {
                "id": result[0].id,
                "text": result[0].payload.get("text", ""),
                "metadata": result[0].payload.get("metadata", {})
            }
        except Exception as e:
            print(f"Erreur lors de la récupération du CV: {e}")
            return None
    
    def get_job_by_id(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Récupérer une offre d'emploi par son ID"""
        try:
            result = self.client.retrieve(
                collection_name=JOB_COLLECTION,
                ids=[job_id]
            )
            
            if not result:
                return None
            
            return {
                "id": result[0].id,
                "text": result[0].payload.get("text", ""),
                "metadata": result[0].payload.get("metadata", {})
            }
        except Exception as e:
            print(f"Erreur lors de la récupération de l'offre: {e}")
            return None

# Instance singleton
vector_db = VectorDB() 