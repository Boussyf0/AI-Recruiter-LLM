#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
local_embeddings.py - Service d'embeddings local utilisant sentence-transformers
pour générer des embeddings sans dépendre d'une API externe
"""

import os
import numpy as np
from typing import List, Dict, Any, Union, Optional
import logging
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("local_embeddings")

# Cache global pour les modèles (partagé entre toutes les instances)
_MODEL_CACHE = {}

class LocalEmbeddingService:
    """Service d'embeddings local utilisant sentence-transformers"""
    
    def __init__(
        self, 
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        cache_dir: Optional[str] = "models/embeddings",
        device: str = None
    ):
        """
        Initialise le service d'embeddings local
        
        Args:
            model_name: Nom du modèle sentence-transformers à utiliser
            cache_dir: Répertoire de cache pour les modèles (si None, utilisera ~/.cache/torch)
            device: Périphérique à utiliser ('cpu', 'cuda', 'mps', ou None pour auto-détection)
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        self.model = None
        self.loaded = False
        
        # Créer le répertoire de cache si spécifié
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Vérifier si le modèle est déjà en cache
        cache_key = f"{model_name}_{device}"
        if cache_key in _MODEL_CACHE:
            logger.info(f"Utilisation du modèle {model_name} depuis le cache")
            self.model = _MODEL_CACHE[cache_key]
            self.loaded = True
        else:
            # Initialiser le modèle
            self._load_model()
        
    def _load_model(self):
        """Charge le modèle sentence-transformers"""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Détection automatique du device si non spécifié
            if self.device is None:
                try:
                    import torch
                    self.device = 'cuda' if torch.cuda.is_available() else \
                                 'mps' if torch.backends.mps.is_available() else 'cpu'
                except (ImportError, AttributeError):
                    self.device = 'cpu'
            
            logger.info(f"Chargement du modèle {self.model_name} sur {self.device}...")
            
            # Charger le modèle
            model_kwargs = {'cache_folder': self.cache_dir} if self.cache_dir else {}
            self.model = SentenceTransformer(self.model_name, device=self.device, **model_kwargs)
            self.loaded = True
            
            # Ajouter au cache global
            cache_key = f"{self.model_name}_{self.device}"
            _MODEL_CACHE[cache_key] = self.model
            
            logger.info(f"Modèle {self.model_name} chargé avec succès")
            
        except ImportError as e:
            logger.error(f"Erreur d'import: {str(e)}")
            logger.error("Veuillez installer sentence-transformers: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
            raise
    
    def get_embeddings(
        self, 
        text: Union[str, List[str]],
        domain: Optional[str] = None,
        batch_size: int = 8,
        normalize: bool = True
    ) -> Union[List[float], List[List[float]]]:
        """
        Génère des embeddings pour un texte ou une liste de textes
        
        Args:
            text: Le texte ou la liste de textes à encoder
            domain: Domaine optionnel pour le contexte (ignoré dans cette implémentation)
            batch_size: Taille des lots pour l'encodage
            normalize: Si vrai, normalise les vecteurs à une norme L2 de 1.0
        
        Returns:
            Liste de vecteurs d'embedding (liste de flottants ou liste de listes)
        """
        if not self.loaded:
            self._load_model()
            
        if not self.loaded:
            raise RuntimeError("Le modèle n'a pas pu être chargé")
        
        # Assurer que text est toujours une liste pour le traitement par lots
        is_single_text = isinstance(text, str)
        texts = [text] if is_single_text else text
        
        # Générer les embeddings
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 5,
                normalize_embeddings=normalize
            )
            
            # Convertir en liste Python standard
            result = embeddings.tolist()
            
            # Si entrée simple, retourner un seul vecteur
            return result[0] if is_single_text else result
        
        except Exception as e:
            logger.error(f"Erreur lors de la génération des embeddings: {str(e)}")
            raise
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calcule la similarité cosinus entre deux embeddings
        
        Args:
            embedding1: Premier vecteur d'embedding
            embedding2: Second vecteur d'embedding
            
        Returns:
            Score de similarité cosinus (entre -1 et 1, 1 étant identique)
        """
        # Convertir en numpy si ce n'est pas déjà fait
        v1 = np.array(embedding1)
        v2 = np.array(embedding2)
        
        # Normaliser si nécessaire
        if np.linalg.norm(v1) > 0:
            v1 = v1 / np.linalg.norm(v1)
        if np.linalg.norm(v2) > 0:
            v2 = v2 / np.linalg.norm(v2)
            
        # Calculer la similarité cosinus
        similarity = np.dot(v1, v2)
        
        return float(similarity)
    
    def get_batch_embeddings(
        self, 
        texts: List[str], 
        batch_size: int = 32
    ) -> Dict[str, List[float]]:
        """
        Génère des embeddings pour un lot de textes et retourne un dictionnaire
        
        Args:
            texts: Liste de textes à encoder
            batch_size: Taille des lots pour l'encodage
            
        Returns:
            Dictionnaire de textes et leurs embeddings correspondants
        """
        embeddings = self.get_embeddings(texts, batch_size=batch_size)
        return {text: emb for text, emb in zip(texts, embeddings)}


# Service singleton pour réutiliser efficacement l'instance
_SINGLETON_SERVICE = None

def get_quick_embeddings(
    text: Union[str, List[str]],
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
) -> Union[List[float], List[List[float]]]:
    """
    Fonction utilitaire pour obtenir des embeddings rapidement avec un singleton
    
    Args:
        text: Texte ou liste de textes à encoder
        model_name: Nom du modèle à utiliser
        
    Returns:
        Embeddings pour le texte fourni
    """
    global _SINGLETON_SERVICE
    
    # Créer le service s'il n'existe pas encore
    if _SINGLETON_SERVICE is None or _SINGLETON_SERVICE.model_name != model_name:
        _SINGLETON_SERVICE = LocalEmbeddingService(model_name=model_name)
    
    return _SINGLETON_SERVICE.get_embeddings(text)


if __name__ == "__main__":
    # Exemple d'utilisation
    embedding_service = LocalEmbeddingService()
    
    # Tester avec un texte simple
    test_text = "Ceci est un exemple de texte pour tester le service d'embeddings local."
    embeddings = embedding_service.get_embeddings(test_text)
    
    print(f"Dimensions de l'embedding: {len(embeddings)}")
    print(f"Premiers éléments: {embeddings[:5]}")
    
    # Tester avec plusieurs textes
    texts = [
        "Ingénieur en développement Python avec 5 ans d'expérience",
        "Développeur frontend React recherché pour startup innovante",
        "Expert en automatismes industriels et programmation Siemens"
    ]
    
    # Démontrer l'utilisation du cache (devrait être rapide car le modèle est déjà chargé)
    print("\nTester la réutilisation du modèle en cache:")
    service2 = LocalEmbeddingService()  # Devrait utiliser le modèle en cache
    batch_embeddings = service2.get_embeddings(texts)
    print(f"Nombre d'embeddings générés: {len(batch_embeddings)}")
    
    # Tester la fonction get_quick_embeddings (singleton)
    print("\nTester la fonction get_quick_embeddings:")
    quick_embeddings = get_quick_embeddings("Test du service singleton")
    print(f"Embedding généré avec {len(quick_embeddings)} dimensions")
    
    # Calculer et afficher les similarités
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            sim = embedding_service.compute_similarity(batch_embeddings[i], batch_embeddings[j])
            print(f"Similarité entre texte {i+1} et texte {j+1}: {sim:.4f}") 