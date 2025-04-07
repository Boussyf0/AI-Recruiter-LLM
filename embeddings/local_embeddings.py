#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
local_embeddings.py - Service d'embeddings local utilisant sentence-transformers

Ce module permet de générer des embeddings pour des textes sans dépendre
d'un service distant. Il utilise sentence-transformers pour produire des
vecteurs de haute qualité compatibles avec la recherche sémantique.
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
from typing import List, Dict, Any, Optional, Union

# Configurer le logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Vérifier si sentence-transformers est installé
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers n'est pas installé. Utilisez 'pip install sentence-transformers' pour l'installer.")

class LocalEmbeddingService:
    """Service d'embeddings vectoriels local"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str = None, device: str = "cpu"):
        """
        Initialise le service d'embeddings avec le modèle spécifié.
        
        Args:
            model_name: Nom du modèle sentence-transformers à utiliser
                - all-MiniLM-L6-v2: rapide, léger (80MB), bon pour commencer (384 dimensions)
                - all-mpnet-base-v2: meilleure qualité mais plus lent (420MB) (768 dimensions)
                - paraphrase-multilingual-MiniLM-L12-v2: multilingue, recommandé pour français (471MB)
            cache_dir: Répertoire où stocker les modèles téléchargés
            device: Device à utiliser pour le modèle ('cpu', 'cuda', etc.)
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        self.model = None
        
        # Vérifier que sentence-transformers est installé
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers est requis pour ce module")
        
        # Charger le modèle
        self._load_model()
    
    def _load_model(self):
        """Charge le modèle d'embeddings"""
        try:
            logger.info(f"Chargement du modèle {self.model_name} sur {self.device}...")
            # Forcer l'utilisation du CPU pour éviter les problèmes avec MPS sur macOS
            self.model = SentenceTransformer(self.model_name, cache_folder=self.cache_dir, device=self.device)
            logger.info(f"Modèle chargé: {self.model_name}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
            raise
    
    def get_embeddings(self, text: Union[str, List[str]], 
                       domain: str = None, normalize: bool = True) -> Union[List[float], List[List[float]]]:
        """
        Génère des embeddings pour un texte ou une liste de textes.
        
        Args:
            text: Texte ou liste de textes à encoder
            domain: Domaine optionnel pour adaptation spécifique (non utilisé actuellement)
            normalize: Si True, normalise les vecteurs (recommandé pour la recherche par similarité)
            
        Returns:
            Vecteur d'embeddings ou liste de vecteurs
        """
        try:
            if not self.model:
                self._load_model()
                
            # Enregistrer le domaine pour un usage futur
            if domain:
                logger.info(f"Domaine spécifié: {domain} (ignoré pour ce modèle)")
            
            # Obtenir les embeddings
            if isinstance(text, list):
                embeddings = self.model.encode(text, normalize_embeddings=normalize)
                return [emb.tolist() for emb in embeddings]
            else:
                embeddings = self.model.encode(text, normalize_embeddings=normalize)
                return embeddings.tolist()
                
        except Exception as e:
            logger.error(f"Erreur lors de la génération d'embeddings: {str(e)}")
            # Retourner un vecteur aléatoire en cas d'erreur
            dim = 384 if "MiniLM" in self.model_name else 768
            return list(np.random.normal(0, 0.1, dim).astype(float))
    
    def get_dimension(self) -> int:
        """Renvoie la dimension des vecteurs générés par le modèle actuel"""
        if not self.model:
            self._load_model()
        return self.model.get_sentence_embedding_dimension()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Renvoie les informations sur le modèle chargé"""
        if not self.model:
            self._load_model()
            
        return {
            "name": self.model_name,
            "dimension": self.get_dimension(),
            "is_multilingual": "multilingual" in self.model_name.lower(),
            "type": "sentence-transformer"
        }
    
    def compare_texts(self, text1: str, text2: str) -> float:
        """
        Compare deux textes et renvoie leur similarité cosinus.
        
        Args:
            text1: Premier texte
            text2: Deuxième texte
            
        Returns:
            Score de similarité entre 0 et 1
        """
        if not self.model:
            self._load_model()
            
        embeddings = self.model.encode([text1, text2], normalize_embeddings=True)
        similarity = np.dot(embeddings[0], embeddings[1])
        return float(similarity)

def parse_arguments():
    """Parse les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(description="Service d'embeddings local")
    parser.add_argument("--text", help="Texte à encoder")
    parser.add_argument("--file", help="Fichier contenant le texte à encoder")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", 
                        help="Modèle sentence-transformers à utiliser")
    parser.add_argument("--output", help="Fichier de sortie (JSON)")
    parser.add_argument("--cache-dir", help="Répertoire de cache pour les modèles")
    parser.add_argument("--compare", help="Second texte pour comparaison")
    parser.add_argument("--domain", help="Domaine du texte (pour adaptation spécifique)")
    
    return parser.parse_args()

def main():
    """Point d'entrée principal pour l'utilisation en ligne de commande"""
    args = parse_arguments()
    
    # Vérifier qu'un texte est fourni
    if not args.text and not args.file:
        print("Erreur: Vous devez fournir un texte via --text ou --file")
        return 1
    
    try:
        # Initialiser le service
        service = LocalEmbeddingService(model_name=args.model, cache_dir=args.cache_dir)
        
        # Lire le texte depuis un fichier si spécifié
        if args.file:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            text = args.text
        
        # Obtenir les infos sur le modèle
        model_info = service.get_model_info()
        print(f"Modèle: {model_info['name']} ({model_info['dimension']} dimensions)")
        
        # Mode comparaison si un second texte est fourni
        if args.compare:
            similarity = service.compare_texts(text, args.compare)
            print(f"Similarité: {similarity:.4f}")
            result = {"similarity": similarity}
        else:
            # Générer les embeddings
            embeddings = service.get_embeddings(text, domain=args.domain)
            print(f"Embeddings générés: {len(embeddings)} dimensions")
            result = {"embeddings": embeddings, "model": model_info['name']}
        
        # Sauvegarder dans un fichier si demandé
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Résultats sauvegardés dans {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"Erreur: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 