#!/usr/bin/env python
import os
import sys
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from transformers import DistilBertModel, DistilBertTokenizer

# Ajouter le répertoire parent au chemin système pour permettre l'importation des modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 512
EMBEDDING_DIM = 768
DOMAIN_WEIGHT = 0.3  # Poids des embeddings de domaine dans le calcul final

# Définition des domaines
DOMAINS = [
    "informatique_reseaux",
    "automatismes_info_industrielle",
    "finance",
    "genie_civil_btp",
    "genie_industriel"
]

# Mots-clés par domaine pour l'enrichissement des embeddings
DOMAIN_KEYWORDS = {
    "informatique_reseaux": [
        "développeur", "devops", "cloud", "azure", "aws", "kubernetes", "docker", 
        "python", "java", "javascript", "react", "angular", "node", "réseau", 
        "cybersécurité", "git", "ci/cd", "microservices", "api", "backend", "frontend"
    ],
    "automatismes_info_industrielle": [
        "automate", "plc", "scada", "siemens", "schneider", "rockwell", "abb", "profinet", 
        "profibus", "modbus", "ethernet/ip", "programmation ladder", "contrôle", "capteur", 
        "vérin", "iiot", "industrie 4.0", "hmi", "ihm", "supervision", "tia portal"
    ],
    "finance": [
        "analyste", "financier", "comptabilité", "audit", "contrôle de gestion", "trésorerie", 
        "budget", "bilan", "investissement", "fiscal", "reporting", "prévision", "valorisation", 
        "fusion-acquisition", "modélisation financière", "kpi", "tableau de bord", "erp"
    ],
    "genie_civil_btp": [
        "structure", "béton", "génie civil", "bâtiment", "construction", "calcul", "chantier", 
        "fondation", "ouvrage d'art", "pont", "tunnel", "charpente", "travaux publics", "route", 
        "hydraulique", "géotechnique", "eurocodes", "plan", "autocad", "revit", "bim"
    ],
    "genie_industriel": [
        "production", "usine", "fabrication", "lean", "six sigma", "amélioration continue", 
        "qualité", "maintenance", "planification", "logistique", "supply chain", "5s", "kaizen", 
        "smed", "kanban", "erp", "mes", "processus", "industriel", "méthodes", "gestion"
    ]
}

class EmbeddingGenerator:
    """Générateur d'embeddings améliorés pour les CV et offres d'emploi"""
    
    def __init__(self, model_name: str = MODEL_NAME, max_length: int = MAX_LENGTH):
        """Initialise le générateur d'embeddings avec un modèle de langage"""
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Chargement du modèle {model_name} sur {self.device}...")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  # Mettre en mode évaluation
        
        # Générer les embeddings des domaines à l'avance
        self.domain_embeddings = self._generate_domain_embeddings()
        
        print("Générateur d'embeddings initialisé avec succès")
    
    def _generate_domain_embeddings(self) -> Dict[str, np.ndarray]:
        """Génère et met en cache les embeddings pour chaque domaine"""
        domain_embeddings = {}
        
        for domain in DOMAINS:
            # Combiner tous les mots-clés du domaine en un seul texte
            keywords = DOMAIN_KEYWORDS.get(domain, [])
            if not keywords:
                continue
                
            domain_text = f"Domaine: {domain.replace('_', ' ')}. Mots-clés: {', '.join(keywords)}"
            # Obtenir l'embedding pour le domaine
            with torch.no_grad():
                domain_embedding = self._get_bert_embedding(domain_text)
                domain_embeddings[domain] = domain_embedding
        
        return domain_embeddings
    
    def _get_bert_embedding(self, text: str) -> np.ndarray:
        """Calcule l'embedding BERT d'un texte"""
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Utiliser la représentation [CLS] de la dernière couche cachée
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        
        return embedding
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalise un embedding pour obtenir un vecteur unitaire"""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding
    
    def get_composite_embedding(self, text: str, domain: Optional[str] = None) -> np.ndarray:
        """
        Génère un embedding composite qui combine:
        1. L'embedding du texte
        2. L'embedding du domaine (si fourni)
        """
        # Obtenir l'embedding du texte
        text_embedding = self._get_bert_embedding(text)
        
        # Si un domaine est spécifié, combiner avec l'embedding du domaine
        if domain and domain in self.domain_embeddings:
            domain_embedding = self.domain_embeddings[domain]
            
            # Combiner avec une pondération
            embedding = (1 - DOMAIN_WEIGHT) * text_embedding + DOMAIN_WEIGHT * domain_embedding
            embedding = self._normalize_embedding(embedding)
        else:
            embedding = text_embedding
        
        return embedding
    
    def get_cv_embedding(self, cv_text: str, domain: Optional[str] = None) -> np.ndarray:
        """Génère un embedding optimisé pour un CV"""
        return self.get_composite_embedding(cv_text, domain)
    
    def get_job_embedding(self, job_text: str, domain: Optional[str] = None) -> np.ndarray:
        """Génère un embedding optimisé pour une offre d'emploi"""
        return self.get_composite_embedding(job_text, domain)
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calcule la similarité cosinus entre deux embeddings"""
        embedding1 = self._normalize_embedding(embedding1)
        embedding2 = self._normalize_embedding(embedding2)
        
        similarity = np.dot(embedding1, embedding2)
        return similarity

def test_embeddings():
    """Fonction de test pour les embeddings améliorés"""
    generator = EmbeddingGenerator()
    
    # Texte de test pour chaque domaine
    test_texts = {
        "informatique_reseaux": "Ingénieur DevOps expérimenté en cloud AWS et Kubernetes avec 5 ans d'expérience",
        "finance": "Analyste financier spécialisé en modélisation financière et évaluation d'investissements",
        "automatismes_info_industrielle": "Ingénieur automaticien expert en programmation d'automates Siemens et systèmes SCADA",
        "genie_civil_btp": "Ingénieur structure spécialisé dans le calcul de structures en béton armé et charpente métallique",
        "genie_industriel": "Responsable de production expert en méthodes Lean et amélioration continue"
    }
    
    # Matrice de résultats pour visualiser les similarités
    results = []
    domains = list(test_texts.keys())
    
    # Générer tous les embeddings
    embeddings = {}
    for domain, text in test_texts.items():
        embeddings[domain] = generator.get_composite_embedding(text, domain)
    
    # Calculer toutes les similarités
    for domain1 in domains:
        row = [domain1]
        for domain2 in domains:
            similarity = generator.compute_similarity(embeddings[domain1], embeddings[domain2])
            row.append(f"{similarity:.4f}")
        results.append(row)
    
    # Afficher les résultats
    print("\nMatrice de similarité entre les domaines:")
    headers = [""] + domains
    for row in results:
        print(f"{row[0]:<30}: {' '.join(row[1:])}")

if __name__ == "__main__":
    test_embeddings() 