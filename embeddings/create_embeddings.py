#!/usr/bin/env python
# Script pour générer des embeddings à partir de CV et d'offres d'emploi

import os
import argparse
import json
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Générer des embeddings à partir de CV et d'offres d'emploi")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Fichier CSV contenant les textes (CV ou offres d'emploi)")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Fichier de sortie pour les embeddings (format NPZ)")
    parser.add_argument("--text_column", type=str, default="text",
                        help="Nom de la colonne contenant les textes")
    parser.add_argument("--domain_mapping", type=str, default="models/cv_classifier/final/domain_mapping.json",
                        help="Fichier JSON contenant le mapping des domaines")
    
    return parser.parse_args()

def generate_mock_embeddings(texts, dim=768):
    """Génère des embeddings simulés pour démonstration"""
    print(f"Génération de {len(texts)} embeddings simulés...")
    np.random.seed(42)  # Pour reproductibilité
    return np.random.normal(0, 1, (len(texts), dim))

def get_domains():
    """Retourne les domaines depuis le fichier de mapping ou les valeurs par défaut"""
    domains = [
        "informatique_reseaux",
        "automatismes_info_industrielle",
        "finance",
        "civil_btp",
        "genie_industriel"
    ]
    return domains

def classify_texts(texts, domains):
    """Classifie les textes en domaines basés sur des mots-clés simples"""
    print("Classification des textes par domaine...")
    domain_keywords = {
        "informatique_reseaux": ["informatique", "réseau", "devops", "docker", "kubernetes", "cloud", "développeur", "python", "javascript"],
        "automatismes_info_industrielle": ["automatisme", "siemens", "automate", "scada", "industriel", "tia portal", "s7"],
        "finance": ["finance", "bancaire", "investissement", "analyste", "financier", "économie", "budget", "comptabilité"],
        "civil_btp": ["génie civil", "btp", "structure", "bâtiment", "construction", "chantier", "béton", "architecture"],
        "genie_industriel": ["production", "industrielle", "lean", "logistique", "supply chain", "amélioration continue", "usine"]
    }
    
    classifications = []
    scores = []
    
    for text in texts:
        text = text.lower()
        domain_scores = {}
        
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword.lower() in text)
            domain_scores[domain] = score
        
        # Trouver le domaine avec le score le plus élevé
        max_domain = max(domain_scores, key=domain_scores.get)
        max_score = domain_scores[max_domain]
        
        if max_score == 0:
            # Si aucun mot-clé trouvé, attribuer le domaine par défaut
            max_domain = "informatique_reseaux"
            confidence = 0.5
        else:
            confidence = min(1.0, max_score / 5)  # Convertir le score en confiance entre 0 et 1
        
        classifications.append(max_domain)
        scores.append(confidence)
    
    return classifications, scores

def main():
    args = parse_args()
    
    # Charger les données
    print(f"Chargement des données depuis {args.input_file}...")
    try:
        df = pd.read_csv(args.input_file)
    except Exception as e:
        print(f"Erreur lors du chargement du fichier CSV: {e}")
        return
    
    if args.text_column not in df.columns:
        available_cols = ", ".join(df.columns)
        print(f"La colonne '{args.text_column}' n'existe pas dans le fichier d'entrée.")
        print(f"Colonnes disponibles: {available_cols}")
        return
    
    texts = df[args.text_column].fillna("").tolist()
    
    # Obtenir les domaines
    try:
        with open(args.domain_mapping, 'r') as f:
            domain_mapping = json.load(f)
            domains = list(domain_mapping.values())
    except Exception as e:
        print(f"Erreur lors du chargement du mapping des domaines: {e}")
        print("Utilisation des domaines par défaut")
        domains = get_domains()
    
    # Générer des embeddings simulés
    embeddings = generate_mock_embeddings(texts)
    
    # Classifier les textes par domaine
    classifications, confidence_scores = classify_texts(texts, domains)
    
    # Sauvegarder les embeddings
    print(f"Sauvegarde des embeddings dans {args.output_file}...")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    
    np.savez_compressed(
        args.output_file,
        embeddings=embeddings,
        texts=texts,
        classifications=classifications,
        confidence_scores=confidence_scores,
        domains=domains,
        metadata=json.dumps({
            "input_file": args.input_file,
            "text_column": args.text_column,
            "embedding_dim": embeddings.shape[1],
            "num_samples": embeddings.shape[0]
        })
    )
    
    print(f"Embeddings générés et sauvegardés. Forme: {embeddings.shape}")
    print("\nClassifications par domaine:")
    for i, (text, domain, confidence) in enumerate(zip(texts, classifications, confidence_scores)):
        short_text = text[:50] + "..." if len(text) > 50 else text
        print(f"{i+1}. {short_text} -> {domain} (confiance: {confidence:.2f})")
    
if __name__ == "__main__":
    main()

