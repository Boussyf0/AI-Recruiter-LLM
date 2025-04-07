#!/usr/bin/env python
# Script pour comparer des CV et des offres d'emploi et trouver les meilleures correspondances

import os
import argparse
import numpy as np
from tabulate import tabulate
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Comparer des CV et des offres d'emploi")
    parser.add_argument("--cv_embeddings", type=str, default="data/test/cv_embeddings.npz",
                        help="Fichier NPZ contenant les embeddings des CV")
    parser.add_argument("--job_embeddings", type=str, default="data/test/job_embeddings.npz",
                        help="Fichier NPZ contenant les embeddings des offres d'emploi")
    parser.add_argument("--top_n", type=int, default=3,
                        help="Nombre de meilleures correspondances à afficher")
    
    return parser.parse_args()

def load_embeddings(file_path):
    """Charge les embeddings depuis un fichier NPZ"""
    print(f"Chargement des embeddings depuis {file_path}...")
    data = np.load(file_path, allow_pickle=True)
    return data

def cosine_similarity(a, b):
    """Calcule la similarité cosinus entre deux vecteurs"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def calculate_domain_similarity(cv_domain, job_domain):
    """Calcule un score de similarité entre domaines"""
    if cv_domain == job_domain:
        return 1.0
    
    # Mapping des domaines proches
    related_domains = {
        "informatique_reseaux": ["automatismes_info_industrielle"],
        "automatismes_info_industrielle": ["informatique_reseaux", "genie_industriel"],
        "finance": [],
        "civil_btp": ["genie_industriel"],
        "genie_industriel": ["automatismes_info_industrielle", "civil_btp"]
    }
    
    if job_domain in related_domains.get(cv_domain, []):
        return 0.5
    
    return 0.0

def find_best_matches(cv_data, job_data, top_n=3):
    """Trouve les meilleures correspondances entre CV et offres d'emploi"""
    cv_embeddings = cv_data['embeddings']
    job_embeddings = job_data['embeddings']
    cv_texts = cv_data['texts']
    job_texts = job_data['description'] if 'description' in job_data else job_data['texts']
    cv_classifications = cv_data['classifications']
    job_classifications = job_data['classifications']
    
    matches = []
    
    for i, (cv_embedding, cv_text, cv_domain) in enumerate(zip(cv_embeddings, cv_texts, cv_classifications)):
        job_scores = []
        
        for j, (job_embedding, job_text, job_domain) in enumerate(zip(job_embeddings, job_texts, job_classifications)):
            # Calculer la similarité cosinus entre les embeddings
            embedding_similarity = cosine_similarity(cv_embedding, job_embedding)
            
            # Calculer la similarité de domaine
            domain_similarity = calculate_domain_similarity(cv_domain, job_domain)
            
            # Score combiné (70% similarité d'embedding, 30% similarité de domaine)
            combined_score = 0.7 * embedding_similarity + 0.3 * domain_similarity
            
            job_scores.append({
                'job_id': j,
                'job_text': job_text,
                'job_domain': job_domain,
                'similarity': combined_score
            })
        
        # Trier les offres par score de similarité décroissant
        job_scores.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Conserver les top_n meilleures correspondances
        cv_matches = job_scores[:top_n]
        
        matches.append({
            'cv_id': i,
            'cv_text': cv_text,
            'cv_domain': cv_domain,
            'matches': cv_matches
        })
    
    return matches

def display_matches(matches):
    """Affiche les correspondances de manière formatée"""
    print("\nRésultats des correspondances CV-Offres d'emploi:")
    
    for match in matches:
        cv_id = match['cv_id']
        cv_text = match['cv_text']
        cv_domain = match['cv_domain']
        
        # Limiter la longueur du texte du CV
        short_cv = cv_text[:100] + "..." if len(cv_text) > 100 else cv_text
        
        print(f"\n{'-'*80}")
        print(f"CV #{cv_id+1} ({cv_domain}):")
        print(f"{short_cv}")
        print(f"{'-'*80}")
        
        # Construire un tableau pour les correspondances
        table_data = []
        for i, job_match in enumerate(match['matches']):
            job_id = job_match['job_id']
            job_text = job_match['job_text']
            job_domain = job_match['job_domain']
            similarity = job_match['similarity']
            
            # Limiter la longueur du texte de l'offre
            short_job = job_text[:100] + "..." if len(job_text) > 100 else job_text
            
            table_data.append([
                f"#{i+1}",
                f"Offre #{job_id+1}",
                job_domain,
                f"{similarity:.2f}",
                short_job
            ])
        
        # Afficher le tableau
        headers = ["Rang", "ID", "Domaine", "Score", "Description de l'offre"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

def main():
    args = parse_args()
    
    # Charger les embeddings
    cv_data = load_embeddings(args.cv_embeddings)
    job_data = load_embeddings(args.job_embeddings)
    
    # Trouver les meilleures correspondances
    matches = find_best_matches(cv_data, job_data, args.top_n)
    
    # Afficher les correspondances
    display_matches(matches)

if __name__ == "__main__":
    main()

