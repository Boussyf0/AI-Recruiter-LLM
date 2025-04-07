#!/usr/bin/env python
# Script pour générer un dataset de fine-tuning à partir d'exemples spécifiques aux domaines

import os
import json
import pandas as pd
import argparse
from typing import List, Dict, Any

# Définir les domaines spécifiques
DOMAINS = [
    "informatique_reseaux",
    "automatismes_info_industrielle",
    "finance",
    "civil_btp",
    "genie_industriel"
]

def parse_args():
    parser = argparse.ArgumentParser(description="Créer un dataset d'entraînement pour le LLM de recrutement")
    parser.add_argument("--examples", type=str, default="data/training_examples.json",
                        help="Fichier d'exemples sources")
    parser.add_argument("--output", type=str, default="data/training_data.csv",
                        help="Fichier CSV de sortie")
    parser.add_argument("--expand", type=int, default=5,
                        help="Nombre d'exemples à générer par domaine (pour enrichir le dataset)")
    parser.add_argument("--domains", nargs='+', choices=DOMAINS, default=DOMAINS,
                        help="Domaines spécifiques à inclure dans le dataset")
    
    return parser.parse_args()

def load_examples(file_path: str) -> List[Dict[str, Any]]:
    """Charge les exemples depuis le fichier JSON"""
    if not os.path.exists(file_path):
        print(f"Le fichier d'exemples {file_path} n'existe pas!")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        examples = json.load(f)
    
    return examples

def filter_examples_by_domains(examples: List[Dict[str, Any]], domains: List[str]) -> List[Dict[str, Any]]:
    """Filtre les exemples par domaines spécifiques"""
    return [ex for ex in examples if ex.get('domain') in domains]

def expand_examples(examples: List[Dict[str, Any]], n_per_domain: int) -> List[Dict[str, Any]]:
    """
    Crée des variations des exemples pour augmenter la taille du dataset
    
    Note: Dans une situation réelle, on pourrait utiliser GPT-4 ou un autre LLM
    pour générer des variations plus intelligentes. Ici on fait juste des modifications
    simples pour simuler l'expansion de dataset.
    """
    expanded = []
    examples_by_domain = {}
    
    # Regrouper les exemples par domaine
    for ex in examples:
        domain = ex.get('domain')
        if domain not in examples_by_domain:
            examples_by_domain[domain] = []
        examples_by_domain[domain].append(ex)
    
    # Pour chaque domaine, créer des variations
    for domain, domain_examples in examples_by_domain.items():
        # Ajouter les exemples originaux
        expanded.extend(domain_examples)
        
        # Nombre d'exemples à ajouter
        n_to_add = max(0, n_per_domain - len(domain_examples))
        
        if n_to_add > 0:
            # Créer des variations basées sur les exemples existants
            for i in range(n_to_add):
                # Prendre un exemple existant comme base
                base_example = domain_examples[i % len(domain_examples)]
                
                # Créer une variante
                variant = {
                    "instruction": f"Pour le domaine {domain.replace('_', ' ')}: {base_example['instruction']}",
                    "input": base_example['input'],
                    "output": base_example['output'],
                    "domain": domain
                }
                
                expanded.append(variant)
    
    return expanded

def examples_to_dataframe(examples: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convertit les exemples en DataFrame"""
    return pd.DataFrame(examples)

def save_dataset(df: pd.DataFrame, output_path: str):
    """Sauvegarde le DataFrame au format CSV"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset sauvegardé dans {output_path}")
    print(f"Nombre total d'exemples: {len(df)}")
    
    # Afficher des statistiques par domaine
    domain_counts = df['domain'].value_counts()
    print("\nDistribution par domaine:")
    for domain, count in domain_counts.items():
        print(f"- {domain}: {count} exemples")

def main():
    args = parse_args()
    
    # Charger les exemples
    examples = load_examples(args.examples)
    if not examples:
        print("Aucun exemple trouvé. Création d'exemples de test...")
        examples = [
            {
                "instruction": "Évalue ce CV pour un poste d'ingénieur DevOps.",
                "input": "John Doe, 5 ans d'expérience en DevOps...",
                "output": "Ce candidat présente une bonne expérience...",
                "domain": "informatique_reseaux"
            }
        ]
    
    # Filtrer par domaines
    filtered_examples = filter_examples_by_domains(examples, args.domains)
    print(f"Nombre d'exemples filtrés: {len(filtered_examples)}")
    
    # Expandre les exemples
    expanded_examples = expand_examples(filtered_examples, args.expand)
    print(f"Nombre d'exemples après expansion: {len(expanded_examples)}")
    
    # Convertir en DataFrame
    df = examples_to_dataframe(expanded_examples)
    
    # Sauvegarder
    save_dataset(df, args.output)

if __name__ == "__main__":
    main() 