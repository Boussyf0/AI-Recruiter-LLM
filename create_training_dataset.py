#!/usr/bin/env python
# Script pour générer un dataset de fine-tuning à partir d'exemples spécifiques aux domaines

import os
import json
import pandas as pd
import argparse
from typing import List, Dict, Any
import subprocess
import zipfile
from pathlib import Path

# Définir les domaines spécifiques
DOMAINS = [
    "informatique_reseaux",
    "automatismes_info_industrielle",
    "finance",
    "civil_btp",
    "genie_industriel"
]

# Mappage des catégories des datasets Kaggle aux domaines
DOMAIN_MAPPING = {
    "IT": "informatique_reseaux",
    "Data Science": "informatique_reseaux",
    "Computer Engineering": "informatique_reseaux",
    "Finance": "finance",
    "Civil Engineering": "civil_btp",
    "Industrial Engineering": "genie_industriel",
    "Automation": "automatismes_info_industrielle",
    "Mechanical": "genie_industriel",
    "Accountant": "finance",
    "Business Analyst": "finance",
    "DevOps Engineer": "informatique_reseaux",
    "Network Engineer": "informatique_reseaux",
    "Software Engineer": "informatique_reseaux",
    "Database Administrator": "informatique_reseaux",
    "Web Developer": "informatique_reseaux",
    "Electrical Engineer": "automatismes_info_industrielle",
    "Mechatronics": "automatismes_info_industrielle",
    "Civil Engineer": "civil_btp",
    "Construction Manager": "civil_btp"
}

# Datasets Kaggle recommandés avec leur structure
KAGGLE_DATASETS = [
    {
        "id": "ravindrasinghrana/job-description-dataset",
        "files": ["job_descriptions.csv"],
        "columns": {
            "text": "job_description",
            "category": "category",
            "skills": "skills_required"
        }
    },
    {
        "id": "gauravduttakiit/resume-dataset",
        "files": ["Resume.csv"],
        "columns": {
            "text": "Resume",
            "category": "Category"
        }
    },
    {
        "id": "elanzaelani/job-skills-extraction-dataset",
        "files": ["job_skills.csv"],
        "columns": {
            "text": "job_description",
            "skills": "skills"
        }
    },
    {
        "id": "thedevastator/jobs-dataset-for-recommender-engines-and-skill-a",
        "files": ["job_postings.csv"],
        "columns": {
            "text": "job_description",
            "category": "role"
        }
    }
]

def parse_args():
    parser = argparse.ArgumentParser(description="Créer un dataset d'entraînement pour le LLM de recrutement")
    parser.add_argument("--sources", type=str, nargs='+', default=["data/training_examples.json"],
                        help="Fichiers d'exemples sources (JSON ou CSV)")
    parser.add_argument("--output", type=str, default="data/training_data.csv",
                        help="Fichier CSV de sortie")
    parser.add_argument("--expand", type=int, default=5,
                        help="Nombre d'exemples à générer par domaine")
    parser.add_argument("--domains", nargs='+', choices=DOMAINS, default=DOMAINS,
                        help="Domaines spécifiques à inclure dans le dataset")
    parser.add_argument("--kaggle", action="store_true",
                        help="Télécharger et utiliser les datasets Kaggle recommandés")
    parser.add_argument("--kaggle-datasets", nargs='+', default=None,
                        help="IDs spécifiques des datasets Kaggle à utiliser")
    
    return parser.parse_args()

def download_kaggle_datasets(datasets_list=None) -> List[str]:
    """Télécharge et prépare les datasets Kaggle"""
    if not os.environ.get('KAGGLE_USERNAME') or not os.environ.get('KAGGLE_KEY'):
        print("Avertissement: Variables d'environnement Kaggle non configurées")
        print("Pour utiliser Kaggle API, créez un fichier ~/.kaggle/kaggle.json avec vos identifiants")
        print("Ou définissez les variables d'environnement KAGGLE_USERNAME et KAGGLE_KEY")
        return []
    
    output_files = []
    data_dir = Path("data/kaggle")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Filtrer les datasets si une liste spécifique est fournie
    if datasets_list:
        datasets_to_download = [d for d in KAGGLE_DATASETS if d["id"] in datasets_list]
    else:
        datasets_to_download = KAGGLE_DATASETS
    
    for dataset in datasets_to_download:
        dataset_id = dataset["id"]
        dataset_dir = data_dir / dataset_id.split("/")[-1]
        
        try:
            # Vérifier si déjà téléchargé
            if dataset_dir.exists() and any(dataset_dir.glob("*")):
                print(f"Dataset {dataset_id} déjà téléchargé dans {dataset_dir}")
            else:
                print(f"Téléchargement du dataset {dataset_id}...")
                subprocess.run(["kaggle", "datasets", "download", "-d", dataset_id, "-p", str(data_dir)], 
                              check=True, capture_output=True)
                
                # Extraire si c'est un zip
                zip_path = data_dir / f"{dataset_id.split('/')[-1]}.zip"
                if zip_path.exists():
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(dataset_dir)
                    os.remove(zip_path)
            
            # Trouver les fichiers d'intérêt
            for file_pattern in dataset["files"]:
                for file_path in dataset_dir.glob(file_pattern):
                    output_files.append(str(file_path))
                    
        except Exception as e:
            print(f"Erreur lors du téléchargement du dataset {dataset_id}: {e}")
    
    return output_files

def load_examples_from_sources(sources: List[str]) -> List[Dict[str, Any]]:
    """Charge les exemples depuis plusieurs sources (JSON ou CSV)"""
    examples = []
    
    for source in sources:
        if not os.path.exists(source):
            print(f"Le fichier {source} n'existe pas, ignoré.")
            continue
        
        try:
            if source.endswith('.json'):
                with open(source, 'r', encoding='utf-8') as f:
                    examples.extend(json.load(f))
            
            elif source.endswith('.csv'):
                df = pd.read_csv(source)
                
                # Identifier le format du dataset
                dataset_format = None
                for dataset in KAGGLE_DATASETS:
                    for file_pattern in dataset["files"]:
                        if any(file_pattern in source for file_pattern in dataset["files"]):
                            dataset_format = dataset
                            break
                
                if dataset_format:
                    # Utiliser le mappage spécifique au dataset
                    columns = dataset_format["columns"]
                    text_col = columns.get("text", None)
                    category_col = columns.get("category", None)
                    skills_col = columns.get("skills", None)
                else:
                    # Format générique
                    text_col = next((col for col in df.columns if any(x in col.lower() for x in ["description", "resume", "text"])), df.columns[0])
                    category_col = next((col for col in df.columns if any(x in col.lower() for x in ["category", "role", "job_title"])), None)
                    skills_col = next((col for col in df.columns if any(x in col.lower() for x in ["skills", "competences", "requirements"])), None)
                
                for _, row in df.iterrows():
                    if text_col and text_col in row:
                        text = row[text_col]
                        if not pd.isna(text) and len(str(text).strip()) > 10:  # Ignorer textes vides ou trop courts
                            # Déterminer le domaine
                            domain = "informatique_reseaux"  # Domaine par défaut
                            if category_col and category_col in row and not pd.isna(row[category_col]):
                                category = str(row[category_col]).strip()
                                domain = DOMAIN_MAPPING.get(category, domain)
                            
                            # Déterminer les compétences
                            skills = "Non spécifié"
                            if skills_col and skills_col in row and not pd.isna(row[skills_col]):
                                skills = row[skills_col]
                            
                            # Création de l'exemple
                            is_resume = "resume" in source.lower() or "cv" in source.lower()
                            
                            if is_resume:
                                instruction = f"Analyse ce CV pour un poste dans le domaine {domain.replace('_', ' ')}."
                                output = f"Compétences identifiées : {skills}. Ce profil est adapté pour des postes en {domain.replace('_', ' ')}."
                            else:
                                instruction = f"Analyse cette offre d'emploi dans le domaine {domain.replace('_', ' ')}."
                                output = f"Compétences requises : {skills}. Cette offre est classée dans le domaine {domain.replace('_', ' ')}."
                            
                            example = {
                                "instruction": instruction,
                                "input": str(text),
                                "output": output,
                                "domain": domain
                            }
                            examples.append(example)
        except Exception as e:
            print(f"Erreur lors du chargement du fichier {source}: {e}")
    
    if not examples:
        print("Aucun exemple trouvé. Création d'exemples par défaut...")
        examples = [
            {
                "instruction": "Évalue ce CV pour un poste d'ingénieur DevOps.",
                "input": "John Doe, 5 ans d'expérience en DevOps...",
                "output": "Ce candidat présente une bonne expérience...",
                "domain": "informatique_reseaux"
            }
        ]
    
    return examples

def filter_examples_by_domains(examples: List[Dict[str, Any]], domains: List[str]) -> List[Dict[str, Any]]:
    """Filtre les exemples par domaines spécifiques"""
    return [ex for ex in examples if ex.get('domain') in domains]

def clean_examples(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Nettoie les exemples (supprime doublons et valide les champs)"""
    cleaned = []
    seen = set()
    
    for ex in examples:
        if not all(k in ex for k in ['instruction', 'input', 'output', 'domain']):
            continue
            
        # Tronquer les textes trop longs
        if len(ex['input']) > 2000:
            ex['input'] = ex['input'][:1997] + "..."
            
        ex_tuple = (ex['instruction'], ex['input'][:100])  # Comparer juste le début pour éviter trop de doublons
        if ex_tuple not in seen:
            seen.add(ex_tuple)
            cleaned.append(ex)
    
    return cleaned

def expand_examples(examples: List[Dict[str, Any]], n_per_domain: int) -> List[Dict[str, Any]]:
    """Crée des variations des exemples pour augmenter la taille du dataset"""
    expanded = []
    cv_templates = [
        "Analyse ce CV pour un poste en {domain}.",
        "Évalue les compétences de ce candidat pour {domain} :",
        "Pour un rôle en {domain}, que penses-tu de ce profil ?",
        "Examine cette candidature pour {domain}.",
        "Donne ton avis sur ce CV pour un poste en {domain}."
    ]
    
    job_templates = [
        "Analyse cette offre d'emploi en {domain}.",
        "Évalue les compétences requises pour ce poste en {domain} :",
        "Quelles sont les exigences clés de cette offre en {domain} ?",
        "Examine cette description de poste en {domain}.",
        "Résume les qualifications demandées pour ce rôle en {domain}."
    ]
    
    examples_by_domain = {domain: [] for domain in DOMAINS}
    for ex in examples:
        domain = ex.get('domain')
        if domain in examples_by_domain:
            examples_by_domain[domain].append(ex)
    
    for domain, domain_examples in examples_by_domain.items():
        expanded.extend(domain_examples)
        n_to_add = max(0, n_per_domain - len(domain_examples))
        
        if n_to_add > 0 and domain_examples:
            for i in range(n_to_add):
                base_example = domain_examples[i % len(domain_examples)]
                
                # Déterminer si c'est un CV ou une offre d'emploi
                is_cv = any(term in base_example['instruction'].lower() 
                          for term in ['cv', 'candidat', 'profil', 'candidature'])
                
                templates = cv_templates if is_cv else job_templates
                
                variant = {
                    "instruction": templates[i % len(templates)].format(domain=domain.replace('_', ' ')),
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
    
    domain_counts = df['domain'].value_counts()
    print("\nDistribution par domaine:")
    for domain, count in domain_counts.items():
        print(f"- {domain}: {count} exemples")

def main():
    args = parse_args()
    
    sources = list(args.sources)
    
    # Télécharger les datasets Kaggle si demandé
    if args.kaggle:
        kaggle_sources = download_kaggle_datasets(args.kaggle_datasets)
        sources.extend(kaggle_sources)
        print(f"Datasets Kaggle ajoutés : {len(kaggle_sources)}")
    
    # Charger les exemples depuis toutes les sources
    examples = load_examples_from_sources(sources)
    print(f"Nombre d'exemples chargés: {len(examples)}")
    
    # Filtrer par domaines
    filtered_examples = filter_examples_by_domains(examples, args.domains)
    print(f"Nombre d'exemples filtrés: {len(filtered_examples)}")
    
    # Nettoyer les exemples
    cleaned_examples = clean_examples(filtered_examples)
    print(f"Nombre d'exemples après nettoyage: {len(cleaned_examples)}")
    
    # Expandre les exemples
    expanded_examples = expand_examples(cleaned_examples, args.expand)
    print(f"Nombre d'exemples après expansion: {len(expanded_examples)}")
    
    # Convertir en DataFrame et sauvegarder
    df = examples_to_dataframe(expanded_examples)
    save_dataset(df, args.output)

if __name__ == "__main__":
    main() 