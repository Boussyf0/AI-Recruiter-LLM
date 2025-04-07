#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
evaluate_matching.py - Évaluation systématique du système de matching CV-offres

Ce script permet d'évaluer les performances du système de matching en comparant 
les résultats automatiques avec des annotations manuelles sur un ensemble de CV de test.
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from improve_matching import improved_cv_matching, fallback_cv_analysis
from tabulate import tabulate

# Configurations par défaut
DEFAULT_TEST_DIR = "data/evaluation/cv_test"
DEFAULT_ANNOTATIONS_FILE = "data/evaluation/annotations.json"
DEFAULT_RESULTS_DIR = "data/evaluation/results"
DEFAULT_API_URL = "http://localhost:8000/api/v1"
DEFAULT_LLM_URL = "http://localhost:8001"
DEFAULT_DOMAINS = [
    "informatique_reseaux",
    "automatismes_info_industrielle",
    "finance",
    "genie_civil_btp",
    "genie_industriel"
]

def parse_arguments():
    """Parse les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(description="Évaluation systématique du matching CV-offres")
    
    parser.add_argument("--test-dir", default=DEFAULT_TEST_DIR, 
                        help=f"Répertoire contenant les CV de test (PDF ou TXT) (défaut: {DEFAULT_TEST_DIR})")
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS_FILE, 
                        help=f"Fichier d'annotations manuelles (défaut: {DEFAULT_ANNOTATIONS_FILE})")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR, 
                        help=f"Répertoire pour sauvegarder les résultats (défaut: {DEFAULT_RESULTS_DIR})")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, 
                        help=f"URL de l'API backend (défaut: {DEFAULT_API_URL})")
    parser.add_argument("--llm-url", default=DEFAULT_LLM_URL, 
                        help=f"URL du service LLM (défaut: {DEFAULT_LLM_URL})")
    parser.add_argument("--use-fallback", action="store_true", 
                        help="Utiliser l'analyse fallback même si l'API est disponible")
    parser.add_argument("--verbose", action="store_true", 
                        help="Afficher des informations détaillées pendant l'évaluation")
    
    return parser.parse_args()

def load_annotations(annotations_file):
    """Charge les annotations manuelles depuis un fichier JSON"""
    if not os.path.exists(annotations_file):
        print(f"Fichier d'annotations introuvable: {annotations_file}")
        return {}
    
    try:
        with open(annotations_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Erreur lors du chargement des annotations: {str(e)}")
        return {}

def create_example_annotations(test_dir, output_file):
    """
    Crée un fichier d'annotations d'exemple basé sur les CV de test disponibles
    pour faciliter l'annotation manuelle
    """
    cv_files = []
    for ext in ['.pdf', '.txt']:
        cv_files.extend(list(Path(test_dir).glob(f'*{ext}')))
    
    # Créer une structure d'annotations vide
    annotations = {}
    for cv_file in cv_files:
        cv_id = cv_file.stem
        annotations[cv_id] = {
            "cv_file": str(cv_file),
            "main_domain": "",  # À remplir manuellement
            "skills": [],  # À remplir manuellement
            "expected_job_titles": [],  # À remplir manuellement
            "notes": ""  # Notes supplémentaires
        }
    
    # Sauvegarder le fichier d'exemple
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=4)
    
    print(f"Fichier d'annotations d'exemple créé: {output_file}")
    print("Veuillez remplir les informations manquantes (domaine, compétences, etc.)")

def evaluate_domain_detection(results, annotations):
    """Évalue la précision de la détection de domaine"""
    true_domains = []
    predicted_domains = []
    
    for cv_id, result in results.items():
        if cv_id in annotations and 'main_domain' in annotations[cv_id]:
            cv_analysis = result.get('cv_analysis', {})
            true_domain = annotations[cv_id]['main_domain']
            predicted_domain = cv_analysis.get('main_domain', '')
            
            if true_domain and predicted_domain:
                true_domains.append(true_domain)
                predicted_domains.append(predicted_domain)
    
    if not true_domains:
        return {
            'accuracy': 0.0,
            'confusion_matrix': None,
            'samples': 0
        }
    
    accuracy = accuracy_score(true_domains, predicted_domains)
    conf_matrix = confusion_matrix(true_domains, predicted_domains, labels=DEFAULT_DOMAINS)
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'samples': len(true_domains)
    }

def evaluate_skills_extraction(results, annotations):
    """Évalue la précision et le rappel de l'extraction des compétences"""
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for cv_id, result in results.items():
        if cv_id in annotations and 'skills' in annotations[cv_id]:
            cv_analysis = result.get('cv_analysis', {})
            true_skills = set([s.lower() for s in annotations[cv_id]['skills']])
            predicted_skills = set([s.lower() for s in cv_analysis.get('skills', [])])
            
            if true_skills and predicted_skills:
                # Calculer la précision (skills correctes / skills prédites)
                precision = len(true_skills & predicted_skills) / len(predicted_skills) if predicted_skills else 0
                
                # Calculer le rappel (skills correctes / toutes les skills attendues)
                recall = len(true_skills & predicted_skills) / len(true_skills) if true_skills else 0
                
                # Calculer le F1-score (moyenne harmonique de précision et rappel)
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)
    
    if not precision_scores:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'samples': 0
        }
    
    return {
        'precision': np.mean(precision_scores),
        'recall': np.mean(recall_scores),
        'f1': np.mean(f1_scores),
        'samples': len(precision_scores)
    }

def evaluate_matching_quality(results, annotations):
    """
    Évalue la qualité du matching avec les offres
    en vérifiant si les offres sont du bon domaine et contiennent les compétences attendues
    """
    domain_match_scores = []
    skills_match_scores = []
    
    for cv_id, result in results.items():
        if cv_id in annotations:
            true_domain = annotations[cv_id].get('main_domain', '')
            true_skills = set([s.lower() for s in annotations[cv_id].get('skills', [])])
            
            if not true_domain or not true_skills:
                continue
            
            job_results = result.get('results', [])
            
            for job in job_results:
                # Vérifier que le domaine de l'offre correspond au domaine attendu
                job_domain = job.get('domain', '')
                domain_match = 1.0 if job_domain == true_domain else 0.0
                domain_match_scores.append(domain_match)
                
                # Vérifier combien de compétences attendues sont mentionnées dans l'offre
                job_text = (job.get('title', '') + ' ' + job.get('description', '')).lower()
                matched_skills = sum(1 for skill in true_skills if skill.lower() in job_text)
                skill_match_score = matched_skills / len(true_skills) if true_skills else 0
                skills_match_scores.append(skill_match_score)
    
    if not domain_match_scores:
        return {
            'domain_match_rate': 0.0,
            'skills_match_rate': 0.0,
            'samples': 0
        }
    
    return {
        'domain_match_rate': np.mean(domain_match_scores),
        'skills_match_rate': np.mean(skills_match_scores),
        'samples': len(domain_match_scores)
    }

def run_matching_on_test_data(test_dir, annotations, api_url, llm_url, use_fallback, verbose=False):
    """Exécute l'algorithme de matching sur tous les CV de test"""
    results = {}
    
    # Récupérer tous les fichiers CV (PDF et TXT)
    cv_files = []
    for ext in ['.pdf', '.txt']:
        cv_files.extend(list(Path(test_dir).glob(f'*{ext}')))
    
    print(f"Évaluation de {len(cv_files)} CV de test...")
    
    for cv_file in tqdm(cv_files):
        cv_id = cv_file.stem
        
        try:
            # Obtenir le texte du CV si c'est un fichier TXT
            cv_text = None
            if cv_file.suffix.lower() == '.txt':
                with open(cv_file, 'r', encoding='utf-8', errors='ignore') as f:
                    cv_text = f.read()
            
            # Exécuter le matching
            if use_fallback and cv_text:
                # Utiliser directement l'analyse fallback
                cv_analysis = fallback_cv_analysis(cv_text)
                result = {
                    'cv_analysis': cv_analysis,
                    'results': [],  # Pas de matching d'offres en mode fallback direct
                    'timestamp': datetime.now().isoformat(),
                    'mode': 'fallback_direct'
                }
            else:
                # Utiliser le matching complet
                result = improved_cv_matching(
                    cv_path=str(cv_file) if not cv_text else None,
                    cv_text=cv_text,
                    api_url=api_url,
                    llm_url=llm_url,
                    top_k=5,
                    min_score=0.3,
                    strict_threshold=False,
                    verbose=verbose
                )
            
            results[cv_id] = result
            
            if verbose:
                cv_analysis = result.get('cv_analysis', {})
                print(f"\n--- CV: {cv_file.name} ---")
                print(f"Domaine détecté: {cv_analysis.get('main_domain', 'N/A')}")
                print(f"Compétences: {', '.join(cv_analysis.get('skills', []))}")
                print(f"Nombre d'offres matchées: {len(result.get('results', []))}")
        
        except Exception as e:
            print(f"Erreur lors du traitement du CV {cv_file.name}: {str(e)}")
            results[cv_id] = {'error': str(e)}
    
    return results

def generate_evaluation_report(metrics, annotations, results_dir):
    """Génère un rapport d'évaluation détaillé"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(results_dir, f"evaluation_report_{timestamp}.txt")
    json_results_file = os.path.join(results_dir, f"evaluation_results_{timestamp}.json")
    
    # Convertir les tableaux numpy en listes pour la sérialisation JSON
    metrics_json = {}
    for key, value in metrics.items():
        if key == 'domain_detection':
            metrics_json[key] = {
                'accuracy': float(value['accuracy']),
                'samples': int(value['samples'])
            }
            # Convertir la matrice de confusion en liste si elle existe
            if value['confusion_matrix'] is not None:
                metrics_json[key]['confusion_matrix'] = value['confusion_matrix'].tolist()
            else:
                metrics_json[key]['confusion_matrix'] = None
        elif key == 'skills_extraction' or key == 'matching_quality':
            # Convertir tous les flottants numpy en flottants Python
            metrics_json[key] = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                                for k, v in value.items()}
        else:
            metrics_json[key] = value
    
    # Préparer les données pour le rapport JSON
    json_results = {
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics_json,
        'annotations': annotations
    }
    
    # Sauvegarder les résultats JSON
    os.makedirs(results_dir, exist_ok=True)
    with open(json_results_file, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, ensure_ascii=False, indent=4)
    
    # Créer le rapport texte
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=================================================\n")
        f.write("  RAPPORT D'ÉVALUATION DU SYSTÈME DE MATCHING CV\n")
        f.write(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=================================================\n\n")
        
        # Métriques de détection de domaine
        f.write("1. DÉTECTION DU DOMAINE PROFESSIONNEL\n")
        f.write("------------------------------------\n")
        domain_metrics = metrics['domain_detection']
        f.write(f"Précision: {domain_metrics['accuracy']:.2%}\n")
        f.write(f"Échantillons évalués: {domain_metrics['samples']}\n\n")
        
        if domain_metrics['confusion_matrix'] is not None:
            f.write("Matrice de confusion:\n")
            f.write(tabulate(
                domain_metrics['confusion_matrix'],
                headers=DEFAULT_DOMAINS,
                showindex=DEFAULT_DOMAINS,
                tablefmt="grid"
            ))
            f.write("\n\n")
        
        # Métriques d'extraction de compétences
        f.write("2. EXTRACTION DES COMPÉTENCES\n")
        f.write("----------------------------\n")
        skills_metrics = metrics['skills_extraction']
        f.write(f"Précision: {skills_metrics['precision']:.2%}\n")
        f.write(f"Rappel: {skills_metrics['recall']:.2%}\n")
        f.write(f"F1-Score: {skills_metrics['f1']:.2%}\n")
        f.write(f"Échantillons évalués: {skills_metrics['samples']}\n\n")
        
        # Métriques de qualité du matching
        f.write("3. QUALITÉ DU MATCHING\n")
        f.write("---------------------\n")
        matching_metrics = metrics['matching_quality']
        f.write(f"Taux de correspondance de domaine: {matching_metrics['domain_match_rate']:.2%}\n")
        f.write(f"Taux de correspondance de compétences: {matching_metrics['skills_match_rate']:.2%}\n")
        f.write(f"Offres évaluées: {matching_metrics['samples']}\n\n")
        
        # Exemples de prédictions
        f.write("4. SOMMAIRE DES CV TESTÉS\n")
        f.write("------------------------\n")
        
        cv_ids = list(annotations.keys())[:5]  # Limiter à 5 exemples maximum
        
        for cv_id in cv_ids:
            annotation = annotations.get(cv_id, {})
            f.write(f"CV: {cv_id}\n")
            f.write(f"  Domaine attendu: {annotation.get('main_domain', 'N/A')}\n")
            f.write(f"  Compétences attendues: {', '.join(annotation.get('skills', []))}\n")
            f.write(f"  Notes: {annotation.get('notes', '')}\n")
            f.write("\n")
        
        f.write("\n=================================================\n")
        f.write(f"Rapport complet sauvegardé dans: {json_results_file}\n")
    
    print(f"Rapport d'évaluation généré: {report_file}")
    print(f"Résultats détaillés sauvegardés: {json_results_file}")
    
    return report_file

def main():
    args = parse_arguments()
    
    # Vérifier que le répertoire de test existe
    if not os.path.exists(args.test_dir):
        print(f"Répertoire de test introuvable: {args.test_dir}")
        os.makedirs(args.test_dir, exist_ok=True)
        print(f"Répertoire créé. Veuillez y placer vos CV de test (PDF ou TXT).")
        return
    
    # Vérifier les annotations
    annotations = load_annotations(args.annotations)
    
    if not annotations:
        print("Aucune annotation trouvée. Création d'un fichier d'exemple...")
        create_example_annotations(args.test_dir, args.annotations)
        print("Veuillez compléter les annotations avant de relancer l'évaluation.")
        return
    
    # Exécuter le matching sur tous les CV de test
    print(f"Utilisation de l'API: {args.api_url}")
    print(f"Utilisation du service LLM: {args.llm_url}")
    print(f"Mode fallback forcé: {'Oui' if args.use_fallback else 'Non'}")
    
    results = run_matching_on_test_data(
        args.test_dir,
        annotations,
        args.api_url,
        args.llm_url,
        args.use_fallback,
        args.verbose
    )
    
    # Calculer les métriques d'évaluation
    print("Calcul des métriques d'évaluation...")
    
    domain_metrics = evaluate_domain_detection(results, annotations)
    skills_metrics = evaluate_skills_extraction(results, annotations)
    matching_metrics = evaluate_matching_quality(results, annotations)
    
    metrics = {
        'domain_detection': domain_metrics,
        'skills_extraction': skills_metrics,
        'matching_quality': matching_metrics
    }
    
    # Afficher un résumé des résultats
    print("\n=== RÉSUMÉ DES RÉSULTATS ===")
    print(f"Précision de détection du domaine: {domain_metrics['accuracy']:.2%}")
    print(f"Précision d'extraction des compétences: {skills_metrics['precision']:.2%}")
    print(f"Rappel d'extraction des compétences: {skills_metrics['recall']:.2%}")
    print(f"F1-Score des compétences: {skills_metrics['f1']:.2%}")
    print(f"Qualité du matching (domaine): {matching_metrics['domain_match_rate']:.2%}")
    print(f"Qualité du matching (compétences): {matching_metrics['skills_match_rate']:.2%}")
    
    # Générer un rapport détaillé
    report_file = generate_evaluation_report(metrics, annotations, args.results_dir)
    
    print(f"\nÉvaluation terminée. Rapport sauvegardé dans: {report_file}")

if __name__ == "__main__":
    main() 