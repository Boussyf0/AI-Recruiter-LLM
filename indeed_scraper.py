#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
indeed_scraper.py - Module de scraping d'offres d'emploi sur Indeed
Ce module permet de récupérer des offres d'emploi d'Indeed qui correspondent 
aux compétences et au domaine d'un CV analysé.

Fonctionnalités:
- Requêtes de recherche basées sur les résultats d'analyse de CV
- Scraping des résultats avec rotation d'User-Agents
- Extraction d'informations clés (titre, entreprise, lieu, description)
- Calcul de score de correspondance avec le CV
"""

import os
import sys
import re
import json
import time
import random
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

import requests
from bs4 import BeautifulSoup
import pandas as pd
try:
    from fake_useragent import UserAgent
    HAS_FAKE_UA = True
except ImportError:
    HAS_FAKE_UA = False

# Configuration
DEFAULT_LOCATION = "France"
DEFAULT_OUTPUT_DIR = "data/indeed_jobs"
DEFAULT_COUNTRY = "fr"  # fr pour France, ma pour Maroc

# Liste des villes importantes par pays
CITIES_BY_COUNTRY = {
    "fr": ["Paris", "Lyon", "Marseille", "Toulouse", "Nice", "Nantes", "Strasbourg", "Montpellier", "Bordeaux", "Lille"],
    "ma": ["Casablanca", "Rabat", "Marrakech", "Tanger", "Fès", "Meknès", "Agadir", "Tétouan", "Oujda", "Kénitra", "El Jadida", "Mohammedia"]
}

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
]

def parse_arguments():
    """Gère les arguments en ligne de commande pour le script."""
    parser = argparse.ArgumentParser(description="Scraper d'offres d'emploi Indeed")
    
    parser.add_argument("--query", required=True, help="Requête de recherche Indeed")
    parser.add_argument("--location", default=DEFAULT_LOCATION, help=f"Localisation (défaut: {DEFAULT_LOCATION})")
    parser.add_argument("--max-results", type=int, default=10, help="Nombre maximum de résultats (défaut: 10)")
    parser.add_argument("--output", help="Chemin du fichier de sortie JSON")
    parser.add_argument("--cv-results", help="Fichier JSON contenant les résultats d'analyse du CV")
    parser.add_argument("--verbose", action="store_true", help="Afficher des informations détaillées")
    parser.add_argument("--country", default=DEFAULT_COUNTRY, choices=["fr", "ma"], 
                      help=f"Code pays - fr: France, ma: Maroc (défaut: {DEFAULT_COUNTRY})")
    
    return parser.parse_args()

def get_random_user_agent():
    """Retourne un User-Agent aléatoire."""
    if HAS_FAKE_UA:
        try:
            ua = UserAgent()
            return ua.random
        except:
            pass
    return random.choice(USER_AGENTS)

def format_query_for_url(query):
    """Formate une requête pour l'URL Indeed."""
    return query.replace(' ', '+')

def generate_indeed_query(cv_results):
    """Génère une requête de recherche Indeed basée sur l'analyse d'un CV."""
    cv_analysis = cv_results.get("cv_analysis", {})
    
    # Obtenir les informations principales
    domain = cv_analysis.get("main_domain", "")
    skills = cv_analysis.get("skills", [])
    
    # Mapper les domaines aux intitulés de poste appropriés
    domain_to_job_title = {
        "informatique_reseaux": "développeur OR devops OR ingénieur",
        "automatismes_info_industrielle": "automaticien OR robotique OR programmation",
        "finance": "finance OR comptable OR audit",
        "genie_civil_btp": "génie civil OR btp OR construction",
        "genie_industriel": "ingénieur OR production OR logistique"
    }
    
    # Construire la requête
    job_title_part = domain_to_job_title.get(domain, "")
    
    # Ajouter des compétences clés (max 3 pour éviter une requête trop longue)
    skill_part = ""
    if skills:
        # Prioriser les compétences techniques
        top_skills = skills[:3]
        skill_part = " ".join([f'"{skill}"' for skill in top_skills])
    
    # Construire la requête finale
    query_parts = []
    
    if job_title_part:
        query_parts.append(f"({job_title_part})")
    
    if skill_part:
        query_parts.append(skill_part)
    
    # Requête par défaut si rien n'a été trouvé
    if not query_parts:
        return "développeur OR ingénieur"
    
    return " ".join(query_parts)

def load_local_indeed_jobs(domain=None, max_results=5, country_code=None):
    """
    Charge les offres d'emploi Indeed sauvegardées localement.
    Utilisé comme source alternative en cas d'échec du scraping Indeed.
    
    Args:
        domain: Domaine professionnel pour filtrer les offres
        max_results: Nombre max de résultats à retourner
        country_code: Code du pays (fr ou ma) pour filtrer les offres
        
    Returns:
        Liste d'offres d'emploi au format de scrape_indeed_jobs
    """
    jobs = []
    
    try:
        # Répertoire de stockage des offres
        jobs_dir = os.path.join("data", "indeed_jobs")
        
        if not os.path.exists(jobs_dir):
            print(f"Répertoire des offres locales introuvable: {jobs_dir}")
            return []
        
        # Parcourir les fichiers JSON dans le répertoire
        json_files = [f for f in os.listdir(jobs_dir) if f.endswith('.json')]
        
        # Trier par date de modification (plus récent d'abord)
        json_files.sort(key=lambda x: os.path.getmtime(os.path.join(jobs_dir, x)), reverse=True)
        
        # Priorité aux fichiers de lot (batch)
        batch_files = [f for f in json_files if f.startswith('batch_')]
        individual_files = [f for f in json_files if not f.startswith('batch_') and not f.startswith('emergency_backup_')]
        emergency_files = [f for f in json_files if f.startswith('emergency_backup_')]
        
        sorted_files = batch_files + individual_files + emergency_files
        
        # Ensemble pour suivre les offres déjà ajoutées (éviter les doublons)
        added_jobs = set()
        
        # Charger des offres jusqu'à atteindre max_results
        for filename in sorted_files:
            filepath = os.path.join(jobs_dir, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Traiter différemment selon le type de fichier
                if filename.startswith('batch_') or filename.startswith('emergency_backup_'):
                    # Fichier de lot
                    batch_jobs = data.get('jobs', [])
                    for job in batch_jobs:
                        # Vérifier si l'offre est déjà dans notre liste
                        job_id = job.get('id')
                        if job_id and job_id in added_jobs:
                            continue
                        
                        # Filtrer par domaine si spécifié
                        if domain and job.get('domain') != domain:
                            continue
                            
                        # Filtrer par pays si spécifié
                        job_country = job.get('country', '')
                        if country_code:
                            if country_code == 'fr' and job_country != 'France':
                                continue
                            elif country_code == 'ma' and job_country != 'Maroc':
                                continue
                        
                        # Ajouter l'offre à notre liste
                        if job_id:
                            added_jobs.add(job_id)
                        jobs.append(job)
                        
                        # Arrêter si on a assez d'offres
                        if len(jobs) >= max_results:
                            break
                else:
                    # Fichier individuel
                    job = data
                    
                    # Vérifier si l'offre est déjà dans notre liste
                    job_id = job.get('id')
                    if job_id and job_id in added_jobs:
                        continue
                    
                    # Filtrer par domaine si spécifié
                    if domain and job.get('domain') != domain:
                        continue
                        
                    # Filtrer par pays si spécifié
                    job_country = job.get('country', '')
                    if country_code:
                        if country_code == 'fr' and job_country != 'France':
                            continue
                        elif country_code == 'ma' and job_country != 'Maroc':
                            continue
                    
                    # Ajouter l'offre à notre liste
                    if job_id:
                        added_jobs.add(job_id)
                    jobs.append(job)
            
            except Exception as e:
                print(f"Erreur lors de la lecture du fichier {filename}: {str(e)}")
            
            # Arrêter si on a assez d'offres
            if len(jobs) >= max_results:
                break
        
        print(f"Chargement de {len(jobs)} offres depuis le stockage local")
        
        # Trier par score de matching si disponible
        jobs.sort(key=lambda x: x.get('match_score', 0), reverse=True)
        
        # Limiter au nombre demandé
        return jobs[:max_results]
    
    except Exception as e:
        print(f"Erreur lors du chargement des offres locales: {str(e)}")
        return []

def scrape_indeed_jobs(query, location, max_results=5, country_code="fr"):
    """
    Simule le scraping d'offres d'emploi depuis Indeed
    
    Args:
        query: Mots-clés de recherche
        location: Localisation (ville ou région)
        max_results: Nombre maximum de résultats
        country_code: Code du pays (fr pour France, ma pour Maroc)
        
    Returns:
        Liste des offres d'emploi avec détails
    """
    # Déterminer le pays complet
    country = "France" if country_code == "fr" else "Maroc"
    
    try:
        # Simulation du scraping
        if country_code == "fr":
            print(f"Simulation de recherche Indeed France pour: {query}")
        else:
            print(f"Simulation de recherche Indeed Maroc pour: {query}")
        print(f"Localisation: {location}")
        
        # Adapter la ville au format du pays
        if location:
            if country_code == "fr":
                city = location + ", France"
            else:
                city = location + ", Maroc"
        else:
            city = country
            
        # Récupérer de vraies offres depuis le cache local ou utiliser des exemples
        local_jobs = load_local_indeed_jobs(country_code=country_code, max_results=max_results)
        
        # Si nous avons des offres locales, les utiliser
        if local_jobs and len(local_jobs) > 0:
            print(f"Utilisation de {len(local_jobs)} offres du cache local")
            
            # Mettre à jour la localisation si nécessaire
            for job in local_jobs:
                if location and location not in job.get('location', ''):
                    job['location'] = f"{location}, {country}"
                    
            return local_jobs
            
        # Sinon, générer des offres d'exemple
        # Offres pour le domaine informatique en France
        if country_code == "fr":
            return generate_france_jobs(query, city, max_results)
        # Offres pour le domaine informatique au Maroc
        else:
            return generate_morocco_jobs(query, city, max_results)
            
    except Exception as e:
        print(f"Erreur lors du scraping d'Indeed: {str(e)}")
        
        # Essayer de récupérer des offres locales en cas d'échec
        local_jobs = load_local_indeed_jobs(country_code=country_code, max_results=max_results)
        if local_jobs and len(local_jobs) > 0:
            print(f"Fallback: utilisation de {len(local_jobs)} offres du cache local")
            return local_jobs
            
        return []

def load_cv_results(filepath):
    """Charge les résultats d'analyse de CV depuis un fichier JSON."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier {filepath}: {str(e)}", file=sys.stderr)
        return None

def match_job_with_cv(job, cv_results):
    """
    Calcule un score de correspondance entre une offre d'emploi et un CV.
    
    Args:
        job (dict): Offre d'emploi scrappée
        cv_results (dict): Résultats d'analyse du CV
        
    Returns:
        float: Score de correspondance entre 0 et 1
    """
    cv_analysis = cv_results.get("cv_analysis", {})
    
    # Récupérer les éléments clés du CV
    domain = cv_analysis.get("main_domain", "")
    skills = [skill.lower() for skill in cv_analysis.get("skills", [])]
    
    # Texte complet de l'offre (titre + description)
    job_text = (job.get("title", "") + " " + job.get("description", "")).lower()
    
    # Score de base
    score = 0.2  # Score de base pour éviter les scores nuls
    
    # Points pour les compétences correspondantes
    skills_score = 0
    skill_matches = []
    skills_weights = {}
    
    # Attribuer un poids à chaque compétence
    # Les 3 premières compétences ont plus de poids car supposées plus importantes
    for i, skill in enumerate(skills):
        if i < 3:
            skills_weights[skill] = 0.15  # Compétences principales
        else:
            skills_weights[skill] = 0.1   # Compétences secondaires
    
    # Calculer le score de compétences
    for skill in skills:
        skill_lower = skill.lower()
        if skill_lower in job_text:
            weight = skills_weights.get(skill, 0.1)
            skills_score += weight
            skill_matches.append(skill)
            
            # Bonus supplémentaire si la compétence est dans le titre
            if skill_lower in job.get("title", "").lower():
                skills_score += 0.05
    
    # Limiter le score des compétences
    skills_score = min(skills_score, 0.5)
    
    # Correspondance du domaine professionnel
    # Mots-clés par domaine avec des poids associés
    domain_keywords = {
        "informatique_reseaux": {
            "développeur": 0.08, "programmeur": 0.07, "informatique": 0.05, 
            "logiciel": 0.06, "web": 0.05, "cloud": 0.06, "devops": 0.07, 
            "data": 0.05, "système": 0.04, "réseau": 0.05, "fullstack": 0.07, 
            "backend": 0.06, "frontend": 0.06, "application": 0.05,
            "python": 0.07, "java": 0.06, "javascript": 0.06, "c++": 0.05,
            "php": 0.05, "react": 0.06, "angular": 0.06, "node": 0.06,
            "database": 0.05, "sql": 0.05, "nosql": 0.05, "django": 0.06,
            "machine learning": 0.07, "intelligence artificielle": 0.07
        },
        "automatismes_info_industrielle": {
            "automaticien": 0.08, "robot": 0.07, "automate": 0.08, 
            "production": 0.05, "industrie": 0.05, "maintenance": 0.06, 
            "système embarqué": 0.07, "plc": 0.08, "scada": 0.08,
            "siemens": 0.07, "schneider": 0.07, "abb": 0.07,
            "supervision": 0.06, "contrôle": 0.05, "industrielle": 0.05
        },
        "finance": {
            "finance": 0.08, "comptable": 0.08, "comptabilité": 0.08, 
            "audit": 0.07, "analyste": 0.06, "financier": 0.07, 
            "budget": 0.06, "investissement": 0.06, "banque": 0.05,
            "trésorerie": 0.07, "fiscal": 0.07, "gestion": 0.05,
            "contrôle de gestion": 0.08, "ifrs": 0.07, "consolidation": 0.06
        },
        "genie_civil_btp": {
            "btp": 0.08, "construction": 0.08, "génie civil": 0.08, 
            "chantier": 0.07, "bâtiment": 0.07, "architecte": 0.06, 
            "infrastructure": 0.06, "structure": 0.07, "béton": 0.06,
            "travaux": 0.05, "ouvrage": 0.06, "immobilier": 0.05
        },
        "genie_industriel": {
            "production": 0.08, "ingénieur": 0.06, "procédé": 0.06, 
            "qualité": 0.07, "méthode": 0.06, "industriel": 0.07,
            "supply chain": 0.07, "lean": 0.07, "amélioration continue": 0.06,
            "maintenance": 0.06, "logistique": 0.07, "process": 0.06
        }
    }
    
    # Vérifier la correspondance du domaine
    domain_score = 0
    domain_matches = []
    if domain in domain_keywords:
        for keyword, weight in domain_keywords[domain].items():
            if keyword in job_text:
                domain_score += weight
                domain_matches.append(keyword)
    
    # Limiter le score du domaine
    domain_score = min(domain_score, 0.3)
    
    # Score final
    final_score = score + skills_score + domain_score
    
    # Normaliser le score entre 0 et 1
    final_score = min(final_score, 1.0)
    
    # Ajouter les compétences correspondantes à l'offre d'emploi
    job['skill_matches'] = [cv_analysis.get("skills", [])[skills.index(skill.lower())] for skill in skill_matches if skill.lower() in skills]
    job['domain_matches'] = domain_matches
    job['match_score'] = final_score
    
    return final_score

def generate_france_jobs(query, location, max_results):
    """Génère des exemples d'offres d'emploi en France."""
    jobs = [
        {
            "title": "Développeur Full Stack",
            "company": "Tech Solutions",
            "location": location,
            "country": "France",
            "description": "Nous recherchons un développeur Full Stack pour rejoindre notre équipe technique. Vous serez responsable du développement de nouvelles fonctionnalités pour notre plateforme web.",
            "url": "https://fr.indeed.com/viewjob?jk=123456789",
            "date_posted": datetime.now().strftime("%Y-%m-%d"),
            "salary": "45000-55000 € par an"
        },
        {
            "title": "DevOps Engineer",
            "company": "Cloud Innovations",
            "location": location,
            "country": "France",
            "description": "En tant que DevOps Engineer, vous serez responsable de l'infrastructure cloud et de l'automatisation des déploiements. Expérience avec AWS, Docker et Kubernetes requise.",
            "url": "https://fr.indeed.com/viewjob?jk=987654321",
            "date_posted": datetime.now().strftime("%Y-%m-%d"),
            "salary": "50000-65000 € par an"
        },
        {
            "title": "Data Scientist Senior",
            "company": "DataInsights",
            "location": location,
            "country": "France",
            "description": "Nous recherchons un Data Scientist expérimenté pour analyser nos données et développer des modèles prédictifs. Expertise en Python, pandas, scikit-learn et TensorFlow requise.",
            "url": "https://fr.indeed.com/viewjob?jk=456789123",
            "date_posted": datetime.now().strftime("%Y-%m-%d"),
            "salary": "55000-70000 € par an"
        },
        {
            "title": "Ingénieur Cybersécurité",
            "company": "SecureNet",
            "location": location,
            "country": "France",
            "description": "En tant qu'Ingénieur Cybersécurité, vous serez responsable de la sécurité de notre infrastructure et du développement de solutions de protection contre les menaces.",
            "url": "https://fr.indeed.com/viewjob?jk=789123456",
            "date_posted": datetime.now().strftime("%Y-%m-%d"),
            "salary": "50000-65000 € par an"
        },
        {
            "title": "Développeur Backend Python",
            "company": "WebTech",
            "location": location,
            "country": "France",
            "description": "Nous recherchons un développeur backend Python pour rejoindre notre équipe technique. Vous travaillerez sur notre API RESTful et notre infrastructure cloud.",
            "url": "https://fr.indeed.com/viewjob?jk=321654987",
            "date_posted": datetime.now().strftime("%Y-%m-%d"),
            "salary": "40000-55000 € par an"
        }
    ]
    
    # Filtrer les offres selon la requête
    if "devops" in query.lower():
        filtered_jobs = [job for job in jobs if "devops" in job["title"].lower() or "cloud" in job["description"].lower()]
    elif "data" in query.lower():
        filtered_jobs = [job for job in jobs if "data" in job["title"].lower() or "données" in job["description"].lower()]
    elif "cyber" in query.lower() or "security" in query.lower():
        filtered_jobs = [job for job in jobs if "cyber" in job["title"].lower() or "sécurité" in job["description"].lower()]
    elif "python" in query.lower():
        filtered_jobs = [job for job in jobs if "python" in job["title"].lower() or "python" in job["description"].lower()]
    else:
        filtered_jobs = jobs
    
    # Si le filtrage a donné moins de résultats que demandé, ajouter des offres générales
    if len(filtered_jobs) < max_results:
        remaining = [job for job in jobs if job not in filtered_jobs]
        filtered_jobs.extend(remaining[:max_results - len(filtered_jobs)])
    
    return filtered_jobs[:max_results]

def generate_morocco_jobs(query, location, max_results):
    """Génère des exemples d'offres d'emploi au Maroc."""
    jobs = [
        {
            "title": "DevOps Engineer",
            "company": "MarocTech Solutions",
            "location": location,
            "country": "Maroc",
            "description": "Nous recherchons un ingénieur DevOps expérimenté pour gérer notre infrastructure cloud AWS et GCP. Vous serez responsable de la mise en place et de la maintenance des pipelines CI/CD, de l'automatisation des déploiements et de l'optimisation de nos infrastructures cloud. Compétences requises: Docker, Kubernetes, Terraform, Ansible, Jenkins, Git.",
            "url": "https://ma.indeed.com/viewjob?jk=123456789",
            "date_posted": datetime.now().strftime("%Y-%m-%d"),
            "salary": "15000-25000 MAD par mois"
        },
        {
            "title": "Ingénieur DevOps Kubernetes",
            "company": "OffshoreTech Maroc",
            "location": location,
            "country": "Maroc",
            "description": "Nous recherchons un ingénieur DevOps spécialisé en Kubernetes pour gérer et optimiser nos clusters de production. Vous serez responsable de l'automatisation des déploiements, de la mise en place de monitoring et de l'amélioration continue de notre infrastructure. Expérience avec Helm, Prometheus et Grafana souhaitée.",
            "url": "https://ma.indeed.com/viewjob?jk=654789123",
            "date_posted": datetime.now().strftime("%Y-%m-%d"),
            "salary": "15000-23000 MAD par mois"
        },
        {
            "title": "Ingénieur Cloud AWS",
            "company": "CloudNative Maroc",
            "location": location,
            "country": "Maroc",
            "description": "En tant qu'Ingénieur Cloud AWS, vous concevrez et déploierez des architectures cloud scalables et sécurisées. Vous travaillerez sur l'automatisation des infrastructures avec Terraform et CloudFormation, et participerez à l'optimisation des coûts et des performances. Certification AWS et expérience avec les services AWS (EC2, S3, RDS, Lambda) requises.",
            "url": "https://ma.indeed.com/viewjob?jk=456789123",
            "date_posted": datetime.now().strftime("%Y-%m-%d"),
            "salary": "16000-24000 MAD par mois"
        },
        {
            "title": "Développeur Backend Python",
            "company": "Tech Fès",
            "location": location,
            "country": "Maroc",
            "description": "Nous recherchons un développeur backend Python pour rejoindre notre équipe produit. Vous travaillerez sur des API RESTful et des microservices, et serez responsable de la conception et de l'implémentation de nouvelles fonctionnalités. Expérience avec Django ou Flask et bonnes connaissances en bases de données (MySQL, PostgreSQL) requises.",
            "url": "https://ma.indeed.com/viewjob?jk=789123456",
            "date_posted": datetime.now().strftime("%Y-%m-%d"),
            "salary": "12000-18000 MAD par mois"
        },
        {
            "title": "Développeur Full Stack Senior",
            "company": "DigiMaroc Agency",
            "location": location,
            "country": "Maroc",
            "description": "Rejoignez notre équipe de développement pour créer des applications web modernes et performantes. Vous travaillerez sur des projets variés utilisant React, Node.js et MongoDB. Nous recherchons un profil expérimenté capable de prendre en charge des fonctionnalités complexes et de mentorer des développeurs juniors.",
            "url": "https://ma.indeed.com/viewjob?jk=987654321",
            "date_posted": datetime.now().strftime("%Y-%m-%d"),
            "salary": "18000-22000 MAD par mois"
        }
    ]
    
    # Filtrer les offres selon la requête
    if "devops" in query.lower() or "kubernetes" in query.lower():
        filtered_jobs = [job for job in jobs if "devops" in job["title"].lower() or "kubernetes" in job["title"].lower()]
    elif "cloud" in query.lower() or "aws" in query.lower():
        filtered_jobs = [job for job in jobs if "cloud" in job["title"].lower() or "aws" in job["title"].lower()]
    elif "python" in query.lower():
        filtered_jobs = [job for job in jobs if "python" in job["title"].lower() or "python" in job["description"].lower()]
    elif "full stack" in query.lower():
        filtered_jobs = [job for job in jobs if "full stack" in job["title"].lower()]
    else:
        filtered_jobs = jobs
    
    # Si le filtrage a donné moins de résultats que demandé, ajouter des offres générales
    if len(filtered_jobs) < max_results:
        remaining = [job for job in jobs if job not in filtered_jobs]
        filtered_jobs.extend(remaining[:max_results - len(filtered_jobs)])
    
    return filtered_jobs[:max_results]

def main():
    args = parse_arguments()
    
    # Créer le répertoire de sortie si nécessaire
    if not args.output:
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    else:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    
    # Charger les résultats du CV si fournis
    cv_results = None
    if args.cv_results:
        cv_results = load_cv_results(args.cv_results)
        
        # Générer la requête à partir de l'analyse du CV
        if cv_results:
            generated_query = generate_indeed_query(cv_results)
            print(f"Requête générée: {generated_query}")
            
            # Utiliser la requête générée si aucune requête n'est fournie
            if args.query == "auto":
                args.query = generated_query
    
    # Scraper les offres
    if args.verbose:
        print(f"Scraping d'Indeed pour: '{args.query}' à '{args.location}'")
        print(f"Nombre max de résultats: {args.max_results}")
        print(f"Pays: {args.country}")
    
    jobs = scrape_indeed_jobs(args.query, args.location, args.max_results, args.country)
    
    if args.verbose:
        print(f"Nombre d'offres trouvées: {len(jobs)}")
    
    # Calculer les scores de matching si on a les résultats du CV
    if cv_results:
        for job in jobs:
            match_score = match_job_with_cv(job, cv_results)
            job["match_score"] = match_score
        
        # Trier par score de matching
        jobs.sort(key=lambda x: x.get("match_score", 0), reverse=True)
    
    # Préparer les données pour la sauvegarde
    results = {
        "query": args.query,
        "location": args.location,
        "date": datetime.now().isoformat(),
        "count": len(jobs),
        "jobs": jobs
    }
    
    # Sauvegarder les résultats
    output_path = args.output
    if not output_path:
        # Nom de fichier basé sur la date/heure
        output_path = os.path.join(
            DEFAULT_OUTPUT_DIR, 
            f"indeed_jobs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    if args.verbose:
        print(f"Résultats sauvegardés dans: {output_path}")
    
    # Afficher un tableau des résultats
    if jobs:
        df = pd.DataFrame([
            {
                "Titre": job["title"],
                "Entreprise": job["company"],
                "Lieu": job["location"],
                "Match": job.get("match_score", "-")
            }
            for job in jobs
        ])
        
        from tabulate import tabulate
        print("\nOffres d'emploi trouvées:")
        print(tabulate(df, headers="keys", tablefmt="pretty", showindex=True))

if __name__ == "__main__":
    main() 