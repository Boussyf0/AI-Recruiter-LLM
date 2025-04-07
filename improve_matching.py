#!/usr/bin/env python
# Script pour améliorer le matching entre CVs et offres d'emploi

import argparse
import httpx
import asyncio
import json
import os
import sys
import mimetypes
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from tabulate import tabulate
from datetime import datetime
import warnings
import pandas as pd
import traceback
from pathlib import Path

# Import pour les PDF
import PyPDF2
from pdfminer.high_level import extract_text as pdfminer_extract_text
import subprocess

# Import pour Indeed
from indeed_scraper import scrape_indeed_jobs, generate_indeed_query, match_job_with_cv

# Importer le service d'embeddings local
try:
    from local_embeddings import LocalEmbeddingService
    LOCAL_EMBEDDINGS_AVAILABLE = True
except ImportError:
    LOCAL_EMBEDDINGS_AVAILABLE = False

# Configuration par défaut
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_LLM_URL = "https://api.deepinfra.com/v1/inference/mistralai/Mixtral-8x7B-Instruct-v0.1"
DEFAULT_OUTPUT_DIR = "data/output"
DEFAULT_JOB_OFFERS_DIR = "data/job_offers"

# Domaines professionnels et leurs domaines proches
DOMAIN_RELATIONS = {
    "informatique_reseaux": ["automatismes_info_industrielle", "genie_industriel"],
    "automatismes_info_industrielle": ["informatique_reseaux", "genie_industriel"],
    "finance": ["genie_industriel"],
    "genie_civil_btp": ["genie_industriel"],
    "genie_industriel": ["automatismes_info_industrielle", "informatique_reseaux"]
}

# Dictionnaire plus détaillé de mots-clés par domaine pour classification
DOMAIN_KEYWORDS = {
    "informatique_reseaux": [
        "développeur", "devops", "fullstack", "backend", "frontend", "cloud", "architecture", 
        "réseau", "système", "logiciel", "web", "sécurité", "software", "java", "python", 
        "javascript", "react", "angular", "node", "spring", "kubernetes", "docker", "aws", 
        "azure", "devops", "ci/cd", "api", "microservices", "machine learning", "data", 
        "database", "sql", "nosql", "git", "agile", "scrum", "php", "c++", "c#", ".net"
    ],
    "automatismes_info_industrielle": [
        "automate", "plc", "programmable", "scada", "supervision", "automaticien", "robot", 
        "automatisme", "contrôle", "régulation", "mesure", "capteur", "actionneur", "siemens", 
        "schneider", "rockwell", "abb", "dcs", "step7", "industrie 4.0", "m2m", "iot", 
        "industrial", "embarqué", "temps réel", "modbus", "profibus", "profinet", "ethernet/ip", 
        "can", "visu", "hmi", "ihm", "wincc", "wonderware", "labview", "matlab"
    ],
    "finance": [
        "finance", "comptabilité", "comptable", "audit", "fiscal", "trésorerie", "contrôle", 
        "gestion", "budgétaire", "ifrs", "consolidation", "analyste", "financier", "controller", 
        "investissement", "risques", "économiste", "bancaire", "banque", "assurance", "actif", 
        "passif", "bilan", "portfolio", "marchés", "financiers", "trading", "trésorier", 
        "analyste crédit", "fusion", "acquisition", "private equity", "fiscal", "sap", "sage"
    ],
    "genie_civil_btp": [
        "génie civil", "btp", "construction", "bâtiment", "structure", "chantier", "ouvrage", 
        "travaux", "architecte", "immobilier", "fondation", "pont", "route", "hydraulique", 
        "géotechnique", "topographie", "autocad", "revit", "bim", "béton", "armé", "coffreur", 
        "maçon", "vrd", "conducteur", "chef de chantier", "second œuvre", "gros œuvre", 
        "matériaux", "bureau d'études", "métreur", "structure", "urbain", "aménagement"
    ],
    "genie_industriel": [
        "production", "industriel", "maintenance", "qualité", "méthode", "amélioration", 
        "continue", "lean", "supply chain", "logistique", "stock", "approvisionnement", 
        "planification", "ordonnancement", "kaizen", "six sigma", "usine", "atelier", 
        "fabrication", "process", "procédé", "performance", "productivité", "efficience", 
        "fiabilité", "sécurité", "ergonomie", "environnement", "iso", "norme", "5s"
    ]
}

# Compétences spécifiques par domaine
DOMAIN_SKILLS = {
    "informatique_reseaux": [
        "programmation", "développement web", "devops", "cloud", "cybersécurité", 
        "bases de données", "réseaux", "systèmes", "machine learning", "data science"
    ],
    "automatismes_info_industrielle": [
        "automates", "PLC", "SCADA", "robotique", "IIoT", "supervision", 
        "électronique", "instrumentation", "capteurs", "actionneurs"
    ],
    "finance": [
        "comptabilité", "contrôle de gestion", "analyse financière", "audit", 
        "trésorerie", "fiscalité", "reporting", "budgétisation", "ERP financier"
    ],
    "genie_civil_btp": [
        "structure", "béton", "construction", "géotechnique", "hydraulique", 
        "thermique", "BIM", "chantier", "matériaux", "environnement"
    ],
    "genie_industriel": [
        "production", "qualité", "maintenance", "logistique", "méthodes", 
        "amélioration continue", "lean", "gestion de projet", "supply chain"
    ]
}

def parse_arguments():
    """Gère les arguments en ligne de commande."""
    parser = argparse.ArgumentParser(description="AI Recruiter - Amélioration du matching CV-offres d'emploi")
    
    # Arguments pour le CV (rendus optionnels si --reimport-local est utilisé)
    cv_group = parser.add_mutually_exclusive_group()
    cv_group.add_argument("--cv", help="Chemin vers le fichier CV (PDF ou TXT)")
    cv_group.add_argument("--text", help="Texte du CV")
    
    # Arguments pour la recherche
    parser.add_argument("--top-k", type=int, default=3, help="Nombre d'offres à retourner (défaut: 3)")
    parser.add_argument("--min-score", type=float, default=0.0, help="Score minimum pour les offres (défaut: 0.0)")
    parser.add_argument("--strict", action="store_true", help="Utiliser un seuil strict pour le matching de domaine")
    
    # Arguments pour les APIs
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help=f"URL de l'API backend (défaut: {DEFAULT_API_URL})")
    parser.add_argument("--llm-url", default=DEFAULT_LLM_URL, help=f"URL du service LLM (défaut: {DEFAULT_LLM_URL})")
    
    # Verbosité
    parser.add_argument("--verbose", "-v", action="store_true", help="Afficher des informations détaillées")
    
    # Indeed
    parser.add_argument("--indeed", action="store_true", help="Rechercher des offres sur Indeed")
    parser.add_argument("--indeed-location", default="France", help="Localisation Indeed (défaut: France)")
    parser.add_argument("--indeed-max-results", type=int, default=5, help="Nombre max de résultats Indeed (défaut: 5)")
    parser.add_argument("--indeed-country", choices=["fr", "ma"], default="fr", help="Pays Indeed (fr ou ma, défaut: fr)")
    parser.add_argument("--store-indeed", action="store_true", help="Stocker les offres Indeed dans la base vectorielle")
    
    # Réimportation des offres locales
    parser.add_argument("--reimport-local", action="store_true", help="Réimporter les offres locales vers la base vectorielle")
    parser.add_argument("--reimport-max", type=int, default=0, help="Nombre max d'offres à réimporter (0 = toutes)")
    
    # Analyser les arguments
    args = parser.parse_args()
    
    # Vérifier si les arguments CV sont requis
    if not args.reimport_local and not args.cv and not args.text:
        parser.error("Un des arguments --cv ou --text est requis sauf si --reimport-local est utilisé.")
    
    return args

def get_embeddings(text, domain=None, api_url=DEFAULT_API_URL):
    """Get embeddings from API or local fallback"""
    # Variable statique pour stocker l'instance du service
    if not hasattr(get_embeddings, "_embedding_service"):
        get_embeddings._embedding_service = None
    
    # Essayer d'abord l'API distante
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                f"{api_url}/get-embeddings",
                json={"text": text, "domain": domain}
            )
            if response.status_code == 200:
                return response.json()["embeddings"]
    except Exception as e:
        print(f"Service d'embeddings distant non disponible: {str(e)}")
        
    # Fallback vers le service local si disponible
    if LOCAL_EMBEDDINGS_AVAILABLE:
        try:
            print("Utilisation du service d'embeddings local...")
            # Réutiliser l'instance existante si disponible
            if get_embeddings._embedding_service is None:
                print("Initialisation du service d'embeddings local (première utilisation)...")
                get_embeddings._embedding_service = LocalEmbeddingService(
                    model_name="paraphrase-multilingual-MiniLM-L12-v2",
                    cache_dir="models/embeddings"
                )
            
            return get_embeddings._embedding_service.get_embeddings(text, domain=domain)
        except Exception as e:
            print(f"Erreur avec le service d'embeddings local: {str(e)}")
            # En cas d'erreur, réinitialiser le service pour la prochaine tentative
            get_embeddings._embedding_service = None
    
    print("Aucun service d'embeddings disponible. Impossible de continuer.")
    return None

def analyze_cv(cv_text, api_url=DEFAULT_API_URL):
    """Analyze a CV and extract key information
    Sends CV text to the backend API for analysis"""
    try:
        # Try to use remote API first
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    f"{api_url}/analyze-cv",
                    json={"cv_text": cv_text}
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    print(f"API error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Could not connect to backend API: {str(e)}")
            
        # Fallback to basic local analysis
        print("Falling back to basic local analysis...")
        
        # Get embeddings using local service
        cv_embedding = get_embeddings(cv_text)
        if not cv_embedding:
            print("Embeddings generation failed. Cannot complete analysis.")
            return None
            
        # Basic CV analysis without the full NLP pipeline
        skills = extract_skills_from_text(cv_text)
        domains = identify_domains(cv_text, skills)
        
        # Determine main domain based on skills frequency
        main_domain = max(domains.items(), key=lambda x: x[1])[0] if domains else "informatique_reseaux"
        
        # Extract job titles from the CV text
        job_titles = extract_job_titles(cv_text)
        
        return {
            "skills": skills[:10],
            "top_skills": skills[:5],
            "main_domain": main_domain,
            "domains": domains,
            "job_titles": job_titles[:3],
            "best_job_title": job_titles[0] if job_titles else "Développeur",
            "embedding": cv_embedding
        }
            
    except Exception as e:
        print(f"Error analyzing CV: {str(e)}")
        traceback.print_exc()
        return None

def extract_skills_from_text(text):
    """Basic skill extraction from text based on keyword matching"""
    common_skills = [
        "Python", "Java", "JavaScript", "C++", "C#", "SQL", "PHP", "Ruby", "Swift", 
        "Kotlin", "TypeScript", "HTML", "CSS", "Angular", "React", "Vue.js", "Node.js",
        "Django", "Flask", "Spring", "AWS", "Azure", "GCP", "Docker", "Kubernetes",
        "Git", "Linux", "DevOps", "CI/CD", "Jenkins", "Terraform", "Ansible",
        "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "Data Science",
        "Agile", "Scrum", "Product Management", "Communication", "Leadership",
        "Problem Solving", "Team Management", "Project Management"
    ]
    
    # Check which skills appear in the text
    found_skills = []
    for skill in common_skills:
        if skill.lower() in text.lower():
            found_skills.append(skill)
            
    return found_skills

def identify_domains(text, skills):
    """Basic domain identification based on skills"""
    domain_keywords = {
        "informatique_reseaux": ["réseau", "network", "systèmes", "infrastructure", "cloud"],
        "developpement_web": ["web", "html", "css", "javascript", "php", "react", "angular"],
        "data_science": ["data", "machine learning", "statistique", "python", "tensorflow", "pytorch"],
        "cybersecurite": ["sécurité", "security", "hacking", "cryptographie", "firewall"],
        "ia": ["intelligence artificielle", "ai", "machine learning", "deep learning"]
    }
    
    domains = {}
    for domain, keywords in domain_keywords.items():
        score = 0
        for keyword in keywords:
            if keyword.lower() in text.lower():
                score += 1
                
        # Add points for domain-specific skills
        for skill in skills:
            if any(kw in skill.lower() for kw in keywords):
                score += 0.5
                
        if score > 0:
            domains[domain] = score
            
    return domains

def extract_job_titles(text):
    """Basic job title extraction using common IT titles"""
    common_titles = [
        "Développeur", "Ingénieur", "Architecte", "Chef de projet", 
        "Data Scientist", "DevOps", "Admin Système", "Admin Réseau", 
        "Consultant", "Analyste", "Full Stack", "Frontend", "Backend",
        "Tech Lead", "Product Owner", "Scrum Master"
    ]
    
    # Extract job titles based on keyword matching
    titles = []
    for title in common_titles:
        if title.lower() in text.lower():
            titles.append(title)
            
    # If no specific titles found, add generic ones based on skills
    if not titles:
        titles = ["Développeur", "Ingénieur Logiciel", "Consultant IT"]
        
    return titles

def safe_read_cv_file(cv_path: str) -> str:
    """Lire le contenu d'un fichier CV de manière sécurisée avec plusieurs encodages"""
    try:
        # Vérifier le type de fichier
        if cv_path.lower().endswith('.pdf'):
            try:
                # Essayer d'extraire le texte avec PyPDF2
                with open(cv_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    cv_text = ""
                    for page in reader.pages:
                        cv_text += page.extract_text() + "\n"
                return cv_text
            except Exception as pdf_error:
                print(f"⚠️ Échec de l'extraction PDF: {str(pdf_error)}")
                try:
                    # Essayer avec pdfminer
                    return pdfminer_extract_text(cv_path)
                except:
                    # Dernier recours: utiliser une chaîne vide
                    print("⚠️ Impossible d'extraire le texte du PDF")
                    return ""
        else:
            # Pour les fichiers texte, essayer plusieurs encodages
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    with open(cv_path, 'r', encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            
            # Si tous les encodages ont échoué, lire en mode binaire avec remplacement
            with open(cv_path, 'rb') as f:
                return f.read().decode('utf-8', errors='replace')
    except Exception as e:
        print(f"⚠️ Impossible de lire le fichier CV: {e}")
        return ""

def analyze_cv_text(api_url: str, llm_url: str, cv_text: str) -> Dict[str, Any]:
    """
    Analyse un texte de CV pour déterminer le domaine professionnel et les compétences.
    
    Args:
        api_url: URL de l'API backend
        llm_url: URL du service LLM
        cv_text: Texte du CV
        
    Returns:
        Dictionnaire contenant l'analyse du CV
    """
    try:
        headers = {"Content-Type": "application/json"}
        
        # Données pour l'API
        data = {
            "cv_text": cv_text
        }
        
        # Envoi au backend
        response = httpx.post(
            f"{api_url}/analyze_cv",
            json=data,
            headers=headers,
            timeout=30.0
        )
        
        # Vérifier le code de statut
        if response.status_code == 200:
            # L'appel a réussi, retourner les résultats
            return response.json()
        else:
            # Si l'API a échoué, tenter une analyse locale
            print(f"Erreur API ({response.status_code}): {response.text}")
            
            # Utiliser l'analyse basée sur les règles
            return fallback_cv_analysis(cv_text)
    except Exception as e:
        print(f"Exception lors de l'analyse du CV: {str(e)}")
        
        # Utiliser l'analyse basée sur les règles
        return fallback_cv_analysis(cv_text)

def search_jobs(api_url: str, cv_text: str = None, domain: str = None, skills: List[str] = None, top_k: int = 3, min_score: float = -1.0) -> List[Dict[str, Any]]:
    """
    Recherche les offres d'emploi correspondant à un CV ou à un domaine/compétences.
    
    Args:
        api_url: URL de l'API backend
        cv_text: Texte du CV (optionnel)
        domain: Domaine professionnel (optionnel)
        skills: Liste des compétences (optionnel)
        top_k: Nombre d'offres à retourner
        min_score: Score minimum pour les offres
        
    Returns:
        Liste des offres d'emploi correspondantes
    """
    try:
        # Préparer les données pour l'API
        data = {
            "limit": top_k,
            "min_score": min_score
        }
        
        # Ajouter le texte du CV si fourni
        if cv_text:
            data["cv_text"] = cv_text
            
        # Ajouter le domaine si fourni
        if domain:
            data["domain"] = domain
            
        # Ajouter les compétences si fournies
        if skills:
            data["skills"] = skills
        
        # Définir les en-têtes
        headers = {"Content-Type": "application/json"}
        
        # Envoyer la requête
        response = httpx.post(
            f"{api_url}/search_jobs_for_resume",
            json=data,
            headers=headers,
            timeout=30.0
        )
        
        # Vérifier le code de statut
        if response.status_code == 200:
            # L'appel a réussi, retourner les résultats
            return response.json().get("results", [])
        else:
            # Une erreur s'est produite
            print(f"Erreur API ({response.status_code}): {response.text}")
            return []
    except Exception as e:
        print(f"Exception lors de la recherche d'offres: {str(e)}")
        return []

def format_results(job_offers: List[Dict[str, Any]], cv_domain: str, skills: List[str] = None) -> List[Dict[str, Any]]:
    """Formate et enrichit les résultats pour l'affichage"""
    formatted_results = []
    
    for i, job in enumerate(job_offers):
        # Récupérer les métadonnées
        metadata = job.get("metadata", {})
        
        # Vérifier si le domaine de l'offre correspond au domaine du CV
        job_domain = metadata.get("domain", "unknown")
        domain_match = job_domain == cv_domain
        
        # Calculer un score de pertinence combiné
        score = job.get("score", 0.0)
        domain_bonus = 0.2 if domain_match else 0.0  # Bonus pour le matching de domaine
        
        # Ajouter une pondération basée sur les compétences spécifiques
        skills_bonus = 0.0
        job_content = job.get("content", "").lower()
        if skills:
            hits = 0
            for skill in skills:
                if skill.lower() in job_content:
                    hits += 1
            
            # Bonus maximum de 0.3 pour les compétences (0.05 par compétence trouvée, jusqu'à 6 max)
            skills_bonus = min(0.3, hits * 0.05)
            
        combined_score = score + domain_bonus + skills_bonus
        
        # Enrichir le résultat avec des informations supplémentaires
        formatted_job = {
            "rank": i + 1,
            "id": job.get("id", "N/A"),
            "title": metadata.get("title", "N/A"),
            "company": metadata.get("company", "N/A"),
            "location": metadata.get("location", "N/A"),
            "domain": job_domain,
            "domain_match": "✅" if domain_match else "❌",
            "raw_score": score,
            "domain_bonus": domain_bonus,
            "skills_bonus": skills_bonus,
            "matched_skills": hits if 'hits' in locals() else 0,
            "score": combined_score,
            "description": job.get("content", "")[:200] + "..." if len(job.get("content", "")) > 200 else job.get("content", "")
        }
        
        formatted_results.append(formatted_job)
    
    # Trier les résultats par score combiné
    formatted_results.sort(key=lambda x: x["score"], reverse=True)
    
    # Mettre à jour les rangs après le tri
    for i, job in enumerate(formatted_results):
        job["rank"] = i + 1
    
    return formatted_results

def improved_cv_matching(
    cv_text: str = None,
    cv_path: str = None,
    api_url: str = DEFAULT_API_URL,
    llm_url: str = DEFAULT_LLM_URL,
    top_k: int = 5,
    min_score: float = 0.0,
    strict_threshold: bool = False,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Version améliorée de la fonction de matching CV-Offres.
    Gère les différents formats de CV et inclut un système de secours pour l'analyse.
    
    Args:
        cv_text: Texte du CV (si déjà extrait)
        cv_path: Chemin vers le fichier CV (PDF ou TXT)
        api_url: URL de l'API backend
        llm_url: URL du service LLM
        top_k: Nombre d'offres à retourner
        min_score: Score minimum pour les offres
        strict_threshold: Utiliser un seuil strict pour le matching de domaine
        verbose: Afficher des informations détaillées
        
    Returns:
        Dictionnaire contenant les résultats du matching
    """
    timestamp = datetime.now().isoformat()
    
    # Obtenir le texte du CV
    if cv_path and not cv_text:
        cv_text = safe_read_cv_file(cv_path)
    
    if not cv_text:
        print("❌ Erreur: Aucun texte de CV fourni ou extrait")
        return None
    
    try:
        # Essayer d'analyser le CV avec l'API backend
        api_success = False
        try:
            # Déterminer le type MIME pour configurer le Content-Type
            content_type = "text/plain"
            if cv_path:
                mime_type, _ = mimetypes.guess_type(cv_path)
                if mime_type:
                    content_type = mime_type
            
            # Préparer les en-têtes avec le type de contenu correct
            headers = {"Content-Type": content_type}
            
            # Faire l'appel à l'API
            response = httpx.post(
                f"{api_url}/analyze_cv",
                headers=headers,
                content=cv_text,
                timeout=30.0  # Augmenter le timeout pour les CVs longs
            )
            
            # Vérifier si la réponse est OK
            if response.status_code == 200:
                cv_analysis = response.json()
                api_success = True
            else:
                print(f"⚠️ Erreur API ({response.status_code}): {response.text}")
                raise Exception(f"Erreur API: {response.status_code}")
                
        except Exception as e:
            print(f"⚠️ Échec de l'analyse du CV via l'API: {str(e)}")
            print("Utilisation de l'analyse de secours...")
            
            # Si l'API a échoué, utiliser l'analyse de secours
            cv_analysis = fallback_cv_analysis(cv_text)
        
        # Rechercher les offres correspondantes
        job_offers = []
        
        if api_success:
            # Utiliser l'API pour rechercher les offres
            try:
                offers_response = httpx.post(
                    f"{api_url}/search_offers",
                    json={
                        "cv_analysis": cv_analysis,
                        "top_k": top_k,
                        "min_score": min_score,
                        "strict": strict_threshold
                    },
                    timeout=30.0
                )
                
                if offers_response.status_code == 200:
                    job_offers = offers_response.json().get("results", [])
                else:
                    print(f"⚠️ Erreur recherche offres ({offers_response.status_code}): {offers_response.text}")
                    # Utiliser une liste vide si la recherche échoue
            except Exception as e:
                print(f"⚠️ Échec de la recherche d'offres: {str(e)}")
        else:
            # Recherche basique par domaine si l'API a échoué
            domain = cv_analysis.get("main_domain")
            if domain:
                print(f"Recherche d'offres dans le domaine: {domain}")
                try:
                    # Parcourir les fichiers d'offres
                    job_offers = search_job_offers_in_domain(domain, top_k)
                except Exception as e:
                    print(f"⚠️ Échec de la recherche d'offres de secours: {str(e)}")
        
        # Préparer les résultats
        results = {
            "timestamp": timestamp,
            "cv_analysis": cv_analysis,
            "results": job_offers,
            "api_success": api_success,
            "parameters": {
                "top_k": top_k,
                "min_score": min_score,
                "strict_threshold": strict_threshold
            }
        }
        
        return results
        
    except Exception as e:
        print(f"❌ Erreur lors du matching CV-Offres: {str(e)}")
        return None

def search_job_offers_in_domain(domain: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Recherche des offres d'emploi dans un domaine spécifique.
    Utilisé comme méthode de secours quand l'API n'est pas disponible.
    
    Args:
        domain: Domaine professionnel
        top_k: Nombre maximum d'offres à retourner
        
    Returns:
        Liste des offres d'emploi correspondantes
    """
    results = []
    
    # Exemples d'offres fictives par domaine (à remplacer par une vraie base de données)
    sample_offers = {
        "informatique_reseaux": [
            {
                "id": "job001",
                "title": "DevOps Engineer Senior",
                "company": "TechCloud Solutions",
                "domain": "informatique_reseaux",
                "description": "Nous recherchons un ingénieur DevOps expérimenté pour gérer notre infrastructure cloud et améliorer nos pipelines CI/CD.",
                "skills": ["AWS", "Docker", "Kubernetes", "CI/CD", "Terraform", "Python", "Ansible"],
                "location": "Paris",
                "raw_score": 0.75,
                "domain_bonus": 0.1,
                "skills_bonus": 0.15,
                "matched_skills": 3
            },
            {
                "id": "job002",
                "title": "Développeur Full Stack JavaScript",
                "company": "WebTech Agency",
                "domain": "informatique_reseaux",
                "description": "Rejoignez notre équipe agile pour développer des applications web modernes utilisant React et Node.js.",
                "skills": ["JavaScript", "React", "Node.js", "MongoDB", "Git", "Agile", "REST API"],
                "location": "Lyon",
                "raw_score": 0.65,
                "domain_bonus": 0.1,
                "skills_bonus": 0.1,
                "matched_skills": 2
            },
            {
                "id": "job003",
                "title": "Ingénieur DevOps",
                "company": "TechCloud Solutions",
                "domain": "informatique_reseaux",
                "description": "Vous serez responsable de l'automatisation des déploiements et de la gestion de l'infrastructure cloud.",
                "skills": ["GCP", "Docker", "Kubernetes", "CI/CD", "Terraform", "Go", "Prometheus"],
                "location": "Paris",
                "raw_score": 0.7,
                "domain_bonus": 0.1,
                "skills_bonus": 0.1,
                "matched_skills": 2
            }
        ],
        "automatismes_info_industrielle": [
            {
                "id": "job004",
                "title": "Ingénieur Automaticien",
                "company": "IndusTech",
                "domain": "automatismes_info_industrielle",
                "description": "Conception et programmation de systèmes automatisés pour l'industrie 4.0.",
                "skills": ["Siemens S7", "Schneider", "SCADA", "HMI", "PLC", "Industrie 4.0"],
                "location": "Grenoble",
                "raw_score": 0.7,
                "domain_bonus": 0.1,
                "skills_bonus": 0.1,
                "matched_skills": 2
            }
        ],
        "finance": [
            {
                "id": "job005",
                "title": "Analyste Financier",
                "company": "FinConsult Group",
                "domain": "finance",
                "description": "Analyse financière et modélisation pour des projets d'investissement.",
                "skills": ["Excel", "Modélisation", "Analyses", "SAP", "VBA", "Power BI"],
                "location": "Paris",
                "raw_score": 0.75,
                "domain_bonus": 0.1,
                "skills_bonus": 0.1,
                "matched_skills": 2
            }
        ],
        "genie_civil_btp": [
            {
                "id": "job006",
                "title": "Ingénieur Structure",
                "company": "BuildTech Solutions",
                "domain": "genie_civil_btp",
                "description": "Conception et calcul de structures pour des bâtiments et ouvrages d'art.",
                "skills": ["Autocad", "Robot", "Revit", "BIM", "Calcul de structures"],
                "location": "Marseille",
                "raw_score": 0.72,
                "domain_bonus": 0.1,
                "skills_bonus": 0.08,
                "matched_skills": 2
            }
        ],
        "genie_industriel": [
            {
                "id": "job007",
                "title": "Ingénieur Méthodes",
                "company": "IndusProd",
                "domain": "genie_industriel",
                "description": "Optimisation des processus de production et amélioration continue.",
                "skills": ["Lean", "Six Sigma", "5S", "Kaizen", "SAP", "Excel"],
                "location": "Lille",
                "raw_score": 0.68,
                "domain_bonus": 0.1,
                "skills_bonus": 0.12,
                "matched_skills": 3
            }
        ]
    }
    
    # Tenter de récupérer les offres depuis notre exemple
    if domain in sample_offers:
        domain_offers = sample_offers[domain]
        
        # Ajouter des index et calculer les scores finaux
        for i, offer in enumerate(domain_offers[:top_k]):
            # Calculer le score final
            score = offer["raw_score"] + offer["domain_bonus"] + offer["skills_bonus"]
            
            # Ajouter à nos résultats
            results.append({
                "id": offer["id"],
                "rank": i + 1,
                "title": offer["title"],
                "company": offer["company"],
                "domain": offer["domain"],
                "description": offer["description"],
                "raw_score": offer["raw_score"],
                "domain_bonus": offer["domain_bonus"],
                "skills_bonus": offer["skills_bonus"],
                "matched_skills": offer["matched_skills"],
                "score": score,
                "domain_match": "✅"  # Toujours un match car nous avons filtré par domaine
            })
    
    # Aussi choisir une variante pour DevOps Engineer Senior
    if domain == "informatique_reseaux" and len(results) > 0:
        # Créer une variante de l'offre DevOps Engineer Senior
        devops_variant = {
            "id": "job003_var",
            "rank": len(results) + 1,
            "title": "DevOps Engineer Senior (Variante 2) (Variante 3)",
            "company": "TechCloud Solutions B C",
            "domain": "informatique_reseaux",
            "description": "Nous recherchons un ingénieur DevOps expérimenté pour gérer notre infrastructure cloud.",
            "raw_score": 0.7,
            "domain_bonus": 0.1,
            "skills_bonus": 0.1,
            "matched_skills": 2,
            "score": 0.9,
            "domain_match": "✅"
        }
        results.append(devops_variant)
    
    # Trier par score décroissant et limiter au nombre demandé
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

async def store_indeed_jobs_in_vector_db(api_url: str, indeed_jobs: List[Dict[str, Any]]) -> List[str]:
    """
    Stocke les offres d'emploi Indeed dans la base de données vectorielle.
    Si la base vectorielle n'est pas disponible, sauvegarde localement.
    
    Args:
        api_url: URL de l'API backend
        indeed_jobs: Liste des offres d'emploi Indeed à stocker
        
    Returns:
        Liste des IDs des offres stockées
    """
    job_ids = []
    
    try:
        # Créer un répertoire local pour le stockage de secours
        backup_dir = os.path.join("data", "indeed_jobs")
        os.makedirs(backup_dir, exist_ok=True)
        
        # Horodatage pour le nom du fichier de sauvegarde groupe
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        batch_file = os.path.join(backup_dir, f"batch_{timestamp}.json")
        
        # Liste pour le fichier groupé
        batch_jobs = []
        success_count = 0
        failed_count = 0
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            for job in indeed_jobs:
                # Préparation des métadonnées
                metadata = {
                    "title": job.get("title", "N/A"),
                    "company": job.get("company", "N/A"),
                    "location": job.get("location", "N/A"),
                    "domain": job.get("domain", "informatique_reseaux"),
                    "country": job.get("country", "France"),
                    "salary": job.get("salary", ""),
                    "source": "indeed",
                    "url": job.get("url", ""),
                    "score": job.get("match_score", 0.0),
                    "date_posted": job.get("date_posted", ""),
                    "scraped_date": datetime.now().isoformat()
                }
                
                # Construction du texte complet de l'offre
                job_content = f"{job['title']} - {job['company']}\n\n{job.get('description', '')}"
                
                # Données pour l'API
                document_data = {
                    "content": job_content,
                    "metadata": metadata
                }
                
                # Ajouter au lot pour sauvegarde locale
                job_copy = job.copy()
                job_copy["metadata"] = metadata
                job_copy["content"] = job_content
                batch_jobs.append(job_copy)
                
                # Tenter d'envoyer à l'API
                try:
                    response = await client.post(
                        f"{api_url}/upload/job",
                        json=document_data
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        job_id = result.get("id")
                        if job_id:
                            job_ids.append(job_id)
                            job_copy["vector_db_id"] = job_id
                            print(f"✅ Offre stockée: {job['title']} (ID: {job_id})")
                            success_count += 1
                        else:
                            print(f"⚠️ ID non retourné pour: {job['title']}")
                            failed_count += 1
                    else:
                        print(f"❌ Échec du stockage pour {job['title']}: {response.status_code} - {response.text}")
                        failed_count += 1
                        
                        # Sauvegarde individuelle en cas d'échec
                        job_id = job.get("id", f"indeed_{hash(job['title'] + job['company']) & 0xffffffff}")
                        individual_file = os.path.join(backup_dir, f"{job_id}.json")
                        
                        with open(individual_file, 'w', encoding='utf-8') as f:
                            json.dump(job_copy, f, ensure_ascii=False, indent=2)
                            
                        print(f"📁 Sauvegarde locale: {individual_file}")
                        
                except Exception as e:
                    print(f"❌ Exception lors du stockage de {job['title']}: {str(e)}")
                    failed_count += 1
                    
                    # Sauvegarde individuelle en cas d'erreur
                    job_id = job.get("id", f"indeed_{hash(job['title'] + job['company']) & 0xffffffff}")
                    individual_file = os.path.join(backup_dir, f"{job_id}.json")
                    
                    with open(individual_file, 'w', encoding='utf-8') as f:
                        json.dump(job_copy, f, ensure_ascii=False, indent=2)
                        
                    print(f"📁 Sauvegarde locale: {individual_file}")
        
        # Sauvegarder le lot complet
        group_data = {
            "date": datetime.now().isoformat(),
            "success_count": success_count,
            "failed_count": failed_count,
            "total_count": len(indeed_jobs),
            "jobs": batch_jobs
        }
        
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(group_data, f, ensure_ascii=False, indent=2)
            
        print(f"📁 Sauvegarde groupée: {batch_file}")
        print(f"Résumé: {success_count} offres stockées en base vectorielle, {failed_count} échecs (sauvegardées localement)")
        
    except Exception as e:
        print(f"Erreur lors du stockage des offres Indeed: {str(e)}")
        
        # Fallback: tout sauvegarder localement en cas d'erreur globale
        try:
            backup_dir = os.path.join("data", "indeed_jobs")
            os.makedirs(backup_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            emergency_file = os.path.join(backup_dir, f"emergency_backup_{timestamp}.json")
            
            with open(emergency_file, 'w', encoding='utf-8') as f:
                json.dump({"jobs": indeed_jobs, "error": str(e)}, f, ensure_ascii=False, indent=2)
                
            print(f"📁 Sauvegarde d'urgence créée: {emergency_file}")
        except Exception as backup_error:
            print(f"❌ Échec complet de sauvegarde: {str(backup_error)}")
    
    return job_ids

async def reimport_local_jobs_to_vector_db(api_url: str, max_jobs: int = 0) -> Tuple[int, int]:
    """
    Réimporte les offres d'emploi sauvegardées localement vers la base vectorielle.
    Utile quand le service d'embeddings redevient disponible après une panne.
    
    Args:
        api_url: URL de l'API backend
        max_jobs: Nombre maximum d'offres à importer (0 = toutes)
        
    Returns:
        Tuple (succès, échecs) contenant le nombre d'offres importées avec succès et le nombre d'échecs
    """
    print("\n" + "=" * 50)
    print("RÉIMPORTATION DES OFFRES LOCALES VERS LA BASE VECTORIELLE")
    print("=" * 50)
    
    try:
        # Vérifier si le répertoire existe
        backup_dir = os.path.join("data", "indeed_jobs")
        if not os.path.exists(backup_dir):
            print(f"❌ Répertoire de sauvegarde introuvable: {backup_dir}")
            return (0, 0)
            
        # Récupérer tous les fichiers JSON
        json_files = [f for f in os.listdir(backup_dir) if f.endswith('.json')]
        if not json_files:
            print("❌ Aucun fichier d'offres à importer")
            return (0, 0)
            
        # Déterminer les types de fichiers
        individual_files = [f for f in json_files if not f.startswith('batch_') and not f.startswith('emergency_backup_')]
        batch_files = [f for f in json_files if f.startswith('batch_')]
        
        # Prioriser les fichiers de lot pour l'importation
        all_jobs = []
        processed_job_ids = set()
        
        # D'abord extraire les offres des fichiers de lot
        for batch_file in batch_files:
            try:
                with open(os.path.join(backup_dir, batch_file), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    batch_jobs = data.get('jobs', [])
                    for job in batch_jobs:
                        job_id = job.get('id')
                        if job_id and job_id not in processed_job_ids:
                            processed_job_ids.add(job_id)
                            all_jobs.append(job)
            except Exception as e:
                print(f"❌ Erreur lors de la lecture du fichier {batch_file}: {str(e)}")
                
        # Ensuite ajouter les offres des fichiers individuels si elles n'ont pas déjà été traitées
        for job_file in individual_files:
            try:
                with open(os.path.join(backup_dir, job_file), 'r', encoding='utf-8') as f:
                    job = json.load(f)
                    job_id = job.get('id')
                    if job_id and job_id not in processed_job_ids:
                        processed_job_ids.add(job_id)
                        all_jobs.append(job)
            except Exception as e:
                print(f"❌ Erreur lors de la lecture du fichier {job_file}: {str(e)}")
        
        # Limiter le nombre d'offres si demandé
        if max_jobs > 0 and len(all_jobs) > max_jobs:
            all_jobs = all_jobs[:max_jobs]
            
        print(f"📊 {len(all_jobs)} offres identifiées pour importation")
        
        # Importer les offres dans la base vectorielle
        success_count = 0
        failed_count = 0
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            for job in all_jobs:
                # Préparation des métadonnées
                metadata = job.get('metadata', {})
                if not metadata:
                    # Créer les métadonnées si absentes
                    metadata = {
                        "title": job.get("title", "N/A"),
                        "company": job.get("company", "N/A"),
                        "location": job.get("location", "N/A"),
                        "domain": job.get("domain", "informatique_reseaux"),
                        "country": job.get("country", "France"),
                        "salary": job.get("salary", ""),
                        "source": "indeed",
                        "url": job.get("url", ""),
                        "score": job.get("match_score", 0.0),
                        "date_posted": job.get("date_posted", ""),
                        "scraped_date": job.get("scraped_date", datetime.now().isoformat())
                    }
                
                # Construction du texte complet de l'offre
                job_content = job.get("content", "")
                if not job_content:
                    # Créer le contenu s'il est absent
                    job_content = f"{job.get('title', '')} - {job.get('company', '')}\n\n{job.get('description', '')}"
                
                # Données pour l'API
                document_data = {
                    "content": job_content,
                    "metadata": metadata
                }
                
                # Envoi à l'API
                try:
                    response = await client.post(
                        f"{api_url}/upload/job",
                        json=document_data
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        job_id = result.get("id")
                        if job_id:
                            print(f"✅ Offre importée: {metadata.get('title')} (ID: {job_id})")
                            success_count += 1
                        else:
                            print(f"⚠️ ID non retourné pour: {metadata.get('title')}")
                            failed_count += 1
                    else:
                        print(f"❌ Échec de l'importation pour {metadata.get('title')}: {response.status_code} - {response.text}")
                        failed_count += 1
                except Exception as e:
                    print(f"❌ Exception lors de l'importation de {metadata.get('title')}: {str(e)}")
                    failed_count += 1
                    
        # Résumé des résultats
        print(f"\n📊 Résumé de l'importation:")
        print(f"  ✅ {success_count} offres importées avec succès")
        print(f"  ❌ {failed_count} échecs d'importation")
        print(f"  📈 Taux de succès: {success_count / (success_count + failed_count) * 100:.1f}%" if (success_count + failed_count) > 0 else "  📈 Aucune offre traitée")
        
        return (success_count, failed_count)
        
    except Exception as e:
        print(f"❌ Erreur lors de la réimportation des offres: {str(e)}")
        return (0, 0)
    


def fallback_cv_analysis(cv_text):
    """
    Analyse de secours d'un CV basée sur les mots-clés.
    Utilisé quand l'API n'est pas disponible.
    
    Args:
        cv_text (str): Texte du CV à analyser
        
    Returns:
        dict: Résultats de l'analyse avec domaine, scores et compétences
    """
    print("Analyse fallback du CV par mots-clés...")
    cv_text = cv_text.lower()
    
    # Mots-clés par domaine
    domain_keywords = {
        "informatique_reseaux": [
            "python", "java", "javascript", "html", "css", "php", "ruby", "c++", "c#", 
            "développeur", "software", "frontend", "backend", "fullstack", "web", "mobile",
            "cloud", "devops", "aws", "azure", "docker", "kubernetes", "linux", "windows",
            "réseau", "network", "cisco", "cybersécurité", "sécurité", "security", "data",
            "database", "sql", "nosql", "mongodb", "mysql", "postgresql", "oracle", "git"
        ],
        "automatismes_info_industrielle": [
            "automatisme", "automate", "plc", "api", "scada", "supervision", "vba", "hmi",
            "modbus", "profinet", "profibus", "ethernet/ip", "siemens", "schneider", "rockwell",
            "automation", "robot", "abb", "kuka", "fanuc", "step7", "tia portal", "unity",
            "grafcet", "électrotechnique", "électrique", "électronique", "capteur", "actionneur"
        ],
        "finance": [
            "finance", "comptabilité", "comptable", "audit", "analyste", "financial", "accounting",
            "bilan", "trésorerie", "fiscal", "impôt", "taxe", "budget", "forecast", "prévision",
            "contrôle", "contrôleur", "gestion", "management", "reporting", "kpi", "tableau de bord",
            "sap", "sage", "cegid", "consolidation", "ifrs", "us gaap", "pcg", "bâle", "solvabilité"
        ],
        "genie_civil_btp": [
            "génie civil", "btp", "bâtiment", "travaux publics", "construction", "chantier",
            "béton", "structure", "fondation", "maçonnerie", "charpente", "menuiserie", "plomberie",
            "électricité", "hvac", "autocad", "revit", "bim", "sketchup", "archicad", "catia",
            "génie", "géotechnique", "topographie", "architecture", "urbanisme", "environnement"
        ],
        "genie_industriel": [
            "génie industriel", "production", "fabrication", "usine", "atelier", "lean", "kaizen",
            "5s", "six sigma", "qualité", "iso", "supply chain", "logistique", "stock", "inventory",
            "amélioration continue", "maintenance", "méthodes", "temps", "ergonomie", "sécurité",
            "hse", "environnement", "gestion projet", "planification", "ordonnancement", "mrp", "erp"
        ]
    }
    
    # Compétences techniques courantes
    technical_skills = [
        # Informatique
        "Python", "Java", "JavaScript", "HTML", "CSS", "PHP", "Ruby", "C++", "C#", ".NET",
        "React", "Angular", "Vue.js", "Node.js", "Django", "Flask", "Spring", "Laravel",
        "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Git", "DevOps", "CI/CD", "Jenkins",
        "MongoDB", "MySQL", "PostgreSQL", "SQL Server", "Oracle", "Redis", "ElasticSearch",
        "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "NLP", "Data Science",
        "Linux", "Windows", "macOS", "Bash", "PowerShell", "Agile", "Scrum", "Kanban",
        
        # Automatismes
        "PLC", "SCADA", "HMI", "Siemens", "Allen Bradley", "Schneider", "Fanuc", "ABB",
        "TIA Portal", "Step7", "Unity Pro", "RSLogix", "Factory Talk", "WinCC", "Ignition",
        "Modbus", "Profinet", "Profibus", "EtherNet/IP", "OPC UA", "MQTT", "IoT", "IIoT",
        
        # Finance
        "SAP", "SAP FI", "SAP CO", "Oracle Finance", "Sage", "Cegid", "NAV", "Dynamics",
        "Excel VBA", "Power BI", "Tableau", "Qlik", "IFRS", "US GAAP", "PCG", "Consolidation",
        
        # Génie Civil & BTP
        "AutoCAD", "Revit", "BIM", "SketchUp", "ArchiCAD", "CATIA", "Tekla", "Robot",
        "Abaqus", "ETABS", "SAP2000", "STAAD.Pro", "Plaxis", "MS Project", "Primavera",
        
        # Génie Industriel
        "ERP", "MRP", "GMAO", "GPAO", "SAP MM", "SAP PP", "SAP QM", "SAP PM", "SAP WM",
        "Lean", "Six Sigma", "Kaizen", "5S", "SMED", "TPM", "TQM", "JIT", "Kanban",
        "ISO 9001", "ISO 14001", "ISO 45001", "HACCP", "BRC", "IFS", "IATF 16949"
    ]
    
    # Calculer les scores par domaine
    domain_scores = {}
    for domain, keywords in domain_keywords.items():
        score = 0.5  # Score de base
        matches = 0
        
        for keyword in keywords:
            if keyword.lower() in cv_text:
                matches += 1
                # Bonus si le mot apparaît plusieurs fois
                occurrences = cv_text.count(keyword.lower())
                if occurrences > 1:
                    score += 0.05 * min(occurrences, 3)  # Limiter le bonus
                else:
                    score += 0.025
        
        # Normaliser le score entre 0 et 1
        domain_scores[domain] = min(score, 1.0)
    
    # Trouver le domaine principal
    main_domain = max(domain_scores, key=domain_scores.get)
    
    # Extraire les compétences
    skills = []
    for skill in technical_skills:
        if skill.lower() in cv_text or any(alt.lower() in cv_text for alt in [f"{skill}s", f"{skill}ing", f"{skill}ed"]):
            skills.append(skill)
    
    # Limiter à 10 compétences maximum
    skills = skills[:10]
    
    # Construire le résultat
    result = {
        "main_domain": main_domain,
        "domain_scores": domain_scores,
        "skills": skills,
        "language": "fr"  # Par défaut en français
    }
    
    print(f"Domaine identifié: {main_domain} (score: {domain_scores[main_domain]:.2f})")
    print(f"Scores par domaine: {domain_scores}")
    print(f"Compétences identifiées: {skills}")
    
    return result

def main():
    """Point d'entrée principal du script."""
    # Récupérer les arguments
    args = parse_arguments()
    
    # Réimportation des offres locales si demandé
    if args.reimport_local:
        import asyncio
        success, failed = asyncio.run(reimport_local_jobs_to_vector_db(args.api_url, args.reimport_max))
        print(f"Réimportation terminée: {success} succès, {failed} échecs.")
        # Si c'est le seul argument, terminer le script
        if not args.cv and not args.text:
            return
    
    # Récupérer le contenu du CV
    if args.cv:
        if args.verbose:
            print(f"Lecture du fichier CV: {args.cv}")
        cv_text = safe_read_cv_file(args.cv)
    else:
        cv_text = args.text
    
    if args.verbose:
        print("Analyse du CV et recherche d'offres en cours...")
    
    # Récupérer les résultats
    results = improved_cv_matching(
        cv_text=cv_text,
        api_url=args.api_url,
        llm_url=args.llm_url,
        top_k=args.top_k,
        min_score=args.min_score,
        strict_threshold=args.strict
    )
    
    # Afficher les résultats
    if results:
        print("\n" + "="*50)
        print("RÉSULTATS DE L'ANALYSE DU CV:")
        print("="*50)
        
        cv_analysis = results.get("cv_analysis", {})
        print(f"Domaine principal: {cv_analysis.get('main_domain', 'Non identifié')}")
        
        print("\nScores par domaine:")
        domain_scores = cv_analysis.get("domain_scores", {})
        for domain, score in domain_scores.items():
            print(f"  - {domain}: {score:.2f}")
        
        print("\nCompétences identifiées:")
        skills = cv_analysis.get("skills", [])
        if skills:
            for skill in skills:
                print(f"  - {skill}")
        else:
            print("  Aucune compétence spécifique identifiée.")
        
        # Afficher les offres correspondantes
        if "results" in results and results["results"]:
            print("\n" + "="*50)
            print("OFFRES D'EMPLOI CORRESPONDANTES:")
            print("="*50)
            
            # Convertir en dataframe pour un affichage tabulaire
            job_results = results["results"]
            df_results = pd.DataFrame([
                {
                    "Rang": job["rank"],
                    "Titre": job["title"],
                    "Entreprise": job["company"],
                    "Domaine": job["domain"],
                    "Score": job["score"],
                    "Match": "✓" if job["domain_match"] == "✅" else "✗"
                }
                for job in job_results
            ])
            
            print(tabulate(df_results, headers="keys", tablefmt="pretty", showindex=False))
            
            # Détails des offres
            print("\nDÉTAILS DES OFFRES:")
            for job in job_results:
                print(f"\n{job['rank']}. {job['title']} - {job['company']} (Score: {job['score']:.2f})")
                print(f"   Domaine: {job['domain']} {job['domain_match']}")
                print(f"   Score brut: {job['raw_score']:.2f}")
                print(f"   Bonus domaine: {job['domain_bonus']:.2f}")
                print(f"   Bonus compétences: {job['skills_bonus']:.2f} ({job['matched_skills']} compétences)")
        else:
            print("\nAucune offre d'emploi correspondante trouvée.")
        
        # Rechercher des offres sur Indeed si l'option est activée
        if args.indeed and cv_analysis:
            print("\n" + "="*50)
            print("OFFRES D'EMPLOI INDEED:")
            print("="*50)
            
            try:
                # Générer une requête de recherche
                query = generate_indeed_query({"cv_analysis": cv_analysis})
                print(f"Recherche Indeed pour: {query}")
                print(f"Localisation: {args.indeed_location}")
                print(f"Pays: {'Maroc' if args.indeed_country == 'ma' else 'France'}")
                
                # Scraper les offres
                jobs = scrape_indeed_jobs(
                    query, 
                    args.indeed_location, 
                    args.indeed_max_results,
                    country_code=args.indeed_country
                )
                
                if jobs:
                    # Calculer les scores de matching
                    for job in jobs:
                        match_score = match_job_with_cv(job, {"cv_analysis": cv_analysis})
                        job["match_score"] = match_score
                    
                    # Trier par score de matching
                    jobs.sort(key=lambda x: x.get("match_score", 0), reverse=True)
                    
                    # Mettre à jour les rangs
                    for i, job in enumerate(jobs):
                        job["rank"] = i + 1
                    
                    # Afficher les résultats Indeed
                    df_indeed = pd.DataFrame([
                        {
                            "Rang": job["rank"],
                            "Titre": job["title"],
                            "Entreprise": job["company"],
                            "Lieu": job["location"],
                            "Score": job["match_score"]
                        }
                        for job in jobs
                    ])
                    
                    print(tabulate(df_indeed, headers="keys", tablefmt="pretty", showindex=False))
                    
                    # Détails des offres
                    print("\nDÉTAILS DES OFFRES INDEED:")
                    for job in jobs:
                        print(f"\n{job['rank']}. {job['title']} - {job['company']} (Score: {job['match_score']:.2f})")
                        print(f"   Lieu: {job['location']}")
                        
                        # Afficher le salaire si disponible
                        if job.get("salary"):
                            print(f"   Salaire: {job['salary']}")
                        
                        # Afficher la description (limitée)
                        desc = job.get('description', '')
                        if desc:
                            # Limiter la longueur
                            if len(desc) > 300:
                                desc = desc[:300] + "..."
                            print(f"   Description: {desc}")
                        
                        if job.get('url'):
                            print(f"   URL: {job['url']}")
                    
                    # Stocker les offres dans la base de données vectorielle si demandé
                    if args.store_indeed:
                        print("\nStockage des offres Indeed dans la base de données vectorielle...")
                        job_ids = asyncio.run(store_indeed_jobs_in_vector_db(args.api_url, jobs))
                        print(f"{len(job_ids)} offres stockées avec succès.")
                else:
                    print("Aucune offre d'emploi trouvée sur Indeed.")
            
            except Exception as e:
                print(f"Erreur lors de la récupération des offres Indeed: {str(e)}")
        
        # Sauvegarder les résultats si demandé
        if args.output or DEFAULT_OUTPUT_DIR:
            output_path = args.output
            if not output_path:
                # Créer un nom de fichier basé sur la date/heure
                os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
                output_path = os.path.join(
                    DEFAULT_OUTPUT_DIR, 
                    f"cv_matching_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            
            if args.verbose:
                print(f"\nRésultats sauvegardés dans: {output_path}")
    else:
        print("Erreur: Impossible d'obtenir des résultats de matching.")

if __name__ == "__main__":
    main() 