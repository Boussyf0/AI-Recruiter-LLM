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
import glob
import hashlib

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
            service = LocalEmbeddingService(model_name="paraphrase-multilingual-MiniLM-L12-v2")
            return service.get_embeddings(text, domain=domain)
        except Exception as e:
            print(f"Erreur avec le service d'embeddings local: {str(e)}")
    
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
    """Extrait les compétences à partir du texte"""
    # Liste de compétences communes à identifier
    common_skills = [
        # Langages de programmation
        "Python", "Java", "C++", "C#", "JavaScript", "TypeScript", "R", "Ruby", "PHP", "Go", "Rust", "Swift",
        "SQL", "NoSQL", "Kotlin", "Scala", "Perl", "Bash", "PowerShell", "Assembly",
        
        # Frameworks et bibliothèques
        "Django", "Flask", "FastAPI", "React", "Angular", "Vue.js", "Express", "Spring Boot", "Laravel",
        "ASP.NET", ".NET Core", "TensorFlow", "PyTorch", "Keras", "Scikit-learn", "Pandas", "NumPy",
        "Node.js", "jQuery", "Bootstrap", "Symfony", "Ruby on Rails", "Spring", "Redux", "Svelte",

        # DevOps & Cloud
        "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Jenkins", "GitLab CI", "GitHub Actions",
        "Terraform", "Ansible", "Puppet", "Chef", "CloudFormation", "Serverless", "CircleCI",
        "Travis CI", "Prometheus", "Grafana", "ELK Stack", "Datadog", "EC2", "S3", "Lambda",
        "Azure Functions", "Google Cloud Functions", "Consul", "Vault", "Nginx", "Apache",
        
        # Bases de données
        "MySQL", "PostgreSQL", "MongoDB", "Cassandra", "Redis", "Elasticsearch", "DynamoDB",
        "Oracle", "SQL Server", "Firebase", "Neo4j", "Couchbase", "CouchDB", "MariaDB",
        
        # Data Science & Big Data
        "Machine Learning", "Deep Learning", "NLP", "Computer Vision", "Data Mining",
        "Hadoop", "Spark", "Kafka", "Airflow", "Databricks", "Snowflake", "Redshift",
        "BigQuery", "ETL", "Data Warehousing", "OLAP", "OLTP", "Data Modeling", "BI", "Tableau",
        "Power BI", "Looker", "Data Studio", "Qlik", "Data Engineering", "MLOps",
        
        # Méthodologies & Pratiques
        "Agile", "Scrum", "Kanban", "DevOps", "TDD", "BDD", "CI/CD", "Microservices", "REST APIs", 
        "GraphQL", "SOAP", "OOP", "Design Patterns", "SOA", "DDD", "SOLID", "XP",
        "Serverless", "Event-Driven", "Reactive Programming", "Functional Programming",
        
        # Mobile
        "iOS", "Android", "Flutter", "React Native", "Xamarin", "Ionic", "Swift", "Kotlin",
        "Objective-C", "Mobile Development", "Progressive Web Apps", "Cordova",
        
        # Frontend
        "HTML", "CSS", "SASS", "LESS", "Webpack", "Vite", "Babel", "ESLint", "Jest", "Mocha",
        "Jasmine", "Cypress", "Selenium", "Responsive Design", "UI/UX", "Material UI", "Tailwind CSS",
        
        # Sécurité
        "Cybersecurity", "Security", "Authentication", "OAuth", "JWT", "SAML", "SSO",
        "Encryption", "Penetration Testing", "Vulnerability Assessment", "Firewall",
        "WAF", "SIEM", "IAM", "GDPR", "HIPAA", "PCI DSS", "SOC 2",
        
        # ERP & CRM
        "SAP", "Oracle ERP", "Microsoft Dynamics", "Salesforce", "HubSpot", "Zoho", "NetSuite",
        
        # Gestion de projet & Collaboration
        "JIRA", "Confluence", "Trello", "Asana", "Monday", "Notion", "Slack", "Microsoft Teams",
        "Git", "SVN", "Mercurial", "BitBucket", "GitHub", "GitLab",
        
        # Autres
        "Data Science", "AI", "Artificial Intelligence", "IoT", "Internet of Things", "Blockchain",
        "AR", "VR", "Augmented Reality", "Virtual Reality", "Web3", "Quantum Computing",
        "Edge Computing", "Serverless", "5G", "Robotics", "Digital Transformation",
        "SaaS", "PaaS", "IaaS", "Cloud Native", "Multicloud", "Hybrid Cloud",
    ]
    
    # Ajout de compétences spécifiques qui apparaissent dans le CV exemple
    additional_skills = [
        # Mentionnés explicitement dans le CV exemple
        "ETL pipelines", "Apache Airflow", "Spark", "Hadoop", 
        "MongoDB", "Redis", "Elasticsearch", "PostgreSQL",
        "EC2", "S3", "Lambda", "Jira", "Confluence", "CI/CD", 
        "Jenkins", "Docker", "Kubernetes", "API",
        "RESTful", "Django", "FastAPI", "Flask", "Microservices"
    ]
    
    # Combiner les listes de compétences
    all_skills = list(set(common_skills + additional_skills))
    
    # Convertir le texte en minuscules pour la recherche
    text_lower = text.lower()
    
    # Trouver les compétences dans le texte
    found_skills = []
    for skill in all_skills:
        # Vérifier si la compétence est présente dans le texte (casse insensible)
        skill_lower = skill.lower()
        if skill_lower in text_lower:
            # Ajouter la compétence avec sa casse d'origine
            found_skills.append(skill)
            
    # S'il n'y a pas de compétences trouvées, ajouter au moins quelques compétences de base
    if not found_skills:
        found_skills = ["Data Science", "Agile", "Scrum"]
            
    return found_skills

def identify_domains(text, skills):
    """
    Identifie les domaines possibles à partir du texte et des compétences.
    
    Args:
        text: Texte à analyser
        skills: Liste de compétences
        
    Returns:
        Dictionnaire des domaines avec leurs scores
    """
    text_lower = text.lower()
    
    # Définir des mots-clés et termes spécifiques pour chaque domaine
    domain_keywords = {
        "informatique_reseaux": [
            # Développement logiciel et programmation
            "développeur", "développement", "software", "engineer", "programming", "code", "coder", "coding",
            "python", "java", "javascript", "typescript", "php", "ruby", "c++", "c#", ".net", "scala", "go", "golang",
            "html", "css", "react", "angular", "vue", "nodejs", "node.js", "django", "flask", "spring", "laravel",
            "frontend", "backend", "fullstack", "full-stack", "web", "application", "mobile", "app", "api", "rest",
            "mvc", "orm", "sql", "nosql", "object-oriented", "functional programming", "agile", "scrum",
            
            # Données et analyse
            "data", "database", "base de données", "big data", "analytics", "analyst", "analyse", "machine learning",
            "deep learning", "ia", "ai", "intelligence artificielle", "data science", "data scientist", "data engineer",
            "sql", "mysql", "postgresql", "mongodb", "cassandra", "neo4j", "redis", "elasticsearch", "hadoop", "spark",
            "tableau", "power bi", "qlik", "data mining", "data warehouse", "etl", "olap", "statistique", "statistics",
            
            # Infrastructure et opérations
            "devops", "sre", "system", "infrastructure", "architecture", "cloud", "aws", "azure", "gcp", "docker",
            "kubernetes", "container", "virtualisation", "vmware", "linux", "unix", "windows", "server", "serveur",
            "network", "réseau", "lan", "wan", "vpc", "firewall", "parefeu", "load balancer", "networking", "switch",
            "router", "routeur", "datacenter", "monitoring", "logging", "log", "backup", "sauvegarde", "disaster recovery",
            
            # Sécurité informatique
            "security", "sécurité", "cybersecurity", "cybersécurité", "pentest", "pen test", "ethical hacking",
            "vulnerability", "vulnérabilité", "encryption", "cryptage", "chiffrement", "authentication", "authorization",
            "firewall", "ips", "ids", "siem", "compliance", "gdpr", "rgpd", "audit", "risk assessment", "risk management",
            
            # Autres termes techniques
            "git", "github", "gitlab", "bitbucket", "ci/cd", "continuous integration", "continuous delivery",
            "jenkins", "travis", "microservices", "micro-services", "sdk", "api", "soap", "rest", "graphql",
            "json", "xml", "yaml", "protobuf", "web service", "http", "https", "tcp/ip", "dns", "tls", "ssl"
        ],
        
        "automatismes_info_industrielle": [
            # Automatismes et contrôle
            "automate", "automatisme", "automation", "contrôle", "control", "plc", "api", "ladder", "grafcet",
            "scada", "hmi", "ihm", "supervision", "dcs", "distributed control system", "instrumentation",
            "siemens", "schneider", "rockwell", "allen bradley", "abb", "omron", "ge fanuc", "mitsubishi",
            "step7", "tia portal", "unity", "concept", "rslogix", "factory talk", "wincc", "citect", "ignition",
            
            # Protocoles et communications industriels
            "modbus", "profibus", "profinet", "ethernet/ip", "devicenet", "canopen", "asi", "hart", "foundation fieldbus",
            "opc", "opc ua", "mqtt", "io-link", "bacnet", "lonworks", "knx", "m-bus", "dali", "zigbee", "lora", "nb-iot",
            
            # Robotique et équipements
            "robot", "robotique", "robotics", "cobot", "collaborative robot", "manipulateur", "abb", "kuka", "fanuc", 
            "stäubli", "yaskawa", "motoman", "universal robots", "rpa", "robotic process automation", "agv", "amr",
            
            # Ingénierie et conception
            "pid", "p&id", "schéma", "schema", "câblage", "cablage", "wiring", "panel", "armoire", "cabinet",
            "eplan", "see electrical", "autocad electrical", "solidworks electrical", "electrical engineering",
            "génie électrique", "ingénierie électrique", "pneuamtique", "pneumatic", "hydraulique", "hydraulic",
            
            # Électronique et électrotechnique
            "électronique", "electronique", "electronic", "électrotechnique", "électricité", "electricity",
            "vfd", "variable frequency drive", "variateur", "drive", "servo", "motor", "moteur", "inverter",
            "capteur", "sensor", "transmetteur", "transmitter", "actuator", "actionneur", "solenoid", "solénoïde",
            
            # Industrie 4.0 et IoT industriel
            "industrie 4.0", "industry 4.0", "iiot", "iot industriel", "industrial iot", "smart manufacturing",
            "usine intelligente", "smart factory", "jumeau numérique", "digital twin", "edge computing", "fog computing",
            "predictive maintenance", "maintenance prédictive", "condition monitoring", "manufacturing execution system",
            "mes", "erp", "manufacturing", "production"
        ],
        
        "finance": [
            # Comptabilité et audit
            "comptabilité", "comptable", "accounting", "audit", "comptes", "accounts", "bilan", "balance sheet",
            "compte de résultat", "income statement", "grand livre", "general ledger", "journal", "immobilisation",
            "fixed asset", "amortissement", "depreciation", "provision", "liasse fiscale", "tax return", "fiscal",
            "ifrs", "gaap", "consolidation", "reconciliation", "rapprochement", "clôture", "closing",
            
            # Finance d'entreprise
            "finance", "financial", "financier", "cfo", "daf", "directeur financier", "trésorerie", "treasury",
            "cash flow", "flux de trésorerie", "investment", "investissement", "financement", "funding", "budget",
            "budgétaire", "forecast", "prévision", "business plan", "plan d'affaires", "valuation", "valorisation",
            "profitability", "rentabilité", "roi", "return on investment", "wacc", "capm", "dcf", "npv", "van",
            
            # Contrôle de gestion
            "contrôle de gestion", "contrôleur de gestion", "controlling", "controller", "management control",
            "reporting", "tableau de bord", "dashboard", "kpi", "indicateur", "metric", "costing", "cost accounting",
            "comptabilité analytique", "abc", "activity based costing", "balanced scorecard", "variance analysis",
            "analyse d'écart", "budget control", "contrôle budgétaire", "forecast", "prévision", "planning",
            
            # Fiscalité et juridique
            "fiscalité", "fiscal", "tax", "impôt", "taxation", "cfe", "cvae", "is", "corporate tax", "vat", "tva",
            "déclaration", "filing", "compliance", "conformité", "tax optimization", "optimisation fiscale", "tax audit",
            "contrôle fiscal", "tax planning", "planification fiscale", "tax law", "droit fiscal", "transfer pricing",
            
            # Banque, marchés financiers et assurance
            "banque", "banking", "bank", "crédit", "credit", "loan", "prêt", "asset management", "gestion d'actifs",
            "portfolio", "portefeuille", "trading", "trader", "market", "marché", "stock", "action", "bond", "obligation",
            "derivative", "dérivé", "forex", "currency", "devise", "interest rate", "taux d'intérêt", "risk", "risque",
            "insurance", "assurance", "actuarial", "actuariat", "underwriting", "souscription", "claim", "sinistre",
            
            # Outils et systèmes
            "sap", "sap fi", "sap co", "oracle financials", "sage", "cegid", "quickbooks", "dynamics", "netsuite",
            "exact", "xero", "excel", "spreadsheet", "vba", "power bi", "tableau", "qlik", "bi", "business intelligence"
        ],
        
        "genie_civil_btp": [
            # Construction et BTP général
            "génie civil", "civil engineering", "btp", "construction", "bâtiment", "building", "travaux publics",
            "public works", "chantier", "site", "worksite", "ouvrage", "structure", "infrastructure", "superstructure",
            "construction manager", "conducteur de travaux", "chef de chantier", "site manager", "maître d'œuvre",
            "maître d'ouvrage", "contracting authority", "project owner", "project manager", "construction project",
            
            # Conception et études
            "conception", "design", "étude", "study", "plan", "drawing", "dessin", "blueprint", "spécification",
            "specification", "architecture", "architectural", "ingénierie", "engineering", "calcul", "calculation",
            "dimensionnement", "sizing", "modélisation", "modeling", "simulation", "analyse", "analysis", "conception",
            
            # Structures et matériaux
            "structure", "structural", "béton", "concrete", "béton armé", "reinforced concrete", "précontraint",
            "prestressed", "acier", "steel", "bois", "wood", "timber", "maçonnerie", "masonry", "fondation", "foundation",
            "dalle", "slab", "poutre", "beam", "colonne", "column", "poteau", "post", "mur", "wall", "charpente", "frame",
            "toiture", "roof", "plancher", "floor", "façade", "facade", "isolation", "insulation", "étanchéité", "waterproofing",
            
            # Routes et ouvrages
            "route", "road", "highway", "autoroute", "pont", "bridge", "tunnel", "barrage", "dam", "viaduc", "viaduct",
            "terrassement", "earthwork", "excavation", "remblai", "embankment", "déblai", "cutting", "assainissement",
            "drainage", "hydraulique", "hydraulic", "hydrologie", "hydrology", "géotechnique", "geotechnical",
            
            # Réseaux et VRD
            "vrd", "voirie", "roadway", "réseaux", "networks", "utilities", "eau", "water", "assainissement", "sewage",
            "plomberie", "plumbing", "électricité", "electrical", "hvac", "cvc", "chauffage", "heating", "ventilation",
            "climatisation", "air conditioning", "éclairage", "lighting", "fluide", "fluid", "réseau", "network",
            
            # Outils et méthodes
            "autocad", "revit", "bim", "building information modeling", "allplan", "archicad", "tekla", "sketchup",
            "robot", "etabs", "sap2000", "plaxis", "gestion de projet", "project management", "planning", "scheduling",
            "estimation", "estimating", "métré", "quantity surveying", "appel d'offres", "tender", "marché", "contract"
        ],
        
        "genie_industriel": [
            # Production et fabrication
            "génie industriel", "industrial engineering", "production", "manufacturing", "fabrication", "usine", "factory",
            "atelier", "workshop", "chaîne", "line", "assemblage", "assembly", "process", "procédé", "opération", "process",
            "fabrication", "fabrication", "manufacturier", "manufacturing", "usinage", "machining", "manufacturing execution system",
            
            # Méthodes et amélioration continue
            "méthode", "method", "industrialisation", "work study", "étude du travail", "temps", "time", "chronométrage",
            "time study", "mtm", "mouvement", "movement", "ergonomie", "ergonomics", "poste de travail", "workstation",
            "lean", "kaizen", "amélioration continue", "continuous improvement", "5s", "six sigma", "tqm", "total quality",
            "vsm", "value stream mapping", "smed", "tpm", "total productive maintenance", "poka yoke", "jidoka",
            
            # Qualité et normes
            "qualité", "quality", "qse", "hse", "qhse", "assurance qualité", "quality assurance", "contrôle qualité",
            "quality control", "inspection", "test", "norme", "standard", "iso", "iso 9001", "iso 14001", "iso 45001",
            "certification", "audit", "non-conformité", "non-conformance", "action corrective", "corrective action",
            "métrologie", "metrology", "étalonnage", "calibration", "measurement", "mesure", "tolérancing", "gd&t",
            
            # Supply chain et logistique
            "supply chain", "chaîne d'approvisionnement", "logistique", "logistics", "approvisionnement", "procurement",
            "purchasing", "achat", "fournisseur", "supplier", "stock", "inventory", "entreposage", "warehousing",
            "magasin", "warehouse", "distribution", "transport", "expédition", "shipping", "réception", "receiving",
            "planning", "planification", "scheduling", "ordonnancement", "mrp", "manufacturing resource planning",
            "erp", "enterprise resource planning", "demand planning", "s&op", "sales and operations planning",
            
            # Maintenance et fiabilité
            "maintenance", "entretien", "préventive", "preventive", "corrective", "curative", "conditionnelle",
            "condition-based", "prédictive", "predictive", "fiabilité", "reliability", "disponibilité", "availability",
            "mtbf", "mean time between failures", "mttr", "mean time to repair", "panne", "breakdown", "failure",
            "défaillance", "rcm", "reliability centered maintenance", "gmao", "cmms", "maintenance management",
            
            # Gestion de projet et management
            "gestion de projet", "project management", "chef de projet", "project manager", "ingénieur projet",
            "project engineer", "planning", "gantt", "pert", "wbs", "work breakdown structure", "milestone", "jalon",
            "budget", "cost", "coût", "risk", "risque", "management", "leadership", "team", "équipe", "coordination",
            "stakeholder", "partie prenante", "pilotage", "steering", "reporting", "kpi", "indicateur", "performance"
        ]
    }
    
    # Initialiser les scores de domaine
    domain_scores = {
        "informatique_reseaux": 0.0,
        "automatismes_info_industrielle": 0.0,
        "finance": 0.0,
        "genie_civil_btp": 0.0,
        "genie_industriel": 0.0
    }
    
    # Associer chaque compétence à son domaine principal
    skill_domain_map = {
        # Informatique et Réseaux
        "Python": "informatique_reseaux",
        "Java": "informatique_reseaux",
        "JavaScript": "informatique_reseaux",
        "React": "informatique_reseaux",
        "Angular": "informatique_reseaux",
        "Vue.js": "informatique_reseaux",
        "Node.js": "informatique_reseaux",
        "Django": "informatique_reseaux",
        "Flask": "informatique_reseaux",
        "FastAPI": "informatique_reseaux",
        "Spring": "informatique_reseaux",
        "DevOps": "informatique_reseaux",
        "AWS": "informatique_reseaux",
        "Azure": "informatique_reseaux",
        "GCP": "informatique_reseaux",
        "Docker": "informatique_reseaux",
        "Kubernetes": "informatique_reseaux",
        "Linux": "informatique_reseaux",
        "Git": "informatique_reseaux",
        "SQL": "informatique_reseaux",
        "NoSQL": "informatique_reseaux",
        "MongoDB": "informatique_reseaux",
        "PostgreSQL": "informatique_reseaux",
        "MySQL": "informatique_reseaux",
        "Redis": "informatique_reseaux",
        "Elasticsearch": "informatique_reseaux",
        "API": "informatique_reseaux",
        "REST": "informatique_reseaux",
        "GraphQL": "informatique_reseaux",
        "Microservices": "informatique_reseaux",
        "CI/CD": "informatique_reseaux",
        
        # Automatismes et Info Industrielle
        "PLC": "automatismes_info_industrielle",
        "SCADA": "automatismes_info_industrielle",
        "HMI": "automatismes_info_industrielle",
        "Siemens": "automatismes_info_industrielle",
        "Schneider": "automatismes_info_industrielle",
        "Allen Bradley": "automatismes_info_industrielle",
        "Rockwell": "automatismes_info_industrielle",
        "ABB": "automatismes_info_industrielle",
        "Automation": "automatismes_info_industrielle",
        "Modbus": "automatismes_info_industrielle",
        "Profinet": "automatismes_info_industrielle",
        "Profibus": "automatismes_info_industrielle",
        "TIA Portal": "automatismes_info_industrielle",
        "Step7": "automatismes_info_industrielle",
        "Unity Pro": "automatismes_info_industrielle",
        "Factory Talk": "automatismes_info_industrielle",
        "WinCC": "automatismes_info_industrielle",
        "Robot": "automatismes_info_industrielle",
        "Grafcet": "automatismes_info_industrielle",
        
        # Finance
        "SAP FI": "finance",
        "SAP CO": "finance",
        "Oracle Financials": "finance",
        "Sage": "finance",
        "Cegid": "finance",
        "Excel": "finance",
        "VBA": "finance",
        "Power BI": "finance",
        "Tableau": "finance",
        "IFRS": "finance",
        "US GAAP": "finance",
        "Consolidation": "finance",
        "Comptabilité": "finance",
        "Accounting": "finance",
        "Audit": "finance",
        "Controlling": "finance",
        
        # Génie Civil & BTP
        "AutoCAD": "genie_civil_btp",
        "Revit": "genie_civil_btp",
        "BIM": "genie_civil_btp",
        "SketchUp": "genie_civil_btp",
        "ArchiCAD": "genie_civil_btp",
        "CATIA": "genie_civil_btp",
        "Tekla": "genie_civil_btp",
        "Robot": "genie_civil_btp",
        "Abaqus": "genie_civil_btp",
        "ETABS": "genie_civil_btp",
        "SAP2000": "genie_civil_btp",
        "STAAD.Pro": "genie_civil_btp",
        "Plaxis": "genie_civil_btp",
        
        # Génie Industriel
        "Lean": "genie_industriel",
        "Six Sigma": "genie_industriel",
        "Kaizen": "genie_industriel",
        "5S": "genie_industriel",
        "SMED": "genie_industriel",
        "TPM": "genie_industriel",
        "TQM": "genie_industriel",
        "JIT": "genie_industriel",
        "Kanban": "genie_industriel",
        "ISO 9001": "genie_industriel",
        "ISO 14001": "genie_industriel",
        "ISO 45001": "genie_industriel",
        "GMAO": "genie_industriel",
        "GPAO": "genie_industriel",
        "ERP": "genie_industriel",
        "MRP": "genie_industriel",
        "SAP MM": "genie_industriel",
        "SAP PP": "genie_industriel"
    }
    
    # Analyser le texte pour les mots-clés de domaine
    for domain, keywords in domain_keywords.items():
        count = 0
        for keyword in keywords:
            if keyword.lower() in text_lower:
                count += 1
        
        # Calculer le score basé sur le nombre de mots-clés trouvés
        score_text = min(0.5, 0.1 + (count / len(keywords)) * 0.4)
        domain_scores[domain] += score_text
    
    # Analyser les compétences
    if skills:
        for skill in skills:
            # Chercher la compétence dans la carte des domaines
            if skill in skill_domain_map:
                domain = skill_domain_map[skill]
                domain_scores[domain] += 0.1  # Bonus par compétence spécifique
            
            # Chercher également dans les mots-clés de domaine
            for domain, keywords in domain_keywords.items():
                if skill.lower() in [k.lower() for k in keywords]:
                    domain_scores[domain] += 0.05  # Bonus plus petit si juste un mot-clé
    
    # Normaliser les scores (s'assurer qu'ils sont entre 0 et 1)
    for domain in domain_scores:
        domain_scores[domain] = min(1.0, domain_scores[domain])
    
    return domain_scores

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

def get_scraped_jobs_matching(cv_analysis, top_k=5, min_score=0.3):
    """
    Récupère et matche les offres d'emploi scrapées avec le CV analysé.
    
    Args:
        cv_analysis: Les résultats de l'analyse du CV
        top_k: Nombre maximum d'offres à retourner
        min_score: Score minimum pour considérer une correspondance
        
    Returns:
        Liste d'offres d'emploi correspondantes
    """
    # Récupérer toutes les offres scrapées
    all_jobs = []
    scraped_jobs_dir = os.path.join("data", "scraped_jobs")
    
    # Vérifier si le répertoire existe
    if not os.path.exists(scraped_jobs_dir):
        return []
    
    # Trouver tous les fichiers JSON d'offres
    job_files = glob.glob(os.path.join(scraped_jobs_dir, "*.json"))
    
    # S'il n'y a pas de fichiers, retourner une liste vide
    if not job_files:
        return []
    
    # Récupérer les offres de tous les fichiers
    for job_file in job_files:
        try:
            with open(job_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Différentes structures possibles selon la source
                if "jobs" in data:
                    # Format de lot (multiple offres)
                    all_jobs.extend(data["jobs"])
                elif "title" in data and "company" in data:
                    # Format d'offre unique
                    all_jobs.append(data)
            except Exception as e:
            print(f"Erreur lors de la lecture du fichier {job_file}: {str(e)}")
            continue
    
    # S'il n'y a pas d'offres, retourner une liste vide
    if not all_jobs:
        return []
    
    # Calculer le score de correspondance pour chaque offre
    cv_domain = cv_analysis.get("main_domain", "")
    cv_skills = cv_analysis.get("skills", [])
    
    matched_jobs = []
    for job in all_jobs:
        # Initialiser le score
        score = 0.1  # Score de base
        
        # Bonus pour le domaine correspondant
        job_domain = job.get("domain", "")
        domain_match = job_domain == cv_domain
        domain_bonus = 0.3 if domain_match else 0.0
        
        # Bonus pour les compétences correspondantes
        job_title = job.get("title", "").lower()
        job_description = job.get("description", "").lower()
        job_text = job_title + " " + job_description
        
        skills_matched = []
        for skill in cv_skills:
            if skill.lower() in job_text:
                skills_matched.append(skill)
        
        # Calcul du bonus de compétences (max 0.6)
        skills_bonus = 0
        if cv_skills:
            skills_bonus = min(0.6, (len(skills_matched) / len(cv_skills)) * 0.6)
        
        # Score final
        raw_score = score
        final_score = min(1.0, score + domain_bonus + skills_bonus)
        
        # Ajouter l'offre si elle dépasse le score minimum
        if final_score >= min_score:
            # Générer un ID unique si absent
            job_id = job.get("id")
            if not job_id:
                # Créer un hash MD5 basé sur le titre, l'entreprise et la description
                job_hash = hashlib.md5(f"{job_title}|{job.get('company', '')}|{job_description}".encode()).hexdigest()
                job_id = job_hash
            
            matched_job = {
                "id": job_id,
                "title": job.get("title", ""),
                "company": job.get("company", ""),
                "location": job.get("location", ""),
                "description": job.get("description", ""),
                "domain": job_domain,
                "url": job.get("url", ""),
                "source": job.get("source", "Scraper"),
                "raw_score": raw_score,
                "domain_bonus": domain_bonus,
                "skills_bonus": skills_bonus,
                "score": final_score,
                "domain_match": "✅" if domain_match else "❌",
                "matched_skills": len(skills_matched)
            }
            matched_jobs.append(matched_job)
    
    # Trier par score décroissant
    matched_jobs.sort(key=lambda x: x["score"], reverse=True)
    
    # Limiter au nombre demandé
    return matched_jobs[:top_k]

def improved_cv_matching(cv_path=None, cv_text=None, api_url=DEFAULT_API_URL, llm_url=DEFAULT_LLM_URL, 
                       top_k=5, min_score=0.3, strict_threshold=False, verbose=False):
    """
    Version améliorée du matching de CV avec les offres d'emploi.
    
    Args:
        cv_path: Chemin vers le fichier CV (PDF ou TXT)
        cv_text: Texte du CV (alternative à cv_path)
        api_url: URL de l'API backend
        llm_url: URL du service LLM
        top_k: Nombre maximum d'offres à retourner
        min_score: Score minimum pour considérer une correspondance
        strict_threshold: Appliquer un seuil strict pour la pertinence
        verbose: Afficher des informations détaillées
        
    Returns:
        Résultats du matching incluant l'analyse du CV et les offres correspondantes
    """
    if verbose:
        print(f"Analyse du CV et recherche d'offres correspondantes...")
    
    # Vérifier les paramètres
    if not cv_path and not cv_text:
        raise ValueError("Vous devez fournir soit un chemin de fichier CV, soit le texte du CV")
    
    # Récupérer le contenu du CV
    if cv_path and not cv_text:
        cv_text = safe_read_cv_file(cv_path)
    
    # Analyser le CV
    cv_analysis = analyze_cv(cv_text, api_url=api_url, llm_url=llm_url)
    
    # Préparer la requête au backend pour le matching avec la base vectorielle
    results = None
    try:
        results = search_job_matches(cv_analysis, api_url=api_url, top_k=top_k, min_score=min_score)
                except Exception as e:
        if verbose:
            print(f"Erreur lors de la recherche de correspondances vectorielles: {str(e)}")
    
    # Récupérer aussi les offres scrapées
    scraped_jobs = get_scraped_jobs_matching(cv_analysis, top_k=top_k, min_score=min_score)
    
    # Si aucun résultat de la base vectorielle, utiliser seulement les offres scrapées
    if not results or "results" not in results or not results["results"]:
        if verbose:
            print("Aucun résultat de la base vectorielle, utilisation des offres scrapées uniquement.")
        
        # Formater les résultats comme attendu par l'interface
        formatted_results = {
            "cv_analysis": cv_analysis,
            "timestamp": datetime.now().isoformat(),
            "results": []
        }
        
        # Ajouter les offres scrapées
        for i, job in enumerate(scraped_jobs):
            job["rank"] = i + 1
            formatted_results["results"].append(job)
        
        return formatted_results
    
    # Mélanger les résultats de la base vectorielle et les offres scrapées
    vector_jobs = results["results"]
    
    # Déduplication par ID
    seen_ids = set()
    for job in vector_jobs:
        if "id" in job:
            seen_ids.add(job["id"])
    
    # Ajouter les offres scrapées non dupliquées
    for job in scraped_jobs:
        if job["id"] not in seen_ids:
            vector_jobs.append(job)
            seen_ids.add(job["id"])
    
    # Retrier par score
    vector_jobs.sort(key=lambda x: x["score"], reverse=True)
    
    # Réassigner les rangs
    for i, job in enumerate(vector_jobs):
        job["rank"] = i + 1
    
    # Limiter au nombre demandé
    results["results"] = vector_jobs[:top_k]
    
    return results

def search_job_offers_in_domain(domain: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Recherche des offres d'emploi dans un domaine spécifique en utilisant des fichiers locaux"""
    # ... existing code ...
    return formatted_results

async def store_scraped_jobs_in_vector_db(api_url: str, max_jobs: int = 0, domain_override: str = None) -> Tuple[int, int]:
    """
    Stocke les offres scrapées dans la base de données vectorielle.
    
    Args:
        api_url: URL de l'API backend
        max_jobs: Nombre maximum d'offres à stocker (0 = toutes)
        domain_override: Si spécifié, force ce domaine pour toutes les offres
        
    Returns:
        Tuple contenant (nombre d'offres traitées, nombre d'offres stockées)
    """
    print(f"Stockage des offres scrapées dans la base vectorielle...")
    
    # Répertoire des offres scrapées
    scraped_jobs_dir = os.path.join("data", "scraped_jobs")
    
    if not os.path.exists(scraped_jobs_dir):
        print(f"Répertoire {scraped_jobs_dir} inexistant")
        return 0, 0
            
        # Récupérer tous les fichiers JSON
    job_files = glob.glob(os.path.join(scraped_jobs_dir, "*.json"))
    
    if not job_files:
        print("Aucun fichier d'offres scrapées trouvé")
        return 0, 0
    
    # Charger toutes les offres
        all_jobs = []
    for job_file in job_files:
            try:
            with open(job_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Gérer différents formats
                if "jobs" in data:
                    # Format de lot (multiple offres)
                    all_jobs.extend(data["jobs"])
                elif "title" in data and "company" in data:
                    # Format d'offre unique
                    all_jobs.append(data)
            except Exception as e:
            print(f"Erreur lors de la lecture du fichier {job_file}: {str(e)}")
            continue
        
    # Limiter le nombre d'offres si nécessaire
        if max_jobs > 0 and len(all_jobs) > max_jobs:
            all_jobs = all_jobs[:max_jobs]
            
    total_jobs = len(all_jobs)
    if total_jobs == 0:
        print("Aucune offre à traiter")
        return 0, 0
    
    print(f"Préparation de {total_jobs} offres pour stockage...")
    
    # Préparer les offres pour l'envoi
    job_batch = []
    stored_count = 0
    
    # Créer des sessions HTTP asynchrone
    import aiohttp
    async with aiohttp.ClientSession() as session:
            for job in all_jobs:
            # Vérifier si l'offre contient les champs nécessaires
            if not job.get("title") or not job.get("description"):
                print(f"Offre incomplète ignorée: {job.get('title', 'Sans titre')}")
                continue
            
            # Déterminer le domaine
            job_domain = domain_override if domain_override else job.get("domain")
            
            # Si pas de domaine ou domaine inconnu, le déterminer à partir du contenu
            if not job_domain or job_domain not in ["informatique_reseaux", "automatismes_info_industrielle", 
                                                   "finance", "genie_civil_btp", "genie_industriel"]:
                # Extraire les compétences du texte
                job_text = job.get("title", "") + " " + job.get("description", "")
                skills = extract_skills_from_text(job_text)
                
                # Identifier le domaine
                domains = identify_domains(job_text, skills)
                if domains:
                    # Prendre le domaine avec le score le plus élevé
                    job_domain = max(domains.items(), key=lambda x: x[1])[0]
                        else:
                    # Domaine par défaut
                    job_domain = "informatique_reseaux"
            
            # Créer une offre formatée
            formatted_job = {
                "id": job.get("id", ""),  # Si vide, l'API générera un ID
                "title": job.get("title", ""),
                "company": job.get("company", ""),
                "location": job.get("location", ""),
                "description": job.get("description", ""),
                "domain": job_domain,
                "country": job.get("country", ""),
                "url": job.get("url", ""),
                "salary": job.get("salary", ""),
                "skills": job.get("skills", []),
                "source": job.get("source", "Scraper"),
                "date_added": datetime.now().isoformat()
            }
            
            # Ajouter à l'offre au lot
            job_batch.append(formatted_job)
            
            # Traiter par lots de 10 offres
            if len(job_batch) >= 10:
                try:
                    async with session.post(
                        f"{api_url}/store_job_offers",
                        json={"jobs": job_batch},
                        timeout=30
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            stored_count += result.get("stored", 0)
                            print(f"Lot de {len(job_batch)} offres stocké avec succès")
                        else:
                            print(f"Erreur lors du stockage du lot: {response.status}")
                            error_text = await response.text()
                            print(f"Message d'erreur: {error_text}")
    except Exception as e:
                    print(f"Exception lors du stockage des offres: {str(e)}")
                
                # Vider le lot
                job_batch = []
        
        # Traiter les offres restantes
        if job_batch:
            try:
                async with session.post(
                    f"{api_url}/store_job_offers",
                    json={"jobs": job_batch},
                    timeout=30
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        stored_count += result.get("stored", 0)
                        print(f"Lot final de {len(job_batch)} offres stocké avec succès")
                else:
                        print(f"Erreur lors du stockage du lot final: {response.status}")
                        error_text = await response.text()
                        print(f"Message d'erreur: {error_text}")
            except Exception as e:
                print(f"Exception lors du stockage des offres restantes: {str(e)}")
    
    print(f"Traitement terminé: {stored_count}/{total_jobs} offres stockées dans la base vectorielle")
    return total_jobs, stored_count

async def store_indeed_jobs_in_vector_db(api_url: str, indeed_jobs: List[Dict[str, Any]]) -> List[str]:
    """Store Indeed jobs in the vector database for future matching"""
    # ... existing code ...

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