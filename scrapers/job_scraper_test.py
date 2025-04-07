#!/usr/bin/env python
# Script de test pour simuler la récupération d'offres d'emploi

import argparse
import json
import os
import asyncio
import httpx
from typing import List, Dict, Any, Optional
from datetime import datetime
import random

# Offres d'emploi simulées par domaine
SIMULATED_JOBS = {
    "informatique_reseaux": [
        {
            "title": "DevOps Engineer Senior",
            "company": "TechCloud Solutions",
            "location": "Paris, France",
            "description": """
Nous recherchons un ingénieur DevOps expérimenté pour rejoindre notre équipe.

Responsabilités:
- Concevoir, développer et maintenir notre infrastructure cloud sur AWS
- Automatiser les processus de CI/CD avec Jenkins, GitLab CI
- Gérer notre infrastructure conteneurisée (Docker, Kubernetes)
- Mettre en place et maintenir des outils de monitoring et de logging
- Collaborer avec les équipes de développement pour améliorer les performances

Compétences requises:
- 3+ ans d'expérience en tant qu'ingénieur DevOps
- Maîtrise de Linux, des conteneurs et de Kubernetes
- Expérience avec AWS, Terraform et IaC
- Connaissances en Python, Bash et autres langages de scripting
- Expérience avec les outils de monitoring (Prometheus, Grafana)
            """
        },
        {
            "title": "Développeur Full Stack JavaScript",
            "company": "WebTech Agency",
            "location": "Lyon, France",
            "description": """
Nous recherchons un développeur Full Stack JavaScript pour nos projets web innovants.

Responsabilités:
- Développer des applications web réactives avec React et Node.js
- Collaborer avec les designers UX/UI pour implémenter les interfaces
- Participer à l'architecture technique des projets
- Mettre en place et maintenir des API RESTful
- Assurer la qualité du code par des tests automatisés

Compétences requises:
- 2+ ans d'expérience en développement web Full Stack
- Maîtrise de JavaScript, React, Node.js, Express
- Expérience avec les bases de données SQL et NoSQL
- Connaissances en HTML5/CSS3 et responsive design
- Familiarité avec Git et les méthodologies Agile
            """
        }
    ],
    "automatismes_info_industrielle": [
        {
            "title": "Ingénieur Automaticien SCADA",
            "company": "Industrial Automation Systems",
            "location": "Grenoble, France",
            "description": """
Nous recherchons un ingénieur automaticien spécialisé en SCADA pour notre département d'automatisation industrielle.

Responsabilités:
- Concevoir et développer des systèmes de supervision industrielle
- Programmer des automates et interfaces homme-machine
- Intégrer des systèmes SCADA et MES
- Participer aux phases de tests et de mise en service
- Assurer la maintenance et le support des installations

Compétences requises:
- 3+ ans d'expérience en automatisme industriel
- Maîtrise des environnements Siemens (TIA Portal, WinCC)
- Expérience avec les protocoles industriels (Profinet, Modbus)
- Connaissances en réseaux industriels et cybersécurité OT
- Capacité à lire et interpréter des schémas électriques
            """
        },
        {
            "title": "Technicien Maintenance Automatisme",
            "company": "FactoryTech",
            "location": "Marseille, France",
            "description": """
Nous recherchons un technicien maintenance spécialisé en automatisme pour notre usine de production.

Responsabilités:
- Assurer la maintenance préventive et curative des installations automatisées
- Diagnostiquer et résoudre les pannes sur les systèmes automatisés
- Participer aux projets d'amélioration des équipements
- Programmer et modifier les automates programmables
- Assurer la documentation technique des interventions

Compétences requises:
- 2+ ans d'expérience en maintenance industrielle
- Maîtrise des automates Schneider et Siemens
- Connaissances en électricité industrielle et pneumatique
- Expérience en dépannage de systèmes automatisés
- Habilitations électriques à jour
            """
        }
    ],
    "finance": [
        {
            "title": "Analyste Financier Corporate",
            "company": "Finance Global",
            "location": "Paris, France",
            "description": """
Nous recherchons un analyste financier pour renforcer notre équipe Corporate Finance.

Responsabilités:
- Analyser les données financières et préparer des rapports détaillés
- Développer des modèles financiers et des prévisions budgétaires
- Effectuer des analyses de rentabilité et des évaluations de projets
- Suivre les performances financières et identifier les opportunités
- Présenter des recommandations à la direction

Compétences requises:
- 3+ ans d'expérience en analyse financière
- Maîtrise d'Excel et des outils d'analyse financière
- Expérience avec SAP Finance ou systèmes similaires
- Diplôme en finance, comptabilité ou domaine connexe
- Certification CFA ou équivalent préférée
            """
        },
        {
            "title": "Contrôleur de Gestion",
            "company": "IndustrieGroup",
            "location": "Nantes, France",
            "description": """
Nous recherchons un contrôleur de gestion pour piloter la performance financière de nos activités.

Responsabilités:
- Élaborer et suivre les budgets et les forecasts
- Réaliser le reporting mensuel et les analyses d'écarts
- Mettre en place et suivre les indicateurs de performance (KPIs)
- Participer à l'optimisation des processus et à la réduction des coûts
- Supporter les opérationnels dans leurs prises de décision

Compétences requises:
- 3+ ans d'expérience en contrôle de gestion industriel
- Maîtrise des outils de reporting et de BI
- Expérience avec SAP et Excel avancé
- Formation supérieure en finance/contrôle de gestion
- Capacité d'analyse et esprit de synthèse
            """
        }
    ],
    "genie_civil_btp": [
        {
            "title": "Ingénieur Structure Senior",
            "company": "BuildTech Engineering",
            "location": "Lyon, France",
            "description": """
Nous recherchons un ingénieur structure expérimenté pour nos projets de construction.

Responsabilités:
- Concevoir et dimensionner des structures en béton armé et métal
- Réaliser des calculs de structures complexes
- Produire des plans d'exécution détaillés
- Suivre les chantiers et apporter un support technique
- Assurer la conformité aux normes et réglementations (Eurocodes)

Compétences requises:
- 5+ ans d'expérience en structures de bâtiment
- Maîtrise des logiciels de calcul (Robot, ETABS)
- Connaissance approfondie des Eurocodes
- Expérience en conception parasismique
- Diplôme d'ingénieur en génie civil/structure
            """
        },
        {
            "title": "Conducteur de Travaux BTP",
            "company": "Construction Générale",
            "location": "Bordeaux, France",
            "description": """
Nous recherchons un conducteur de travaux pour nos chantiers de bâtiments résidentiels.

Responsabilités:
- Gérer l'exécution des travaux et le planning des chantiers
- Coordonner les équipes et les sous-traitants
- Assurer le respect des budgets, délais et qualité
- Veiller à l'application des règles de sécurité
- Gérer la relation client et les réunions de chantier

Compétences requises:
- 3+ ans d'expérience en conduite de travaux
- Maîtrise de la lecture de plans et des techniques de construction
- Connaissances des normes de construction et de sécurité
- Capacité à gérer plusieurs chantiers simultanément
- Formation en génie civil ou équivalent
            """
        }
    ],
    "genie_industriel": [
        {
            "title": "Responsable Production Industrielle",
            "company": "Manufacturing Excellence",
            "location": "Lille, France",
            "description": """
Nous recherchons un responsable de production pour notre usine de fabrication.

Responsabilités:
- Superviser l'ensemble des opérations de production
- Optimiser les processus de fabrication et la chaîne logistique
- Mettre en œuvre des méthodes Lean Manufacturing
- Gérer les équipes de production (30+ personnes)
- Assurer le respect des normes de qualité et de sécurité

Compétences requises:
- 5+ ans d'expérience en gestion de production industrielle
- Maîtrise des méthodes Lean, Six Sigma
- Expérience avec les systèmes MES et ERP
- Solides compétences en leadership et gestion d'équipe
- Connaissance des normes ISO 9001, 14001
            """
        },
        {
            "title": "Ingénieur Méthodes Industrielles",
            "company": "TechIndustry",
            "location": "Toulouse, France",
            "description": """
Nous recherchons un ingénieur méthodes pour améliorer nos processus de production.

Responsabilités:
- Analyser et optimiser les processus de fabrication
- Développer et mettre en place des gammes de fabrication
- Réaliser des études de temps et de coûts
- Participer aux projets d'amélioration continue
- Proposer et implémenter des solutions d'automatisation

Compétences requises:
- 3+ ans d'expérience en méthodes industrielles
- Maîtrise des outils d'analyse de processus
- Expérience en AMDEC, 5S, SMED
- Connaissances en conception mécanique et CAO
- Formation d'ingénieur en génie industriel
            """
        }
    ]
}

# Configuration par défaut
DEFAULT_API_URL = "http://localhost:8000/api/v1"
DEFAULT_LLM_URL = "http://localhost:8001"

def parse_args():
    parser = argparse.ArgumentParser(description="Simuler la récupération d'offres d'emploi et les indexer dans notre base vectorielle")
    parser.add_argument("--query", type=str, help="Requête de recherche (par exemple 'devops', 'finance', etc.)")
    parser.add_argument("--domain", type=str, choices=list(SIMULATED_JOBS.keys()), 
                        help="Domaine spécifique à cibler")
    parser.add_argument("--output", type=str, default="data/simulated_job_offers.json", 
                        help="Fichier de sortie (JSON ou CSV)")
    parser.add_argument("--max_results", type=int, default=5, 
                        help="Nombre maximum d'offres à générer")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="URL de l'API backend")
    parser.add_argument("--llm-url", default=DEFAULT_LLM_URL, help="URL du service LLM")
    parser.add_argument("--add-to-db", action="store_true", help="Ajouter les offres à la base vectorielle")
    return parser.parse_args()

def fetch_simulated_jobs(query: str = None, domain: str = None, max_results: int = 5) -> List[Dict[str, Any]]:
    """Génère des offres d'emploi simulées en fonction du domaine ou de la requête"""
    job_offers = []
    
    # Si un domaine est spécifié, n'utiliser que les offres de ce domaine
    if domain and domain in SIMULATED_JOBS:
        source_jobs = SIMULATED_JOBS[domain]
        # Si on a besoin de plus d'offres que disponibles, dupliquer avec des variations
        if len(source_jobs) < max_results:
            additional_needed = max_results - len(source_jobs)
            for i in range(additional_needed):
                base_job = random.choice(source_jobs).copy()
                base_job["title"] = f"{base_job['title']} (Variante {i+1})"
                base_job["company"] = f"{base_job['company']} {chr(65+i)}"
                source_jobs.append(base_job)
        
        # Prendre le nombre demandé
        job_offers = source_jobs[:max_results]
    
    # Si une requête est spécifiée, rechercher dans tous les domaines
    elif query:
        query = query.lower()
        matching_jobs = []
        
        # Parcourir toutes les offres et trouver celles qui correspondent à la requête
        for domain, jobs in SIMULATED_JOBS.items():
            for job in jobs:
                if (query in job["title"].lower() or 
                    query in job["company"].lower() or 
                    query in job["description"].lower()):
                    job_copy = job.copy()
                    job_copy["domain"] = domain  # Ajouter le domaine
                    matching_jobs.append(job_copy)
        
        # Si suffisamment d'offres correspondantes trouvées
        if matching_jobs:
            random.shuffle(matching_jobs)  # Mélanger pour la diversité
            job_offers = matching_jobs[:max_results]
        else:
            # Si aucune correspondance, prendre des offres aléatoires
            all_jobs = []
            for domain, jobs in SIMULATED_JOBS.items():
                for job in jobs:
                    job_copy = job.copy()
                    job_copy["domain"] = domain
                    all_jobs.append(job_copy)
            
            random.shuffle(all_jobs)
            job_offers = all_jobs[:max_results]
    
    # Si ni domaine ni requête, prendre un échantillon aléatoire
    else:
        all_jobs = []
        for domain, jobs in SIMULATED_JOBS.items():
            for job in jobs:
                job_copy = job.copy()
                job_copy["domain"] = domain
                all_jobs.append(job_copy)
        
        random.shuffle(all_jobs)
        job_offers = all_jobs[:max_results]
    
    # Ajouter des métadonnées
    for job in job_offers:
        if "domain" not in job and domain:
            job["domain"] = domain
        job["date_collected"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        job["source"] = "Simulation"
    
    print(f"{len(job_offers)} offres simulées générées.")
    return job_offers

def save_offers_to_file(job_offers: List[Dict[str, Any]], output_path: str):
    """Stocke les offres dans un fichier JSON ou CSV"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if output_path.endswith('.json'):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(job_offers, f, ensure_ascii=False, indent=4)
        print(f"Offres sauvegardées dans {output_path} (JSON)")
    
    elif output_path.endswith('.csv'):
        import pandas as pd
        df = pd.DataFrame(job_offers)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Offres sauvegardées dans {output_path} (CSV)")
    
    else:
        print("Format de fichier non supporté. Utilisez .json ou .csv")

async def add_job_to_vector_db(api_url: str, llm_url: str, job: Dict[str, Any]) -> bool:
    """Ajouter une offre d'emploi à la base de vecteurs"""
    try:
        # Préparer les données de l'offre
        job_data = {
            "content": job["description"],
            "metadata": {
                "title": job["title"],
                "company": job["company"],
                "location": job["location"],
                "domain": job["domain"],
                "source": job["source"],
                "date_collected": job["date_collected"]
            }
        }
        
        # Envoyer l'offre au backend
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{api_url}/upload/job",
                json=job_data
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Offre ajoutée avec succès: {job['title']} (ID: {result.get('id', 'N/A')})")
                return True
            else:
                print(f"❌ Erreur lors de l'ajout de l'offre {job['title']}: {response.status_code}")
                print(f"Détails: {response.text}")
                return False
    except Exception as e:
        print(f"Exception lors de l'ajout de l'offre {job['title']}: {str(e)}")
        return False

async def add_jobs_to_db(api_url: str, llm_url: str, job_offers: List[Dict[str, Any]]):
    """Ajouter toutes les offres d'emploi à la base de vecteurs"""
    print(f"Ajout de {len(job_offers)} offres d'emploi à la base de vecteurs...")
    print(f"URL API: {api_url}")
    print(f"URL LLM: {llm_url}")
    
    success_count = 0
    for job in job_offers:
        if await add_job_to_vector_db(api_url, llm_url, job):
            success_count += 1
    
    print(f"\nRésumé: {success_count}/{len(job_offers)} offres ajoutées avec succès")

async def main_async(args):
    """Version asynchrone de la fonction principale"""
    # Générer les offres simulées
    job_offers = fetch_simulated_jobs(args.query, args.domain, args.max_results)
    
    if not job_offers:
        print("Aucune offre trouvée.")
        return
    
    # Sauvegarder les offres
    save_offers_to_file(job_offers, args.output)
    
    # Ajouter à la base vectorielle si demandé
    if args.add_to_db:
        await add_jobs_to_db(args.api_url, args.llm_url, job_offers)

def main():
    args = parse_args()
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main() 