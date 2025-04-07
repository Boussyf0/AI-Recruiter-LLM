#!/usr/bin/env python
import httpx
import asyncio
import json
import argparse
from typing import List, Dict, Any

# Offres d'emploi test pour les 5 domaines
TEST_JOBS = [
    {
        "title": "Ingénieur DevOps",
        "company": "TechCloud Solutions",
        "domain": "informatique_reseaux",
        "description": """
Nous recherchons un ingénieur DevOps expérimenté pour rejoindre notre équipe.

Responsabilités:
- Concevoir, développer et maintenir notre infrastructure cloud sur AWS
- Automatiser les processus de CI/CD avec Jenkins, GitLab CI et GitHub Actions
- Gérer notre infrastructure conteneurisée (Docker, Kubernetes)
- Mettre en place et maintenir des outils de monitoring et de logging
- Collaborer avec les équipes de développement pour améliorer les performances

Compétences requises:
- 3+ ans d'expérience en tant qu'ingénieur DevOps ou SRE
- Maîtrise de Linux, des conteneurs et de Kubernetes
- Expérience avec AWS, Terraform et IaC
- Connaissances en Python, Bash et autres langages de scripting
- Expérience avec les outils de monitoring (Prometheus, Grafana)
- Bonne compréhension des pratiques de sécurité
"""
    },
    {
        "title": "Analyste Financier Senior",
        "company": "Finance Global",
        "domain": "finance",
        "description": """
Nous recherchons un analyste financier senior pour renforcer notre équipe.

Responsabilités:
- Analyser les données financières et préparer des rapports détaillés
- Développer des modèles financiers et des prévisions budgétaires
- Effectuer des analyses de rentabilité et des évaluations de projets
- Suivre les performances financières et identifier les opportunités d'amélioration
- Présenter des recommandations à la direction

Compétences requises:
- 5+ ans d'expérience en analyse financière
- Maîtrise de Excel et des outils d'analyse financière
- Expérience avec SAP, Oracle Financials ou systèmes similaires
- Diplôme en finance, comptabilité ou domaine connexe
- Certification CFA ou équivalent préférée
- Excellentes compétences analytiques et attention aux détails
"""
    },
    {
        "title": "Ingénieur Automaticien",
        "company": "Industrial Automation Systems",
        "domain": "automatismes_info_industrielle",
        "description": """
Nous recherchons un ingénieur automaticien pour notre département d'automatisation industrielle.

Responsabilités:
- Concevoir et développer des systèmes de contrôle automatisés
- Programmer des automates (API) et des IHM
- Intégrer des systèmes SCADA et MES
- Participer aux phases de tests et de mise en service
- Assurer la maintenance et le support des installations

Compétences requises:
- 3+ ans d'expérience en automatisme industriel
- Maîtrise des environnements Siemens (TIA Portal, Step 7) et Schneider
- Expérience avec les protocoles de communication industriels (Profinet, Modbus, etc.)
- Connaissances en réseaux industriels et cybersécurité OT
- Capacité à lire et interpréter des schémas électriques
- Expérience en programmation (C/C++, Python ou équivalent)
"""
    },
    {
        "title": "Ingénieur Structure BTP",
        "company": "BuildTech Engineering",
        "domain": "genie_civil_btp",
        "description": """
Nous recherchons un ingénieur structure pour nos projets de construction.

Responsabilités:
- Concevoir et dimensionner des structures en béton armé, métal et bois
- Réaliser des calculs de structures et produire des plans d'exécution
- Suivre les chantiers et apporter un support technique
- Assurer la conformité aux normes et réglementations
- Collaborer avec les architectes et autres corps de métier

Compétences requises:
- 4+ ans d'expérience en génie civil/structures
- Maîtrise des logiciels de calcul de structures (Robot, ETABS, SAP2000)
- Connaissance approfondie des Eurocodes et normes de construction
- Expérience en conception parasismique
- Diplôme d'ingénieur en génie civil/structure
- Capacité à gérer plusieurs projets simultanément
"""
    },
    {
        "title": "Responsable Production Industrielle",
        "company": "Manufacturing Excellence",
        "domain": "genie_industriel",
        "description": """
Nous recherchons un responsable de production pour notre usine de fabrication.

Responsabilités:
- Superviser l'ensemble des opérations de production
- Optimiser les processus de fabrication et la chaîne logistique
- Mettre en œuvre des méthodes Lean Manufacturing
- Gérer les équipes de production (30+ personnes)
- Assurer le respect des normes de qualité et de sécurité
- Analyser les KPIs et proposer des améliorations continues

Compétences requises:
- 5+ ans d'expérience en gestion de production industrielle
- Maîtrise des méthodes Lean, Six Sigma, SMED
- Expérience avec les systèmes MES et ERP
- Solides compétences en leadership et gestion d'équipe
- Connaissance des normes ISO 9001, 14001
- Capacité à gérer plusieurs priorités dans un environnement dynamique
"""
    }
]

async def get_llm_embeddings(llm_url: str, text: str, domain: str = None) -> List[float]:
    """Obtenir les embeddings d'un texte depuis le service LLM"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{llm_url}/embeddings",
                json={"text": text, "domain": domain}
            )
            if response.status_code == 200:
                result = response.json()
                return result.get("embeddings", [])
            else:
                print(f"Erreur lors de la récupération des embeddings: {response.status_code}")
                print(f"Détails: {response.text}")
                return []
    except Exception as e:
        print(f"Exception lors de la récupération des embeddings: {str(e)}")
        return []

async def add_job_to_vector_db(api_url: str, llm_url: str, job: Dict[str, Any]) -> bool:
    """Ajouter une offre d'emploi à la base de vecteurs"""
    try:
        # Obtenir les embeddings pour la description de l'offre
        embeddings = await get_llm_embeddings(llm_url, job["description"], job["domain"])
        if not embeddings:
            print(f"Impossible d'obtenir des embeddings pour l'offre: {job['title']}")
            return False
        
        # Préparer les données de l'offre
        job_data = {
            "content": job["description"],
            "metadata": {
                "title": job["title"],
                "company": job["company"],
                "domain": job["domain"]
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

async def add_test_jobs(api_url: str, llm_url: str):
    """Ajouter toutes les offres d'emploi test"""
    print(f"Ajout de {len(TEST_JOBS)} offres d'emploi test à la base de vecteurs...")
    print(f"URL API: {api_url}")
    print(f"URL LLM: {llm_url}")
    
    success_count = 0
    for job in TEST_JOBS:
        if await add_job_to_vector_db(api_url, llm_url, job):
            success_count += 1
    
    print(f"\nRésumé: {success_count}/{len(TEST_JOBS)} offres ajoutées avec succès")

def main():
    parser = argparse.ArgumentParser(description="Ajouter des offres d'emploi test dans la base de vecteurs")
    parser.add_argument("--api-url", default="http://localhost:8000/api/v1", help="URL de l'API backend")
    parser.add_argument("--llm-url", default="http://localhost:8001", help="URL du service LLM")
    
    args = parser.parse_args()
    
    # Exécuter la tâche asynchrone
    asyncio.run(add_test_jobs(args.api_url, args.llm_url))

if __name__ == "__main__":
    main() 