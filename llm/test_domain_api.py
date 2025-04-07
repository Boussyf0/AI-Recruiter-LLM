#!/usr/bin/env python
# Script pour tester l'API de génération spécifique par domaine

import requests
import json
import argparse
from typing import Dict, List, Any
import os
from colorama import Fore, Style, init

# Initialiser colorama pour les couleurs dans le terminal
init()

# URL par défaut de l'API
DEFAULT_API_URL = "http://localhost:8001"

# Exemples de requêtes pour chaque domaine
DOMAIN_TEST_EXAMPLES = {
    "informatique_reseaux": {
        "cv_evaluation": {
            "messages": [
                {"role": "user", "content": "Peux-tu analyser ce CV pour un poste de développeur fullstack?"}
            ],
            "resume_content": """
Alice Martin
Développeuse Web Full Stack - 4 ans d'expérience

Compétences:
- Frontend: JavaScript, React, Vue.js, HTML5, CSS3, Sass
- Backend: Node.js, Express, Python, Django, PHP
- Bases de données: MongoDB, PostgreSQL, MySQL
- DevOps: Docker, CI/CD, Git, GitHub Actions
- Tests: Jest, Cypress, pytest

Expérience:
- Développeuse Full Stack chez TechSolutions (2021-présent)
  * Développement d'applications web responsive pour clients B2B
  * Migration d'applications monolithiques vers une architecture microservices
  * Mise en place de pipelines CI/CD automatisés

- Développeuse Frontend chez WebAgency (2019-2021)
  * Création d'interfaces utilisateur pour sites e-commerce
  * Optimisation des performances frontend (WebVitals, Lighthouse)

Formation:
- Master en Développement Web, Université de Lyon (2019)
- Licence en Informatique, Université de Paris (2017)
            """
        },
        "interview": {
            "messages": [
                {"role": "user", "content": "Je dois faire passer un entretien pour un poste d'administrateur système Linux. Quelles questions techniques dois-je poser?"}
            ]
        }
    },
    "automatismes_info_industrielle": {
        "cv_evaluation": {
            "messages": [
                {"role": "user", "content": "Analyse ce CV pour un poste d'ingénieur en automatismes industriels"}
            ],
            "resume_content": """
Marc Dupont
Ingénieur Automaticien - 7 ans d'expérience

Compétences:
- Programmation automates: Siemens (S7-300/400/1200/1500), TIA Portal, Schneider, Allen Bradley
- Supervision: WinCC, Wonderware, FactoryTalk View
- Réseaux industriels: Profinet, Profibus, Modbus TCP/IP, EtherNet/IP
- Langages: SCL, LADDER, SFC, FBD, C++
- Gestion de projets industriels & Documentation technique

Expérience:
- Ingénieur Automaticien Senior chez PharmaProcess (2019-présent)
  * Conception et mise en oeuvre de systèmes de contrôle pour lignes de production pharmaceutiques
  * Validation de systèmes automatisés selon normes FDA/GMP
  * Migration de systèmes S7-300 vers S7-1500

- Ingénieur Automaticien chez AutoControl (2016-2019)
  * Développement de solutions SCADA pour l'industrie automobile
  * Intégration de robots industriels (ABB, Kuka)

Formation:
- Diplôme d'ingénieur en automatismes industriels, INSA Lyon (2016)
- Certifications: Siemens TIA Portal Professional, Schneider Unity Pro
            """
        },
        "interview": {
            "messages": [
                {"role": "user", "content": "Quelles questions dois-je poser lors d'un entretien pour un poste de technicien en automatismes pour notre usine agroalimentaire?"}
            ]
        }
    },
    "finance": {
        "cv_evaluation": {
            "messages": [
                {"role": "user", "content": "Évalue ce CV pour un poste d'analyste financier dans une banque d'investissement"}
            ],
            "resume_content": """
Thomas Laurent
Analyste Financier - 5 ans d'expérience

Compétences:
- Analyse financière et modélisation (DCF, LBO, Comps, M&A)
- Valorisation d'entreprises et due diligence
- Analyse de risques et reporting financier
- Excel avancé, VBA, Bloomberg Terminal, FactSet
- Connaissance des normes IFRS et US GAAP

Expérience:
- Analyste Financier chez GrandBanque Capital (2020-présent)
  * Analyse financière pour transactions M&A (deals $50M-$200M)
  * Modélisation financière et valorisation d'entreprises
  * Préparation de mémos d'investissement pour comité de direction

- Analyste Junior chez ConseilFinance (2018-2020)
  * Support aux missions de due diligence financière
  * Analyse des états financiers et prévisions budgétaires

Formation:
- Master en Finance, HEC Paris (2018)
- Bachelor en Économie, Sciences Po Paris (2016)
- CFA Niveau II
            """
        },
        "interview": {
            "messages": [
                {"role": "user", "content": "Je dois interviewer un candidat pour un poste de contrôleur de gestion. Quelles questions clés dois-je lui poser?"}
            ]
        }
    },
    "civil_btp": {
        "cv_evaluation": {
            "messages": [
                {"role": "user", "content": "Analyse ce CV pour un poste d'ingénieur structure dans notre bureau d'études"}
            ],
            "resume_content": """
Sophie Moreau
Ingénieure Structures - 6 ans d'expérience

Compétences:
- Calcul de structures béton armé, métal, bois
- Dimensionnement parasismique (Eurocode 8)
- Logiciels: Robot Structural Analysis, ETABS, ANSYS, AutoCAD, Revit
- BIM (niveau 2), coordination multidisciplinaire
- Suivi d'exécution et contrôle qualité sur chantier

Expérience:
- Ingénieure Structures chez BuildConsult (2019-présent)
  * Conception structurelle de bâtiments tertiaires et résidentiels (R+10 à R+30)
  * Études sismiques pour projets internationaux
  * Coordination technique avec architectes et bureaux d'études fluides

- Ingénieure d'études chez ConstructTech (2017-2019)
  * Dimensionnement d'éléments en béton armé et charpente métallique
  * Participation aux revues de conception BIM

Formation:
- Diplôme d'ingénieur en génie civil, École des Ponts ParisTech (2017)
- Master spécialisé en construction durable, ESTP (2018)
            """
        },
        "interview": {
            "messages": [
                {"role": "user", "content": "Quelles questions dois-je poser à un candidat pour un poste de conducteur de travaux dans le secteur du BTP?"}
            ]
        }
    },
    "genie_industriel": {
        "cv_evaluation": {
            "messages": [
                {"role": "user", "content": "Évalue ce CV pour un poste de responsable amélioration continue dans notre usine de production"}
            ],
            "resume_content": """
François Leroy
Ingénieur en Génie Industriel - 8 ans d'expérience

Compétences:
- Lean Manufacturing, Six Sigma (Black Belt), TPM, 5S, SMED, VSM
- Gestion de production et planification
- Management visuel et résolution de problèmes (A3, PDCA, 8D)
- ERP (SAP PP/MM), GPAO, MES
- Animation d'équipes d'amélioration continue

Expérience:
- Responsable Amélioration Continue chez IndustrieAuto (2018-présent)
  * Déploiement du Lean Manufacturing sur 4 lignes de production
  * Réduction de 25% des temps de changement de série
  * Amélioration de l'OEE de 65% à 82% en 2 ans
  * Animation des chantiers Kaizen et formation des équipes

- Ingénieur Méthodes chez ProdTech (2015-2018)
  * Optimisation des flux de production et implantations d'ateliers
  * Mise en place d'un système de management visuel

Formation:
- Diplôme d'ingénieur en génie industriel, Centrale Nantes (2015)
- Certification Six Sigma Black Belt (2019)
            """
        },
        "interview": {
            "messages": [
                {"role": "user", "content": "Quelles questions dois-je poser lors d'un entretien pour un poste d'ingénieur logistique dans le secteur industriel?"}
            ]
        }
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description="Tester l'API de génération par domaine")
    parser.add_argument("--api-url", type=str, default=DEFAULT_API_URL,
                      help=f"URL de l'API LLM (défaut: {DEFAULT_API_URL})")
    parser.add_argument("--domain", type=str, choices=list(DOMAIN_TEST_EXAMPLES.keys()),
                      help="Domaine spécifique à tester (si non spécifié, tous les domaines seront testés)")
    parser.add_argument("--type", type=str, choices=["cv_evaluation", "interview"],
                      default="cv_evaluation", help="Type de test à exécuter")
    parser.add_argument("--output", type=str, help="Fichier de sortie pour les résultats (JSON)")
    
    return parser.parse_args()

def test_domain_api(api_url: str, domain: str, test_type: str) -> Dict[str, Any]:
    """Test l'API LLM pour un domaine et un type de test spécifiques"""
    
    print(f"{Fore.BLUE}Testing domain: {Fore.GREEN}{domain}{Fore.BLUE}, type: {Fore.GREEN}{test_type}{Style.RESET_ALL}")
    
    # Construire la requête
    url = f"{api_url}/generate"
    
    # Récupérer l'exemple de test
    test_example = DOMAIN_TEST_EXAMPLES[domain][test_type]
    
    # Ajouter le domaine à la requête
    request_data = {**test_example, "domain": domain}
    
    try:
        # Faire la requête API
        print(f"Sending request to {url}...")
        response = requests.post(url, json=request_data)
        response.raise_for_status()  # Lever une exception si erreur HTTP
        
        # Extraire la réponse
        result = response.json()
        
        print(f"{Fore.GREEN}Success!{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Tokens used: {result.get('token_usage', {}).get('total_tokens', 'N/A')}{Style.RESET_ALL}")
        print("\n")
        print(f"{Fore.CYAN}Response:{Style.RESET_ALL}")
        print(f"{result['response']}")
        print("\n" + "-"*80 + "\n")
        
        return {
            "domain": domain,
            "type": test_type,
            "success": True,
            "response": result["response"],
            "token_usage": result.get("token_usage", {})
        }
        
    except Exception as e:
        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        return {
            "domain": domain,
            "type": test_type,
            "success": False,
            "error": str(e)
        }

def test_all_domains(api_url: str, test_type: str) -> List[Dict[str, Any]]:
    """Teste tous les domaines avec le type de test spécifié"""
    results = []
    
    for domain in DOMAIN_TEST_EXAMPLES.keys():
        result = test_domain_api(api_url, domain, test_type)
        results.append(result)
    
    return results

def save_results(results: List[Dict[str, Any]], output_file: str):
    """Sauvegarde les résultats dans un fichier JSON"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"{Fore.GREEN}Results saved to {output_file}{Style.RESET_ALL}")

def check_api_health(api_url: str) -> bool:
    """Vérifie si l'API est disponible"""
    try:
        response = requests.get(f"{api_url}/health")
        response.raise_for_status()
        health_data = response.json()
        
        print(f"{Fore.GREEN}API is healthy!{Style.RESET_ALL}")
        print(f"Model: {health_data.get('model', 'unknown')}")
        return True
    except Exception as e:
        print(f"{Fore.RED}API is not available: {str(e)}{Style.RESET_ALL}")
        print(f"Make sure the LLM service is running at {api_url}")
        return False

def get_available_domains(api_url: str) -> List[str]:
    """Récupère la liste des domaines disponibles depuis l'API"""
    try:
        response = requests.get(f"{api_url}/available-domains")
        response.raise_for_status()
        domains_data = response.json()
        
        domains = domains_data.get("domains", [])
        mapping = domains_data.get("domains_mapping", {})
        
        print(f"{Fore.GREEN}Available domains:{Style.RESET_ALL}")
        for domain in domains:
            print(f"- {mapping.get(domain, domain)}")
        
        return domains
    except Exception as e:
        print(f"{Fore.YELLOW}Could not retrieve available domains: {str(e)}{Style.RESET_ALL}")
        return []

def main():
    args = parse_args()
    
    print(f"{Fore.CYAN}AI Recruiter LLM - Domain API Test{Style.RESET_ALL}")
    print(f"API URL: {args.api_url}")
    
    # Vérifier la santé de l'API
    if not check_api_health(args.api_url):
        return
    
    # Récupérer les domaines disponibles
    get_available_domains(args.api_url)
    
    # Exécuter les tests
    results = []
    if args.domain:
        # Tester un domaine spécifique
        result = test_domain_api(args.api_url, args.domain, args.type)
        results = [result]
    else:
        # Tester tous les domaines
        results = test_all_domains(args.api_url, args.type)
    
    # Sauvegarder les résultats si un fichier de sortie est spécifié
    if args.output:
        save_results(results, args.output)

if __name__ == "__main__":
    main() 