#!/usr/bin/env python
import httpx
import asyncio
import json
import argparse
from typing import List, Dict, Any
from tabulate import tabulate

# Exemples de CV pour le test
TEST_CVS = [
    {
        "name": "Jean Dupont",
        "title": "Ingénieur DevOps",
        "domain": "informatique_reseaux",
        "content": """
JEAN DUPONT
Ingénieur DevOps | Paris, France
jean.dupont@exemple.com | +33 6 12 34 56 78 | github.com/jeandupont

EXPÉRIENCE PROFESSIONNELLE

Senior DevOps Engineer, Cloud Solutions (2020-présent)
- Administration d'infrastructures cloud sur AWS et Azure
- Déploiement et gestion de clusters Kubernetes pour microservices
- Automatisation CI/CD avec Jenkins, GitHub Actions et GitLab CI
- Monitoring et observabilité avec Prometheus, Grafana et ELK

Cloud Infrastructure Engineer, Tech Innovate (2018-2020)
- Mise en place d'infrastructures IaC avec Terraform et CloudFormation
- Migration d'applications monolithiques vers des architectures cloud-native
- Implémentation de politiques de sécurité et conformité cloud
- Optimisation des coûts cloud et amélioration des performances

COMPÉTENCES TECHNIQUES

- Cloud: AWS (EC2, EKS, S3, Lambda), Azure, GCP
- Conteneurisation: Docker, Kubernetes, Helm
- CI/CD: Jenkins, GitHub Actions, GitLab CI
- IaC: Terraform, CloudFormation, Ansible
- Monitoring: Prometheus, Grafana, ELK
- Langages: Python, Bash, Go
- Sécurité: Active Directory, IAM, sécurité réseau

FORMATION

- Ingénieur en Informatique, École Centrale Paris (2015-2018)
- Certification AWS Solutions Architect & DevOps Engineer
- Certification Kubernetes Administrator (CKA)
"""
    },
    {
        "name": "Marie Lambert",
        "title": "Analyste Financière",
        "domain": "finance",
        "content": """
MARIE LAMBERT
Analyste Financière | Lyon, France
marie.lambert@exemple.com | +33 6 98 76 54 32

EXPÉRIENCE PROFESSIONNELLE

Analyste Financière Senior, Groupe Finance Global (2019-présent)
- Analyse des données financières et préparation de rapports pour la direction
- Développement de modèles financiers complexes pour l'évaluation d'investissements
- Coordination avec les départements comptables et fiscaux
- Participation aux présentations des résultats trimestriels aux investisseurs

Analyste Finance, BanqueInvest (2016-2019)
- Analyse de la rentabilité des projets d'investissement
- Préparation des prévisions budgétaires et suivi de trésorerie
- Élaboration de tableaux de bord de gestion financière
- Support aux décisions d'investissement stratégique

COMPÉTENCES

- Analyse financière approfondie et modélisation
- Maîtrise d'Excel, PowerBI, Tableau
- Expertise en évaluation de projets (VAN, TRI, payback)
- Connaissances approfondies en comptabilité et fiscalité
- Expérience avec SAP Finance et Oracle Financials
- Excellentes capacités d'analyse et de présentation

FORMATION

- Master en Finance d'Entreprise, HEC Paris (2014-2016)
- Licence en Sciences Économiques, Université Lyon 2 (2011-2014)
- Certification CFA Niveau II
"""
    },
    {
        "name": "Ahmed Malik",
        "title": "Ingénieur en Automatisme Industriel",
        "domain": "automatismes_info_industrielle",
        "content": """
AHMED MALIK
Ingénieur en Automatisme Industriel | Grenoble, France
ahmed.malik@exemple.com | +33 7 45 67 89 01

EXPÉRIENCE PROFESSIONNELLE

Ingénieur Automaticien Senior, Automation Systems (2018-présent)
- Conception et programmation de systèmes d'automatisation industrielle
- Développement de solutions SCADA et IHM pour industries diverses
- Mise en œuvre de projets d'intégration MES/ERP
- Coordination technique avec équipes multidisciplinaires

Ingénieur Automatisme, Industrial Control (2015-2018)
- Programmation d'automates Siemens et Schneider Electric
- Conception de schémas électriques et d'architecture réseau industriel
- Mise en service de systèmes automatisés sur site client
- Amélioration des processus de production par automatisation

COMPÉTENCES TECHNIQUES

- Automates: Siemens (TIA Portal, Step7), Schneider, Allen Bradley
- SCADA/IHM: WinCC, Wonderware, Ignition
- Réseaux industriels: Profinet, Profibus, Modbus, OPC UA
- Programmation: Ladder, SCL, STL, FBD, C/C++, Python
- CAO électrique: EPLAN, AutoCAD Electrical
- Gestion de projets techniques et coordination d'équipes

FORMATION

- Ingénieur en Automatisme et Systèmes Industriels, INSA Lyon (2012-2015)
- DUT GEII (Génie Électrique et Informatique Industrielle), IUT Grenoble (2010-2012)
- Formation continue: Cybersécurité des systèmes industriels, TÜV Rheinland
"""
    },
    {
        "name": "Sophie Chen",
        "title": "Ingénieure Génie Civil",
        "domain": "genie_civil_btp",
        "content": """
SOPHIE CHEN
Ingénieure en Génie Civil | Bordeaux, France
sophie.chen@exemple.com | +33 6 45 23 78 90

EXPÉRIENCE PROFESSIONNELLE

Ingénieure Structures, BuildTech Engineering (2017-présent)
- Conception et dimensionnement de structures en béton armé et charpente métallique
- Analyse structurelle par calculs aux éléments finis
- Suivi technique de chantiers et coordination avec bureaux d'études
- Élaboration de plans d'exécution et notes de calcul

Ingénieure Bâtiment, BTP Consultants (2014-2017)
- Études techniques de bâtiments résidentiels et tertiaires
- Dimensionnement d'ouvrages selon les Eurocodes
- Participation aux réunions de coordination tous corps d'état
- Rédaction de rapports techniques et expertises

COMPÉTENCES TECHNIQUES

- Calcul de structures (béton armé, acier, bois)
- Logiciels: Robot Structural Analysis, ETABS, SAP2000, Advance Design
- Connaissance approfondie des Eurocodes et DTU
- CAO: AutoCAD, Revit Structure (BIM)
- Gestion de projets BTP et coordination technique
- Expérience en conception parasismique

FORMATION

- Diplôme d'Ingénieur en Génie Civil, ESTP Paris (2011-2014)
- Licence en Sciences et Technologies, Université Bordeaux (2008-2011)
- Formation continue: BIM pour les structures, certification Revit Structure
"""
    },
    {
        "name": "Thomas Moreau",
        "title": "Responsable Production Industrielle",
        "domain": "genie_industriel",
        "content": """
THOMAS MOREAU
Responsable Production Industrielle | Lille, France
thomas.moreau@exemple.com | +33 6 78 90 12 34

EXPÉRIENCE PROFESSIONNELLE

Directeur de Production, Manufacturing Excellence (2019-présent)
- Direction des opérations de production pour 3 lignes de fabrication (80 employés)
- Mise en œuvre de méthodologies Lean Manufacturing et amélioration continue
- Réduction des coûts de production de 15% sur 2 ans
- Gestion du planning de production et optimisation des ressources

Responsable Production, Industrial Systems (2015-2019)
- Supervision des équipes de production (40 personnes)
- Implémentation de normes ISO 9001 et 14001
- Analyse et amélioration des KPIs de production
- Coordination avec les départements qualité et maintenance

COMPÉTENCES

- Gestion de production industrielle
- Expertise en Lean Manufacturing, Six Sigma (Green Belt)
- Maîtrise des outils SMED, 5S, Kaizen, VSM
- Expérience avec ERP (SAP) et MES
- Leadership et gestion d'équipes multidisciplinaires
- Gestion de projets d'amélioration continue
- Analyse de données et reporting industriel

FORMATION

- Ingénieur en Génie Industriel, Arts et Métiers ParisTech (2012-2015)
- DUT Gestion de Production, IUT Lille (2010-2012)
- Formation: Green Belt Six Sigma, Management d'équipe
"""
    }
]

async def upload_cv(api_url: str, cv: Dict[str, Any]) -> Dict[str, Any]:
    """Télécharger un CV pour analyse"""
    try:
        print(f"Téléchargement du CV de {cv['name']}...")
        
        # Préparer les données du CV
        cv_data = {
            "content": cv["content"],
            "metadata": {
                "name": cv["name"],
                "title": cv["title"],
                "domain": cv["domain"]
            }
        }
        
        # Envoyer le CV au backend
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{api_url}/upload/resume",
                json=cv_data
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ CV de {cv['name']} téléchargé avec succès (ID: {result.get('id', 'N/A')})")
                return result
            else:
                print(f"❌ Erreur lors du téléchargement du CV de {cv['name']}: {response.status_code}")
                print(f"Détails: {response.text}")
                return {"status": "error"}
    except Exception as e:
        print(f"Exception lors du téléchargement du CV de {cv['name']}: {str(e)}")
        return {"status": "error"}

async def find_matching_jobs(api_url: str, cv_text: str, domain: str = None, limit: int = 3) -> List[Dict[str, Any]]:
    """Rechercher des offres d'emploi correspondant à un CV"""
    try:
        # Rechercher des offres similaires
        params = {"resume_text": cv_text, "limit": limit}
        if domain:
            params["domain"] = domain
            
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(
                f"{api_url}/search/jobs",
                params=params
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("results", [])
            else:
                print(f"❌ Erreur lors de la recherche d'offres correspondantes: {response.status_code}")
                print(f"Détails: {response.text}")
                return []
    except Exception as e:
        print(f"Exception lors de la recherche d'offres: {str(e)}")
        return []

async def test_cv_matching(api_url: str, limit: int = 3):
    """Tester le matching entre CV et offres d'emploi"""
    print(f"Test de matching pour {len(TEST_CVS)} CV...")
    print(f"URL API: {api_url}")
    
    all_results = []
    
    for cv in TEST_CVS:
        print("\n" + "="*80)
        print(f"CV: {cv['name']} - {cv['title']} ({cv['domain']})")
        print("-"*80)
        
        # Rechercher des offres correspondantes
        matches = await find_matching_jobs(api_url, cv["content"], cv["domain"], limit)
        
        if not matches:
            print("Aucune offre correspondante trouvée.")
            continue
        
        # Formater les résultats pour l'affichage
        result_data = []
        for i, match in enumerate(matches):
            job_metadata = match.get("metadata", {})
            result_data.append([
                i + 1,
                job_metadata.get("title", "Titre inconnu"),
                job_metadata.get("company", "Entreprise inconnue"),
                job_metadata.get("domain", "Domaine inconnu"),
                f"{match.get('score', 0.0):.2f}",
            ])
        
        # Afficher les résultats en tableau
        headers = ["Rang", "Titre", "Entreprise", "Domaine", "Score"]
        print(tabulate(result_data, headers=headers, tablefmt="grid"))
        
        # Enregistrer pour le résumé
        all_results.append({
            "cv": cv,
            "matches": matches
        })
    
    # Afficher un résumé
    print("\n" + "="*80)
    print("RÉSUMÉ DES RÉSULTATS")
    print("-"*80)
    
    summary_data = []
    for result in all_results:
        cv = result["cv"]
        top_match = result["matches"][0] if result["matches"] else None
        
        if top_match:
            job_metadata = top_match.get("metadata", {})
            cv_domain = cv["domain"]
            job_domain = job_metadata.get("domain", "")
            domain_match = "✅" if cv_domain == job_domain else "❌"
            
            summary_data.append([
                cv["name"],
                cv["domain"],
                job_metadata.get("title", ""),
                job_domain,
                domain_match,
                f"{top_match.get('score', 0.0):.2f}"
            ])
        else:
            summary_data.append([
                cv["name"],
                cv["domain"],
                "Aucun match",
                "-",
                "❌",
                "0.00"
            ])
    
    # Afficher le résumé en tableau
    headers = ["Candidat", "Domaine CV", "Meilleure offre", "Domaine offre", "Match domaine", "Score"]
    print(tabulate(summary_data, headers=headers, tablefmt="grid"))

def main():
    parser = argparse.ArgumentParser(description="Tester le matching entre CV et offres d'emploi")
    parser.add_argument("--api-url", default="http://localhost:8000/api/v1", help="URL de l'API backend")
    parser.add_argument("--limit", type=int, default=3, help="Nombre d'offres à récupérer par CV")
    
    args = parser.parse_args()
    
    # Exécuter la tâche asynchrone
    asyncio.run(test_cv_matching(args.api_url, args.limit))

if __name__ == "__main__":
    main() 