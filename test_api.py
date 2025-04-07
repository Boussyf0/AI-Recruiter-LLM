import subprocess
import json
import argparse
import sys
import time

def test_generation_api(domain="informatique_reseaux", query_type="competences"):
    """
    Test l'API de génération du LLM avec différents types de requêtes.
    
    Args:
        domain (str): Le domaine pour la requête (ex: informatique_reseaux, sante, etc.)
        query_type (str): Le type de requête à tester (competences, cv_evaluation, etc.)
    """
    
    # Définir différents types de requêtes
    queries = {
        "competences": {
            "system": "Tu es un assistant RH spécialisé en recrutement technique.",
            "user": "Quelles sont les compétences clés pour un développeur full-stack?"
        },
        "cv_evaluation": {
            "system": "Tu es un recruteur technique spécialisé qui évalue les CV pour des postes techniques. Donne ton avis de façon précise et détaillée.",
            "user": "Évaluer ce CV pour un poste de développeur full-stack.",
            "cv": """
    JEAN DUPONT
    Développeur Full-Stack | Paris, France
    jean.dupont@exemple.com | +33 6 12 34 56 78 | github.com/jeandupont
    
    EXPÉRIENCE PROFESSIONNELLE
    
    Senior Développeur Full-Stack, Tech Solutions (2020-présent)
    - Développement d'applications web React/Node.js pour des clients du secteur bancaire
    - Mise en place d'une architecture microservices avec Docker et Kubernetes
    - Optimisation des performances frontend, réduisant les temps de chargement de 40%
    - Encadrement d'une équipe de 3 développeurs juniors
    
    Développeur Web, StartupInno (2017-2020)
    - Développement full-stack avec Angular et Express.js
    - Création d'APIs RESTful pour des applications mobiles
    - Implémentation de tests automatisés (Jest, Cypress)
    
    COMPÉTENCES TECHNIQUES
    
    Frontend: JavaScript, TypeScript, React, Redux, Angular, HTML5, CSS3, SASS
    Backend: Node.js, Express, MongoDB, PostgreSQL, REST API, GraphQL
    DevOps: Docker, Kubernetes, AWS, CI/CD, Git
    Testing: Jest, React Testing Library, Cypress
    
    FORMATION
    
    Diplôme d'Ingénieur en Informatique, École Polytechnique (2015-2017)
    Licence en Informatique, Université Paris-Sud (2012-2015)
    """
        },
        "entretien": {
            "system": "Tu es un recruteur technique spécialisé dans l'informatique et les réseaux. Propose des questions pertinentes pour un entretien technique.",
            "user": "Propose 5 questions techniques pertinentes pour un entretien de développeur full-stack spécialisé en React et Node.js. Pour chaque question, indique ce que tu cherches à évaluer et les éléments de réponse attendus."
        }
    }
    
    # Vérifier si le type de requête existe
    if query_type not in queries:
        print(f"Type de requête inconnu: {query_type}")
        print(f"Types disponibles: {', '.join(queries.keys())}")
        return
    
    # Construire la requête
    query = queries[query_type]
    request_data = {
        "messages": [
            {"role": "system", "content": query["system"]},
            {"role": "user", "content": query["user"]}
        ],
        "temperature": 0.3,
        "max_length": 1500,
        "domain": domain
    }
    
    # Ajouter le CV si présent
    if "cv" in query:
        request_data["resume_content"] = query["cv"]
    
    print(f"Envoi d'une requête de type '{query_type}' pour le domaine '{domain}'...")
    print("-" * 50)
    
    # Préparation de la commande curl avec timeout
    cmd = [
        "curl", 
        "-s",                    # Silent mode
        "-m", "5",               # Timeout après 5 secondes
        "-v",                    # Verbose pour le débogage
        "-X", "POST", 
        "http://localhost:8001/generate",
        "-H", "Content-Type: application/json",
        "-d", json.dumps(request_data)
    ]
    
    # Exécution de la commande
    try:
        print("Exécution de la commande curl...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Afficher les détails en cas d'erreur
        if result.returncode != 0:
            print(f"Erreur (code {result.returncode}):")
            print(f"  stderr: {result.stderr}")
            # Vérifier si c'est un problème de connexion
            if "Connection refused" in result.stderr:
                print("\nErreur de connexion: Impossible de se connecter à l'API.")
                print("Assurez-vous que le service LLM est en cours d'exécution sur le port 8001.")
            elif "Connection timed out" in result.stderr:
                print("\nErreur de timeout: L'API ne répond pas dans le délai imparti.")
            return
        
        # Afficher la réponse brute pour le débogage
        if not result.stdout.strip():
            print("Erreur: Réponse vide du serveur")
            print(f"STDERR: {result.stderr}")
            return
            
        # Tentative de parsing JSON
        try:
            response_data = json.loads(result.stdout)
            
            if "response" in response_data:
                print("\nRéponse générée:")
                print("=" * 50)
                print(response_data["response"])
                print("=" * 50)
                
                if "token_usage" in response_data:
                    print("\nUtilisation des tokens:")
                    print(f"- Input: {response_data['token_usage']['input_tokens']}")
                    print(f"- Output: {response_data['token_usage']['output_tokens']}")
                    print(f"- Total: {response_data['token_usage']['total_tokens']}")
                
                if "execution_time" in response_data:
                    print(f"\nTemps d'exécution: {response_data['execution_time']:.2f} secondes")
            else:
                print("Erreur: La réponse ne contient pas le champ 'response'")
                print(f"Contenu de la réponse: {response_data}")
                
        except json.JSONDecodeError as e:
            print(f"\nErreur lors du décodage de la réponse JSON: {e}")
            print("Réponse brute:")
            print(result.stdout)
            print("\nErreur standard:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("Erreur: Le délai d'exécution de la commande a expiré")
    except subprocess.SubprocessError as e:
        print(f"Erreur lors de l'exécution de la commande: {e}")
    except Exception as e:
        print(f"Erreur inattendue: {e}")
        print(f"Type d'erreur: {type(e).__name__}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test de l'API de génération LLM")
    parser.add_argument("--domain", default="informatique_reseaux", help="Domaine pour la requête")
    parser.add_argument("--type", default="competences", choices=["competences", "cv_evaluation", "entretien"], 
                        help="Type de requête à tester")
    
    args = parser.parse_args()
    test_generation_api(args.domain, args.type) 