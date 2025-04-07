import os
import json
import time
import logging
import random
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configurer le logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
LLM_HOST = os.getenv("LLM_HOST", "0.0.0.0")
LLM_PORT = int(os.getenv("LLM_PORT", "8001"))
MODEL_NAME = os.getenv("MODEL_NAME", "distilbert/distilbert-base-uncased")
USE_OPENAI = os.getenv("USE_OPENAI", "false").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
# Si True, essayera de charger un modèle, sinon utilisera juste les réponses prédéfinies
USE_MODEL = os.getenv("USE_MODEL", "false").lower() == "true"

# Variables globales pour le modèle
model = None
tokenizer = None
feature_extraction = None
generation_pipeline = None

# Initialize FastAPI app
app = FastAPI(title="AI Recruiter LLM API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class Message(BaseModel):
    role: str
    content: str

class GenerationRequest(BaseModel):
    messages: List[Message]
    job_description_id: Optional[str] = None
    resume_id: Optional[str] = None
    job_description: Optional[str] = None
    resume_content: Optional[str] = None
    max_length: int = 1024
    temperature: float = 0.7
    system_prompt: Optional[str] = None
    domain: Optional[str] = None

class GenerationResponse(BaseModel):
    response: str
    token_usage: Optional[Dict[str, int]] = None

class HealthResponse(BaseModel):
    status: str
    model: str
    use_openai: bool

class EvaluationRequest(BaseModel):
    domain_id: str
    type: str
    content: str

class AnalysisRequest(BaseModel):
    domain_id: str
    content: str

class SummarizeRequest(BaseModel):
    content: str

class EmbeddingRequest(BaseModel):
    text: str
    domain: Optional[str] = None

class EmbeddingResponse(BaseModel):
    embeddings: List[float]
    model: str

class AnalysisResponse(BaseModel):
    match_score: float
    competences: List[Dict[str, Any]]
    summary: str

class SummarizeResponse(BaseModel):
    summary: str

# Réponses prédéfinies pour les différents cas d'utilisation
PREDEFINED_RESPONSES = {
    "competences": """Voici les compétences clés pour un développeur full-stack moderne dans le domaine informatique & réseaux:

Compétences techniques:
1. JavaScript/TypeScript et frameworks modernes (React, Vue.js, Angular)
2. Langages backend (Python, Node.js, Java, etc.) et leurs frameworks (Django, Express, Spring)
3. Bases de données relationnelles et NoSQL (PostgreSQL, MongoDB)
4. DevOps et conteneurisation (Docker, Kubernetes, CI/CD)
5. Architecture cloud et microservices (AWS, Azure, GCP)

Soft skills:
1. Communication efficace et travail d'équipe
2. Résolution de problèmes complexes et pensée critique
3. Adaptabilité et apprentissage continu
""",

    "cv_evaluation": """Évaluation du CV pour le poste de développeur full-stack:

Forces:
- Profil technique complet avec maîtrise des technologies frontend et backend
- Expérience significative (4 ans) avec progression de responsabilités
- Compétences DevOps qui complètent le profil full-stack
- Stack technique moderne et recherchée (React, Node.js, Docker)
- Formation solide en développement web

Points d'amélioration:
- Pas de mention de projets open-source ou contributions personnelles
- Expérience limitée à deux entreprises
- Manque de détails sur les résultats obtenus ou métriques d'impact

Adéquation globale:
Le profil correspond bien aux attentes pour un poste de développeur full-stack, avec une combinaison équilibrée de compétences frontend, backend et DevOps. L'expérience en migration vers des microservices est particulièrement pertinente pour des environnements modernes.
""",

    "entretien": """Voici 5 questions techniques pertinentes pour un entretien de développeur full-stack spécialisé en React et Node.js:

1. Question: "Expliquez comment vous implémenteriez une authentification sécurisée dans une application React/Node.js?"
   Évaluation: Compréhension des mécanismes d'authentification (JWT, sessions), stockage sécurisé, gestion des tokens côté client.
   Réponse attendue: Mention de JWT, refresh tokens, HttpOnly cookies, protection CSRF, et stratégies de validation.

2. Question: "Comment optimiseriez-vous les performances d'une application React qui commence à ralentir?"
   Évaluation: Connaissance des techniques d'optimisation React et capacité à diagnostiquer des problèmes.
   Réponse attendue: Utilisation de React.memo, useMemo, useCallback, analyse avec Profiler, code splitting, lazy loading.

3. Question: "Décrivez votre approche pour concevoir une API RESTful scalable avec Node.js"
   Évaluation: Compréhension des principes REST, architecture et scalabilité.
   Réponse attendue: Structure de routes claire, middleware pour validation, gestion des erreurs, documentation swagger, tests.

4. Question: "Comment assureriez-vous la qualité du code dans un projet d'équipe React/Node.js?"
   Évaluation: Pratiques de développement professionnel et expérience avec les outils de qualité.
   Réponse attendue: Tests unitaires/e2e (Jest, React Testing Library, Cypress), ESLint, Prettier, CI/CD, revues de code.

5. Question: "Expliquez comment vous géreriez l'état global d'une application React complexe"
   Évaluation: Connaissance des solutions de gestion d'état et capacité à choisir la bonne approche.
   Réponse attendue: Comparaison entre Context API, Redux, Zustand ou Recoil, et justification du choix selon la complexité.
"""
}

# Définir les compétences par domaine pour la simulation
DOMAIN_COMPETENCES = {
    "informatique_reseaux": [
        {"name": "Développement Frontend", "keywords": ["javascript", "typescript", "react", "angular", "vue", "html", "css"]},
        {"name": "Développement Backend", "keywords": ["node.js", "python", "java", "c#", "php", "api", "rest"]},
        {"name": "DevOps", "keywords": ["docker", "kubernetes", "ci/cd", "jenkins", "git", "aws", "azure", "cloud"]},
        {"name": "Réseaux", "keywords": ["tcp/ip", "dns", "vpn", "firewall", "lan", "wan", "routage"]},
        {"name": "Sécurité", "keywords": ["cybersécurité", "cryptographie", "oauth", "jwt", "authentification"]}
    ],
    "automatismes_info_industrielle": [
        {"name": "Automatismes", "keywords": ["plc", "automate", "scada", "api", "supervision", "grafcet"]},
        {"name": "Électronique", "keywords": ["microcontrôleur", "arduino", "raspberry", "capteur", "actionneur"]},
        {"name": "Informatique industrielle", "keywords": ["temps réel", "embarqué", "iot", "industrie 4.0"]},
        {"name": "Protocoles industriels", "keywords": ["modbus", "profinet", "canopen", "ethernet/ip", "opc ua"]},
        {"name": "Robotique", "keywords": ["robot", "cobot", "programmation robot", "vision"]}
    ],
    "finance": [
        {"name": "Analyse financière", "keywords": ["ratio", "bilan", "compte de résultat", "trésorerie", "analyse"]},
        {"name": "Comptabilité", "keywords": ["comptabilité", "normes", "ifrs", "audit", "bilan"]},
        {"name": "Gestion de portefeuille", "keywords": ["investissement", "actions", "obligations", "actif", "risque"]},
        {"name": "Modélisation financière", "keywords": ["excel", "modèle", "prévision", "simulation", "valorisation"]},
        {"name": "Conformité", "keywords": ["réglementation", "conformité", "kyc", "aml", "rgpd"]}
    ],
    "genie_civil_btp": [
        {"name": "Conception structurelle", "keywords": ["structure", "calcul", "béton", "charpente", "fondation"]},
        {"name": "Gestion de projet BTP", "keywords": ["planification", "chantier", "budget", "délai", "coordination"]},
        {"name": "Logiciels CAO/BIM", "keywords": ["autocad", "revit", "bim", "archicad", "sketchup"]},
        {"name": "Matériaux", "keywords": ["béton", "acier", "bois", "matériaux", "durabilité"]},
        {"name": "Réglementation", "keywords": ["norme", "dtu", "eurocodes", "réglementation", "thermique"]}
    ],
    "genie_industriel": [
        {"name": "Gestion de production", "keywords": ["production", "lean", "juste-à-temps", "kanban", "flux"]},
        {"name": "Supply Chain", "keywords": ["logistique", "supply chain", "approvisionnement", "stock", "transport"]},
        {"name": "Qualité", "keywords": ["qualité", "iso", "amélioration continue", "kaizen", "six sigma"]},
        {"name": "Maintenance", "keywords": ["maintenance", "tpm", "préventif", "correctif", "fiabilité"]},
        {"name": "Méthodes", "keywords": ["méthodes", "temps", "ergonomie", "poste", "process"]}
    ]
}

@app.get("/")
def read_root():
    return {"status": "ok", "message": "AI Recruiter LLM service is running", "model": MODEL_NAME}

@app.on_event("startup")
async def startup_event():
    global model, tokenizer, feature_extraction, generation_pipeline, MODEL_NAME, USE_OPENAI
    
    logger.info(f"Configuration de l'API LLM: MODEL_NAME={MODEL_NAME}, USE_OPENAI={USE_OPENAI}, USE_MODEL={USE_MODEL}")
    
    if USE_MODEL and not USE_OPENAI:
        try:
            logger.info(f"Tentative de chargement du modèle: {MODEL_NAME}")
            
            # Import transformers seulement si nécessaire
            from transformers import AutoModel, AutoTokenizer, pipeline
            import torch

            # Charger le modèle BERT et son tokenizer
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModel.from_pretrained(MODEL_NAME)
            
            # Créer un pipeline pour l'extraction de caractéristiques
            feature_extraction = pipeline('feature-extraction', model=model, tokenizer=tokenizer)
            
            logger.info("Modèle BERT chargé avec succès pour l'extraction de caractéristiques")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
            logger.info("Utilisation des réponses prédéfinies comme fallback")
            model = None
            tokenizer = None
            feature_extraction = None
    else:
        logger.info("Mode réponses prédéfinies activé, pas de chargement de modèle")

def extract_key_phrases(text, n=5):
    """Extrait les phrases ou termes clés d'un texte en utilisant BERT"""
    if feature_extraction is None or tokenizer is None:
        return []
    
    try:
        # Limiter la taille du texte
        max_length = 512
        text = text[:max_length]
        
        # Extraire les embeddings
        features = feature_extraction(text, return_tensors=True)
        features = features[0]
        
        # Convertir le texte en tokens
        tokens = tokenizer.tokenize(text)
        
        # Supprimer les tokens de ponctuation et les stopwords
        import string
        stop_words = ['le', 'la', 'les', 'un', 'une', 'des', 'et', 'ou', 'de', 'du', 'ce', 'cette', 'ces']
        filtered_tokens = [token for token in tokens if token not in string.punctuation and token not in stop_words]
        
        # Retourner les tokens les plus significatifs (en utilisant l'amplitude des embeddings)
        import numpy as np
        token_importances = np.mean(np.abs(features), axis=1)[:len(filtered_tokens)]
        
        # Trouver les indices des n tokens les plus importants
        if len(token_importances) > 0:
            top_indices = np.argsort(-token_importances)[:min(n, len(token_importances))]
            key_tokens = [filtered_tokens[i] for i in top_indices]
            return key_tokens
        return []
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction des phrases clés: {str(e)}")
        return []

def analyze_cv_with_bert(cv_text, job_description=None):
    """Analyse un CV en utilisant BERT et génère un rapport d'évaluation"""
    if model is None or tokenizer is None:
        return PREDEFINED_RESPONSES["cv_evaluation"]
    
    try:
        # Extraire les mots-clés du CV
        cv_keywords = extract_key_phrases(cv_text, 10)
        
        # Extraire les mots-clés de la description de poste (si disponible)
        job_keywords = []
        if job_description:
            job_keywords = extract_key_phrases(job_description, 10)
        
        # Créer une réponse enrichie avec les mots-clés extraits
        job_match = "N/A"
        if job_keywords and cv_keywords:
            # Calculer le chevauchement des mots-clés pour estimer la correspondance
            matching_keywords = set(cv_keywords).intersection(set(job_keywords))
            match_percentage = len(matching_keywords) / len(job_keywords) if len(job_keywords) > 0 else 0
            job_match = f"{int(match_percentage * 100)}%"
        
        # Adapter la réponse prédéfinie avec l'information extraite
        template = PREDEFINED_RESPONSES["cv_evaluation"]
        
        # Enrichir avec les informations extraites
        enriched_response = (
            f"Analyse de CV avec DistilBERT:\n\n"
            f"Mots-clés détectés: {', '.join(cv_keywords)}\n"
            f"Correspondance avec la description de poste: {job_match}\n\n"
            f"{template}"
        )
        
        return enriched_response
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse du CV avec BERT: {str(e)}")
        return PREDEFINED_RESPONSES["cv_evaluation"]

def analyze_skillset_with_bert(request):
    """Analyse une demande de compétences en utilisant BERT"""
    if model is None or tokenizer is None:
        return PREDEFINED_RESPONSES["competences"]
    
    try:
        # Extraire le texte de la requête
        query_text = ""
        for msg in request.messages:
            if msg.role == "user":
                query_text += msg.content + " "
        
        # Extraire les domaines et termes techniques mentionnés
        keywords = extract_key_phrases(query_text, 8)
        
        # Adapter la réponse avec les termes techniques extraits
        domain_specifics = ""
        
        # Ajouter des informations spécifiques au domaine si disponible
        if request.domain:
            if request.domain == "informatique_reseaux":
                domain_specifics = "\nCompétences spécifiques en Informatique & Réseaux:\n1. Architectures cloud et serverless\n2. Sécurité des API et OAuth 2.0\n3. Microservices et orchestration\n"
            elif request.domain == "automatismes_info_industrielle":
                domain_specifics = "\nCompétences spécifiques en Automatismes & Info Industrielle:\n1. Programmation d'automates (PLC)\n2. Interfaces SCADA\n3. Communications industrielles (OPC UA, Modbus)\n"
        
        # Créer la réponse enrichie
        enriched_response = (
            f"Analyse avec DistilBERT:\n\n"
            f"Termes techniques détectés: {', '.join(keywords)}\n"
            f"{PREDEFINED_RESPONSES['competences']}"
            f"{domain_specifics}"
        )
        
        return enriched_response
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse des compétences avec BERT: {str(e)}")
        return PREDEFINED_RESPONSES["competences"]

@app.post("/generate")
async def generate(request: GenerationRequest):
    """Génère une réponse basée sur les messages et contexte fournis"""
    # Mesurer le temps d'exécution
    start_time = time.time()
    
    # Déterminer le type de requête (question générale, analyse de CV, etc.)
    request_type = determine_request_type(request)
    
    # Gérer les différents types de requêtes
    response_text = ""
    tokens_in = 0
    tokens_out = 0
    
    if request_type == "competences":
        response_text = analyze_skillset_with_bert(request)
        tokens_in = 150  # Valeurs simulées
        tokens_out = 350
    elif request_type == "cv_evaluation":
        response_text = analyze_cv_with_bert(request.resume_content, request.job_description)
        tokens_in = 500
        tokens_out = 400
    elif request_type == "entretien":
        response_text = PREDEFINED_RESPONSES["entretien"]
        tokens_in = 150
        tokens_out = 500
    else:
        # Requête générale non spécifique
        response_text = "Je suis l'assistant IA Recruiter, spécialisé dans l'analyse de CV et l'évaluation des compétences. Comment puis-je vous aider?"
        tokens_in = 50
        tokens_out = 100
    
    # Calculer le temps d'exécution
    execution_time = time.time() - start_time
    logger.info(f"Requête traitée en {execution_time:.2f} secondes")
    
    # Retourner la réponse avec des statistiques d'utilisation de tokens
    return GenerationResponse(
        response=response_text,
        token_usage={
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "total_tokens": tokens_in + tokens_out
        }
    )

def determine_request_type(request: GenerationRequest) -> str:
    """Détermine le type de requête (compétences, CV, entretien)"""
    # Vérifier s'il y a un CV dans la requête
    if request.resume_content and len(request.resume_content) > 100:
        return "cv_evaluation"
    
    # Analyser le contenu des messages
    query_text = ""
    for msg in request.messages:
        if msg.role == "user":
            query_text += msg.content.lower() + " "
    
    # Vérifier les mots clés pour déterminer le type
    if any(keyword in query_text for keyword in ["compétence", "skill", "capacité", "savoir-faire"]):
        return "competences"
    elif any(keyword in query_text for keyword in ["entretien", "interview", "question", "embauche"]):
        return "entretien"
    else:
        return "general"

@app.post("/evaluate", response_model=Dict[str, Any])
async def evaluate_document(request: EvaluationRequest):
    """Évalue un document pour un domaine spécifique"""
    logger.info(f"Évaluation demandée pour le domaine: {request.domain_id}, type: {request.type}")
    
    start_time = time.time()
    
    try:
        # Simuler une analyse (dans une implémentation réelle, utilisez un vrai LLM)
        if request.type == "cv_evaluation":
            # Extraire les mots-clés supposés du CV
            cv_keywords = ["python", "javascript", "react", "node.js", "api", "cloud"]
            
            # Adapter la réponse prédéfinie
            response = (
                f"Analyse de CV avec DistilBERT pour le domaine {request.domain_id}:\n\n"
                f"Mots-clés détectés: {', '.join(cv_keywords)}\n\n"
                f"{PREDEFINED_RESPONSES['cv_evaluation']}"
            )
        else:
            response = "Type d'évaluation non supporté"
        
        # Calculer le temps d'exécution
        execution_time = time.time() - start_time
        logger.info(f"Évaluation traitée en {execution_time:.2f} secondes")
        
        return {
            "response": response,
            "token_usage": {
                "input_tokens": 500,
                "output_tokens": 400,
                "total_tokens": 900
            },
            "execution_time": execution_time
        }
    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'évaluation: {str(e)}")

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_document(request: AnalysisRequest):
    """Analyse un CV pour un domaine spécifique et retourne un score et des compétences"""
    logger.info(f"Analyse demandée pour le domaine: {request.domain_id}")
    
    # Vérifier si le domaine est valide
    if request.domain_id not in DOMAIN_COMPETENCES:
        raise HTTPException(status_code=400, detail=f"Domaine non supporté: {request.domain_id}")
    
    try:
        # Simuler une analyse (dans une implémentation réelle, utilisez un vrai LLM)
        # Calculer un score de correspondance simulé pour le domaine
        match_score = random.uniform(0.5, 0.95)
        
        # Extraire les compétences du domaine et leur attribuer des scores aléatoires
        domain_competences = DOMAIN_COMPETENCES[request.domain_id]
        competences_with_scores = []
        
        for comp in domain_competences:
            # Vérifier combien de mots-clés de la compétence sont dans le CV
            # Dans une vraie implémentation, on analyserait réellement le texte
            level = random.uniform(0.3, 0.9)  # Simulé
            competences_with_scores.append({
                "name": comp["name"],
                "level": level,
                "keywords_found": random.sample(comp["keywords"], min(3, len(comp["keywords"])))
            })
        
        # Créer un résumé simulé
        summary = f"Profil orienté {request.domain_id.replace('_', ' ').title()} avec des compétences variées. Points forts dans {competences_with_scores[0]['name']} et {competences_with_scores[1]['name']}."
        
        return AnalysisResponse(
            match_score=match_score,
            competences=competences_with_scores,
            summary=summary
        )
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'analyse: {str(e)}")

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_document(request: SummarizeRequest):
    """Résume un document (CV) pour en extraire les points clés"""
    try:
        # Dans une implémentation réelle, utilisez un vrai LLM pour résumer le document
        # Pour cet exemple, nous retournons un résumé simulé
        summary = (
            "Profil de développeur full-stack avec 4 ans d'expérience, spécialisé en React et Node.js. "
            "Formation en informatique et expérience dans des environnements agiles. "
            "Compétences en DevOps et déploiement cloud. Bonne capacité de communication et travail d'équipe."
        )
        
        return SummarizeResponse(summary=summary)
    except Exception as e:
        logger.error(f"Erreur lors de la création du résumé: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la création du résumé: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Vérifier l'état de santé de l'API et du modèle"""
    return HealthResponse(
        status="healthy",
        model=MODEL_NAME,
        use_openai=USE_OPENAI
    )

@app.get("/available-domains")
def get_available_domains():
    """Retourne la liste des domaines disponibles pour l'analyse"""
    return {
        "domains": [
            {"id": "informatique_reseaux", "name": "Informatique et Réseaux"},
            {"id": "automatismes_info_industrielle", "name": "Automatismes et Informatique Industrielle"},
            {"id": "finance", "name": "Finance"},
            {"id": "genie_civil_btp", "name": "Génie Civil et BTP"},
            {"id": "genie_industriel", "name": "Génie Industriel"}
        ]
    }

@app.post("/embeddings", response_model=EmbeddingResponse)
async def generate_embeddings(request: EmbeddingRequest):
    """
    Génère des embeddings pour un texte donné
    """
    global model, tokenizer, feature_extraction
    
    if USE_OPENAI:
        try:
            import openai
            openai.api_key = OPENAI_API_KEY
            
            response = openai.Embedding.create(
                input=request.text,
                model="text-embedding-ada-002"
            )
            
            return {"embeddings": response["data"][0]["embedding"], "model": "text-embedding-ada-002"}
        except Exception as e:
            logger.error(f"Erreur lors de la génération d'embeddings avec OpenAI: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur d'embedding OpenAI: {str(e)}")
    elif USE_MODEL:
        # Utilisation d'un modèle local
        try:
            # Importer le générateur d'embeddings amélioré
            try:
                from improve_embeddings import EmbeddingGenerator
                
                # Vérifier si le générateur existe déjà
                if not hasattr(app.state, "embedding_generator"):
                    logger.info("Initialisation du générateur d'embeddings amélioré...")
                    app.state.embedding_generator = EmbeddingGenerator(model_name=MODEL_NAME)
                
                # Générer l'embedding avec le nouveau générateur
                embedding = app.state.embedding_generator.get_composite_embedding(
                    request.text, 
                    domain=request.domain
                ).tolist()
                
                logger.info(f"Embedding généré avec le générateur amélioré (domain: {request.domain})")
                return {"embeddings": embedding, "model": f"{MODEL_NAME}-composite"}
            
            except ImportError:
                logger.warning("Module improve_embeddings non disponible, utilisation de la méthode standard")
                
                import torch
                from transformers import AutoTokenizer, AutoModel
                
                # Charger le modèle et tokenizer si nécessaire
                if feature_extraction is None:
                    logger.info(f"Chargement du modèle d'embedding: {MODEL_NAME}")
                    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
                    feature_extraction = AutoModel.from_pretrained(MODEL_NAME)
                    feature_extraction.eval()
                
                # Tokenization et génération d'embeddings
                inputs = tokenizer(request.text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = feature_extraction(**inputs)
                
                # Utiliser la représentation CLS (première token) comme embedding du document
                embeddings = outputs.last_hidden_state[:, 0, :].squeeze().tolist()
                
                return {"embeddings": embeddings, "model": MODEL_NAME}
        except Exception as e:
            logger.error(f"Erreur lors de la génération d'embeddings avec le modèle local: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Erreur d'embedding avec modèle local: {str(e)}")
    else:
        # Génération d'embeddings simulés pour les tests
        logger.warning("Génération d'embeddings simulés (mode sans modèle)")
        import numpy as np
        
        # Générer un vecteur aléatoire de dimension 768 (taille standard pour BERT/DistilBERT)
        fake_embeddings = list(np.random.normal(0, 1, 768).astype(float))
        
        return {"embeddings": fake_embeddings, "model": "simulated-embeddings"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("inference:app", host=LLM_HOST, port=LLM_PORT, reload=True)
