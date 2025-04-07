#!/usr/bin/env python
# Script pour télécharger ou sélectionner un modèle LLM pour l'application de recrutement
import os
import argparse
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

# Modèles recommandés pour différentes situations
RECOMMENDED_MODELS = {
    # Modèles légers (< 1Go)
    "tiny": [
        "distilgpt2",                      # Très léger, performances limitées
        "microsoft/phi-1_5",               # Petit mais performant
    ],
    
    # Modèles moyens (3-6Go)
    "medium": [
        "meta-llama/Llama-2-7b-chat-hf",   # Bon équilibre performance/taille
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0", # Plus petit, performances correctes
    ],
    
    # Modèles performants (7-20Go)
    "large": [
        "mistralai/Mistral-7B-Instruct-v0.2", # Excellent pour le chat
        "HuggingFaceH4/zephyr-7b-beta",    # Fine-tuné pour les instructions
    ],
    
    # Modèles multilingues
    "multilingual": [
        "bigscience/bloom-1b7",           # Multilingue, taille moyenne
        "facebook/xglm-1.7B",             # Multilingue, taille moyenne
    ],
    
    # Modèles spécialisés pour le recrutement (si disponibles)
    "recruitment": [
        "local:models/recruiter-llm",     # Modèle fine-tuné localement
    ]
}

def parse_args():
    parser = argparse.ArgumentParser(description="Télécharger et configurer un modèle LLM pour le recrutement")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2",
                        help="Modèle à télécharger (nom Hugging Face ou 'local:path')")
    parser.add_argument("--list", action="store_true", 
                        help="Lister les modèles recommandés")
    parser.add_argument("--category", type=str, choices=list(RECOMMENDED_MODELS.keys()),
                        help="Catégorie de modèle à utiliser")
    parser.add_argument("--output", type=str, default="model_config.json",
                        help="Fichier de configuration de sortie")
    
    return parser.parse_args()

def list_recommended_models():
    """Affiche la liste des modèles recommandés par catégorie"""
    print("Modèles recommandés pour l'IA de recrutement:")
    
    for category, models in RECOMMENDED_MODELS.items():
        print(f"\n== {category.upper()} ==")
        for model in models:
            print(f"- {model}")

def download_model(model_name):
    """Télécharge un modèle depuis Hugging Face (cache seulement)"""
    if model_name.startswith("local:"):
        local_path = model_name.split(":", 1)[1]
        print(f"Utilisation du modèle local: {local_path}")
        if not os.path.exists(local_path):
            print(f"ATTENTION: Le chemin {local_path} n'existe pas!")
            return False
        return True
    
    try:
        print(f"Téléchargement du modèle: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Ne pas charger le modèle complet, juste vérifier qu'il est téléchargeable
        AutoModelForCausalLM.from_pretrained(model_name, device_map=None, torch_dtype="auto")
        print(f"Modèle {model_name} téléchargé avec succès dans le cache Hugging Face")
        return True
    except Exception as e:
        print(f"Erreur lors du téléchargement du modèle {model_name}: {str(e)}")
        return False

def save_model_config(model_name, output_file):
    """Enregistre la configuration du modèle dans un fichier JSON"""
    config = {
        "model_name": model_name,
        "timestamp": "",  # sera ajouté par le serveur au démarrage
        "description": "Modèle pour l'IA de recrutement",
        "parameters": {
            "temperature": 0.7,
            "max_length": 1024,
            "top_p": 0.9,
        }
    }
    
    # Si c'est un modèle local, ajuster le chemin
    if model_name.startswith("local:"):
        config["model_path"] = model_name.split(":", 1)[1]
        config["is_local"] = True
    else:
        config["is_local"] = False
    
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration du modèle enregistrée dans {output_file}")

def main():
    args = parse_args()
    
    if args.list:
        list_recommended_models()
        return
    
    model_name = args.model
    if args.category:
        # Sélectionner le premier modèle de la catégorie spécifiée
        if args.category in RECOMMENDED_MODELS and RECOMMENDED_MODELS[args.category]:
            model_name = RECOMMENDED_MODELS[args.category][0]
            print(f"Sélection du modèle {model_name} de la catégorie {args.category}")
        else:
            print(f"Aucun modèle disponible dans la catégorie {args.category}")
            return
    
    if download_model(model_name):
        save_model_config(model_name, args.output)
        print("\nInstructions pour utiliser ce modèle:")
        print(f"1. Dans docker-compose.yml, définir MODEL_NAME={model_name}")
        print("2. Redémarrer les services avec: docker compose down && docker compose up --build")
    else:
        print("\nEchec de la configuration du modèle")

if __name__ == "__main__":
    main() 