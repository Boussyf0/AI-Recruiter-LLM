#!/bin/bash

# Vérifie si une clé API OpenAI est définie
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️ Attention: Aucune clé API OpenAI n'est définie."
    echo "Pour utiliser l'API OpenAI, ajoutez votre clé dans le fichier .env"
    echo "ou exportez-la directement: export OPENAI_API_KEY=votre_cle_api"
    echo ""
fi

# Charge les variables d'environnement
if [ -f .env ]; then
    echo "🔧 Chargement des variables d'environnement depuis .env"
    export $(grep -v '^#' .env | xargs -0)
fi

# Affiche la configuration
echo "📊 Configuration actuelle:"
echo "- USE_OPENAI: ${USE_OPENAI:-false}"
echo "- OPENAI_MODEL: ${OPENAI_MODEL:-gpt-3.5-turbo}"
echo "- MODEL_NAME: ${MODEL_NAME:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
echo ""

# Démarre les services Docker
echo "🚀 Démarrage des services..."
docker compose -f docker/docker-compose.yml down
docker compose -f docker/docker-compose.yml up --build

# Pour exécuter en arrière-plan, décommentez la ligne suivante:
# docker compose -f docker/docker-compose.yml up --build -d 