#!/bin/bash
# Script pour préparer les données et entraîner le modèle LLM

# Définir les couleurs pour le terminal
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== AI Recruiter - Préparation de l'entraînement du modèle LLM ===${NC}"

# Créer les répertoires nécessaires s'ils n'existent pas
mkdir -p data/models

# 1. Générer le dataset d'entraînement à partir des exemples
echo -e "${YELLOW}Étape 1: Génération du dataset d'entraînement${NC}"
python llm/create_training_dataset.py --examples data/training_examples.json --output data/training_data.csv --expand 10

# Vérifier si la génération s'est bien passée
if [ $? -ne 0 ]; then
    echo -e "${RED}Erreur lors de la génération du dataset d'entraînement${NC}"
    exit 1
fi

echo -e "${GREEN}Dataset d'entraînement généré avec succès!${NC}"

# 2. Lancer l'entraînement du modèle
echo -e "${YELLOW}Étape 2: Lancement de l'entraînement du modèle LLM${NC}"
python llm/train.py --model_name "distilbert/distilbert-base-uncased" --dataset_path data/training_data.csv --output_dir data/models/recruiter-llm --num_epochs 3

# Vérifier si l'entraînement s'est bien passé
if [ $? -ne 0 ]; then
    echo -e "${RED}Erreur lors de l'entraînement du modèle${NC}"
    exit 1
fi

echo -e "${GREEN}Entraînement du modèle terminé avec succès!${NC}"
echo -e "${GREEN}Le modèle entraîné est disponible dans data/models/recruiter-llm${NC}"

# 3. Initialiser la base de vecteurs
echo -e "${YELLOW}Étape 3: Initialisation de la base de vecteurs Qdrant${NC}"

# Vérifier si qdrant est en cours d'exécution
if curl -s -f http://localhost:6333/collections > /dev/null; then
    echo "Création des collections dans Qdrant..."
    
    # Créer la collection pour les CV
    curl -X PUT http://localhost:6333/collections/cv_collection \
        -H 'Content-Type: application/json' \
        -d '{
            "vectors": {
                "size": 768,
                "distance": "Cosine"
            }
        }'
    
    # Créer la collection pour les offres d'emploi
    curl -X PUT http://localhost:6333/collections/job_collection \
        -H 'Content-Type: application/json' \
        -d '{
            "vectors": {
                "size": 768,
                "distance": "Cosine"
            }
        }'
    
    echo -e "${GREEN}Collections Qdrant créées avec succès!${NC}"
else
    echo -e "${RED}Impossible de se connecter à Qdrant. Assurez-vous que le service est en cours d'exécution.${NC}"
    echo -e "${YELLOW}Note: Les collections seront automatiquement créées lors du premier accès via l'API.${NC}"
fi

echo -e "${GREEN}=== Préparation terminée! ===${NC}"
echo -e "Pour utiliser le modèle entraîné, définissez MODEL_NAME=data/models/recruiter-llm dans votre fichier .env"
echo -e "Pour reconstruire les conteneurs avec le nouveau modèle : docker compose -f docker/docker-compose.yml down && docker compose -f docker/docker-compose.yml up --build -d" 