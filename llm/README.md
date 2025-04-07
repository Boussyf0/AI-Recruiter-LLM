# AI Recruiter LLM

Ce module contient le service d'inférence LLM (Large Language Model) pour l'IA de recrutement. Il permet d'utiliser des modèles de langage avancés pour analyser des CV, conduire des entretiens automatisés et évaluer l'adéquation entre candidats et postes.

## Domaines spécialisés

Ce système est spécialisé dans 5 secteurs spécifiques :

1. **Informatique et Réseaux** - Recrutement pour les postes de développeurs, administrateurs systèmes, DevOps, experts cybersécurité, etc.
2. **Automatismes et Informatique Industrielle** - Spécialistes en programmation d'automates, SCADA, systèmes de contrôle industriels
3. **Finance** - Analystes financiers, contrôleurs de gestion, comptables, experts en conformité
4. **Génie Civil et BTP** - Ingénieurs structures, conducteurs de travaux, architectes, spécialistes en construction
5. **Génie Industriel** - Experts en amélioration continue, ingénieurs production, logisticiens, responsables qualité

Chaque domaine dispose de prompts spécialisés pour l'évaluation de CV, la conduite d'entretiens et l'analyse d'adéquation entre profils et postes.

## Fonctionnalités

- API FastAPI pour générer des réponses contextuelles
- Support de différents modèles de langage via Hugging Face Transformers
- Prompts spécialisés par secteur et type d'analyse
- Possibilité de fine-tuner sur des données de recrutement spécifiques aux domaines

## Modèles disponibles

Par défaut, le service utilise `mistralai/Mistral-7B-Instruct-v0.2`, un modèle performant adapté au dialogue. Vous pouvez choisir parmi différentes options selon vos besoins :

- **Léger** : `distilgpt2` ou `microsoft/phi-1_5` (pour test/développement)
- **Moyen** : `meta-llama/Llama-2-7b-chat-hf` ou `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Performant** : `mistralai/Mistral-7B-Instruct-v0.2` ou `HuggingFaceH4/zephyr-7b-beta`
- **Multilingue** : `bigscience/bloom-1b7` ou `facebook/xglm-1.7B`

Pour voir toutes les options, exécutez : 
```bash
python download_model.py --list
```

## Utilisation

### 1. Télécharger un modèle

```bash
# Utiliser un modèle léger
python download_model.py --category tiny

# Utiliser un modèle spécifique
python download_model.py --model meta-llama/Llama-2-7b-chat-hf

# Voir toutes les options
python download_model.py --list
```

### 2. Fine-tuning spécifique aux domaines

Vous pouvez adapter le modèle à vos domaines spécialisés :

```bash
# Créer un dataset de fine-tuning à partir des exemples
python create_training_dataset.py --expand 10

# Fine-tuning sur tous les domaines
python train.py --dataset_path data/training_data.csv --num_epochs 3

# Fine-tuning sur un domaine spécifique
python create_training_dataset.py --domains informatique_reseaux finance --output data/tech_finance.csv
python train.py --dataset_path data/tech_finance.csv --num_epochs 5
```

Format attendu pour le dataset CSV :
- Colonnes : `instruction`, `input`, `output`, `domain`
- Le domaine doit être l'un des 5 secteurs spécialisés mentionnés ci-dessus

### 3. Intégration avec Docker

Le modèle est automatiquement utilisé par le conteneur Docker :

```bash
# Utiliser un modèle spécifique
MODEL_NAME=microsoft/phi-1_5 docker compose -f docker/docker-compose.yml up --build
```

## API

L'API expose les endpoints suivants :

- `GET /` : Vérification que le service est actif
- `GET /health` : Vérification de l'état du service
- `GET /available-prompts` : Liste des prompts génériques pour le recrutement
- `GET /available-domains` : Liste des domaines spécialisés disponibles
- `POST /generate` : Génère une réponse basée sur l'historique de conversation

Exemple de requête avec précision du domaine :

```json
{
  "messages": [
    {"role": "user", "content": "Évalue le CV suivant pour un poste d'ingénieur automaticien"}
  ],
  "resume_content": "Jean Dupont\nIngénieur automaticien avec 7 ans d'expérience...",
  "job_description": "Nous recherchons un ingénieur en automatismes pour notre usine...",
  "domain": "automatismes_info_industrielle"
}
```

## Personnalisation avancée

Pour ajouter vos propres prompts spécialisés ou de nouveaux domaines:

1. Modifiez le dictionnaire `DOMAIN_PROMPTS` dans `inference.py`
2. Ajoutez le nouveau domaine dans la fonction `get_domain_key()`
3. Ajoutez des exemples dans `data/training_examples.json`
4. Exécutez `create_training_dataset.py` puis `train.py` pour fine-tuner le modèle

Pour optimiser les performances :
- Sur les machines avec peu de RAM, utilisez les modèles de la catégorie "tiny"
- Sur les GPU, modifiez la configuration `device_map` dans `inference.py`
- Pour un déploiement en production, considérez l'utilisation de vLLM ou TensorRT-LLM

---

## Dépannage

**Problème de mémoire** : Si vous obtenez des erreurs OOM (Out of Memory) :
1. Utilisez un modèle plus petit (`--category tiny`)
2. Augmentez la limite de mémoire dans `docker-compose.yml`
3. Essayez une quantification plus agressive (4-bit)

**Modèle lent** : Pour améliorer les performances :
1. Utilisez un modèle plus petit
2. Activez le cache de requêtes dans `inference.py`
3. Réduisez `max_length` dans les requêtes 