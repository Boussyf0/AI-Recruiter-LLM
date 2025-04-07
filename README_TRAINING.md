# Guide d'entraînement pour l'IA Recruiter

Ce guide explique comment utiliser les scripts fournis pour préparer un dataset d'entraînement, fine-tuner le modèle DistilBERT, et générer des embeddings pour les CV et offres d'emploi.

## Prérequis

Avant de commencer, assurez-vous d'avoir installé les dépendances nécessaires:

```bash
pip install pandas numpy torch transformers scikit-learn tqdm
```

Pour télécharger les datasets depuis Kaggle, vous aurez également besoin de l'API Kaggle:

```bash
pip install kaggle
```

Et configurez votre API Kaggle en créant un fichier `~/.kaggle/kaggle.json` avec vos identifiants:

```json
{
  "username": "votre_nom_utilisateur",
  "key": "votre_clé_api"
}
```

## Étape 1: Création du dataset d'entraînement

Le script `create_training_dataset.py` vous permet de créer un dataset d'entraînement en utilisant des exemples provenant de diverses sources, y compris des datasets Kaggle.

### Utilisation basique:

```bash
python create_training_dataset.py --output data/training_data.csv
```

### Télécharger et utiliser les datasets Kaggle recommandés:

```bash
python create_training_dataset.py --kaggle --output data/training_data_kaggle.csv
```

### Spécifier des sources personnalisées:

```bash
python create_training_dataset.py --sources data/mes_exemples.json data/mes_cv.csv --output data/training_data_custom.csv
```

### Options avancées:

```bash
python create_training_dataset.py --kaggle --domains informatique_reseaux finance --expand 10 --output data/training_data_specific.csv
```

### Arguments disponibles:

- `--sources`: Liste des fichiers sources (JSON ou CSV) à utiliser
- `--output`: Fichier CSV de sortie
- `--expand`: Nombre d'exemples à générer par domaine
- `--domains`: Domaines spécifiques à inclure
- `--kaggle`: Télécharger et utiliser les datasets Kaggle recommandés
- `--kaggle-datasets`: IDs spécifiques des datasets Kaggle à utiliser

## Étape 2: Entraînement du modèle

Le script `train.py` permet de fine-tuner un modèle DistilBERT sur votre dataset d'entraînement.

### Utilisation basique:

```bash
python train.py --data data/training_data.csv --output_dir models/cv_classifier
```

### Options avancées:

```bash
python train.py --data data/training_data.csv --model distilbert/distilbert-base-uncased --batch_size 8 --epochs 3 --learning_rate 2e-5 --output_dir models/cv_classifier_optimized
```

### Arguments disponibles:

- `--data`: Fichier CSV contenant les données d'entraînement
- `--model`: Modèle de base à fine-tuner
- `--output_dir`: Répertoire pour sauvegarder le modèle
- `--batch_size`: Taille du batch pour l'entraînement
- `--epochs`: Nombre d'époques d'entraînement
- `--learning_rate`: Taux d'apprentissage
- `--max_length`: Longueur maximale des séquences
- `--eval_ratio`: Ratio des données à utiliser pour l'évaluation

## Étape 3: Génération d'embeddings

Le script `create_embeddings.py` permet de générer des embeddings à partir de CV ou d'offres d'emploi en utilisant le modèle entraîné.

### Utilisation basique:

```bash
python create_embeddings.py --input_file data/cv_list.csv --output_file data/cv_embeddings.npz
```

### Options avancées:

```bash
python create_embeddings.py --model_path models/cv_classifier/final --input_file data/job_offers.csv --text_column description --output_file data/job_embeddings.npz --batch_size 16 --device cuda
```

### Arguments disponibles:

- `--model_path`: Chemin vers le modèle entraîné
- `--input_file`: Fichier CSV contenant les textes (CV ou offres d'emploi)
- `--output_file`: Fichier de sortie pour les embeddings (format NPZ)
- `--text_column`: Nom de la colonne contenant les textes
- `--batch_size`: Taille des batchs pour la génération d'embeddings
- `--max_length`: Longueur maximale des séquences
- `--device`: Dispositif à utiliser (cuda ou cpu)

## Exemple de flux de travail complet

Voici un exemple de flux de travail complet pour entraîner un modèle et générer des embeddings:

```bash
# 1. Télécharger les datasets Kaggle et créer un dataset d'entraînement
python create_training_dataset.py --kaggle --output data/training_data_kaggle.csv

# 2. Entraîner le modèle
python train.py --data data/training_data_kaggle.csv --output_dir models/cv_classifier --batch_size 16 --epochs 5

# 3. Générer des embeddings pour des CV
python create_embeddings.py --model_path models/cv_classifier/final --input_file data/cv_list.csv --output_file data/cv_embeddings.npz

# 4. Générer des embeddings pour des offres d'emploi
python create_embeddings.py --model_path models/cv_classifier/final --input_file data/job_offers.csv --text_column description --output_file data/job_embeddings.npz
```

## Datasets Kaggle recommandés

Le script `create_training_dataset.py` est configuré pour utiliser les datasets Kaggle suivants:

1. **Job Description Dataset** (ravindrasinghrana/job-description-dataset)
   - Contient plus de 2000 descriptions de postes IT avec compétences requises

2. **Resume Dataset** (gauravduttakiit/resume-dataset)
   - Contient ~2400 CV catégorisés en 25 professions différentes

3. **Job Skills Extraction Dataset** (elanzaelani/job-skills-extraction-dataset)
   - Annotations d'offres d'emploi avec compétences extraites

4. **Tech Job Skills Dataset** (thedevastator/jobs-dataset-for-recommender-engines-and-skill-a)
   - Ensemble riche de compétences techniques par poste

## Notes importantes

- Le processus d'entraînement peut prendre du temps selon la taille du dataset et les ressources disponibles.
- Pour de meilleurs résultats, utilisez un GPU pour l'entraînement et la génération d'embeddings.
- Les modèles entraînés sont sauvegardés dans le répertoire spécifié par `--output_dir/final`.
- Les embeddings générés sont sauvegardés au format NPZ, qui peut être chargé avec `numpy.load()`. 