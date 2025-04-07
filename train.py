#!/usr/bin/env python
# Script pour fine-tuner le modèle DistilBERT sur les données d'entraînement préparées

import os
import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from tqdm.auto import tqdm

# Définir les domaines pour la classification
DOMAINS = [
    "informatique_reseaux",
    "automatismes_info_industrielle",
    "finance",
    "civil_btp",
    "genie_industriel"
]

class CVJobDataset(Dataset):
    """Dataset pour les CV et offres d'emploi"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):  # Réduit la longueur maximale
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Supprimer la dimension batch ajoutée par tokenizer
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return inputs

def compute_metrics(pred):
    """Calcule les métriques pour évaluer le modèle"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuner DistilBERT pour la classification de CV et offres d'emploi")
    parser.add_argument("--data", type=str, default="data/training_data.csv",
                        help="Fichier CSV contenant les données d'entraînement")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased",  # Corrigé le chemin du modèle
                        help="Modèle de base à fine-tuner")
    parser.add_argument("--output_dir", type=str, default="models/cv_classifier",
                        help="Répertoire pour sauvegarder le modèle")
    parser.add_argument("--batch_size", type=int, default=4,  # Batch size réduit
                        help="Taille du batch pour l'entraînement")
    parser.add_argument("--epochs", type=int, default=3,  # Nombre d'époques réduit
                        help="Nombre d'époques d'entraînement")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Taux d'apprentissage")
    parser.add_argument("--max_length", type=int, default=128,  # Longueur maximale réduite
                        help="Longueur maximale des séquences")
    parser.add_argument("--eval_ratio", type=float, default=0.2,
                        help="Ratio des données à utiliser pour l'évaluation")
    parser.add_argument("--use_cpu", action="store_true", 
                        help="Forcer l'utilisation du CPU même si GPU disponible")
    
    return parser.parse_args()

def prepare_data(data_path, eval_ratio=0.2):
    """Prépare les données pour l'entraînement et l'évaluation"""
    # Charger les données
    df = pd.read_csv(data_path)
    
    # Créer un mapping domaine -> indice
    domain_to_idx = {domain: idx for idx, domain in enumerate(DOMAINS)}
    
    # Préparer les textes et les labels
    texts = df['input'].tolist()
    labels = [domain_to_idx.get(domain, 0) for domain in df['domain']]
    
    # Diviser en ensembles d'entraînement et d'évaluation
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        texts, labels, test_size=eval_ratio, stratify=labels, random_state=42
    )
    
    print(f"Données d'entraînement: {len(train_texts)} exemples")
    print(f"Données d'évaluation: {len(eval_texts)} exemples")
    
    return (train_texts, train_labels), (eval_texts, eval_labels), domain_to_idx

def train_model(args):
    """Entraîne le modèle avec les paramètres spécifiés"""
    # Définir le device
    if args.use_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Utilisation du device: {device}")
    
    # Préparer les données
    train_data, eval_data, domain_to_idx = prepare_data(args.data, args.eval_ratio)
    
    # Initialiser le tokenizer et le modèle
    tokenizer = DistilBertTokenizer.from_pretrained(args.model)
    model = DistilBertForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(DOMAINS)
    )
    
    # Créer les datasets
    train_dataset = CVJobDataset(
        train_data[0], train_data[1], tokenizer, max_length=args.max_length
    )
    eval_dataset = CVJobDataset(
        eval_data[0], eval_data[1], tokenizer, max_length=args.max_length
    )
    
    # Définir les arguments d'entraînement
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=0,  # Pas de warmup pour un petit dataset
        weight_decay=0.01,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,  # Log plus fréquemment
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        push_to_hub=False,
        learning_rate=args.learning_rate,
        no_cuda=args.use_cpu,  # Respecter l'option use_cpu
        report_to="none",  # Désactiver les rapports wandb/tensorboard
    )
    
    # Initialiser l'entraîneur
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  # Réduire la patience
    )
    
    # Entraîner le modèle
    print("Début de l'entraînement...")
    trainer.train()
    
    # Évaluer le modèle final
    metrics = trainer.evaluate()
    print(f"Métriques finales: {metrics}")
    
    # Sauvegarder le modèle, le tokenizer et le mapping des domaines
    model_path = os.path.join(args.output_dir, "final")
    os.makedirs(model_path, exist_ok=True)
    
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    # Sauvegarder le mapping des domaines
    idx_to_domain = {str(idx): domain for domain, idx in domain_to_idx.items()}
    domain_mapping_path = os.path.join(model_path, "domain_mapping.json")
    import json
    with open(domain_mapping_path, 'w', encoding='utf-8') as f:
        json.dump(idx_to_domain, f, ensure_ascii=False, indent=2)
    
    print(f"Modèle sauvegardé dans {model_path}")
    return model, tokenizer, domain_to_idx

def main():
    args = parse_args()
    train_model(args)

if __name__ == "__main__":
    main() 