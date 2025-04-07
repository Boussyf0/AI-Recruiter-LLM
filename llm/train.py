#!/usr/bin/env python
# Script de fine-tuning de LLM pour les tâches de recrutement
# Utilise PEFT/LoRA pour réduire la demande en ressources

import os
import argparse
from datasets import load_dataset, Dataset
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

# Modèle par défaut adapté aux instructions
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune un LLM pour les tâches de recrutement")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL, 
                        help="Modèle de base à utiliser pour le fine-tuning")
    parser.add_argument("--dataset_path", type=str, default="data/training_data.csv",
                        help="Chemin vers le dataset de fine-tuning")
    parser.add_argument("--output_dir", type=str, default="models/recruiter-llm",
                        help="Répertoire où sauvegarder le modèle")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Nombre d'époques d'entraînement")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Taux d'apprentissage")
    parser.add_argument("--per_device_batch_size", type=int, default=2,
                        help="Taille de batch par appareil")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="Rang de la matrice LoRA")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="Alpha de LoRA")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="Dropout de LoRA")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Longueur maximale de séquence")
    
    return parser.parse_args()

def load_training_data(dataset_path):
    """Charge le jeu de données d'entraînement"""
    print(f"Loading dataset from {dataset_path}")
    
    if dataset_path.endswith('.csv'):
        # Si c'est un fichier CSV
        df = pd.read_csv(dataset_path)
        return Dataset.from_pandas(df)
    else:
        # Si c'est un répertoire ou un dataset HuggingFace
        return load_dataset(dataset_path)

def create_sample_dataset():
    """Crée un exemple de dataset si aucun n'est fourni"""
    print("Creating sample recruitment dataset")
    
    # Exemples de prompts pour le recrutement
    samples = [
        {
            "instruction": "Évalue ce CV pour un poste de développeur Python.",
            "input": "John Doe\nDéveloppeur logiciel avec 5 ans d'expérience\nCompétences: Python, Django, Flask, SQL, Git\nExpérience:\n- Développeur senior chez TechCorp (2019-2023)\n- Développeur junior chez StartupXYZ (2017-2019)",
            "output": "Le candidat présente un profil solide pour un poste de développeur Python, avec 5 ans d'expérience pertinente. Points forts: expérience avec des frameworks web Python (Django, Flask), connaissance des bases de données (SQL) et maîtrise de Git. Son passage de junior à senior montre une progression de carrière. Recommandation: entretien technique recommandé pour évaluer son niveau de compétence en Python et sa capacité à résoudre des problèmes complexes."
        },
        {
            "instruction": "Prépare une liste de questions d'entretien pour un poste d'ingénieur DevOps.",
            "input": "Nous recherchons un ingénieur DevOps expérimenté avec des compétences en AWS, Docker et Kubernetes.",
            "output": "Questions techniques:\n1. Décrivez votre expérience avec AWS. Quels services avez-vous utilisés et dans quels contextes?\n2. Comment orchestrez-vous des conteneurs avec Kubernetes? Donnez des exemples concrets.\n3. Expliquez comment vous mettriez en place un pipeline CI/CD pour une application web.\n4. Comment gérez-vous la sécurité dans un environnement conteneurisé?\n5. Quelle est votre approche pour le monitoring et l'observabilité?\n\nQuestions comportementales:\n1. Décrivez une situation où vous avez dû résoudre un problème d'infrastructure critique. Comment avez-vous procédé?\n2. Comment collaborez-vous avec les équipes de développement?\n3. Comment restez-vous à jour avec les technologies DevOps en constante évolution?"
        },
        {
            "instruction": "Compare ces deux candidats pour un poste de data scientist.",
            "input": "Candidat A: PhD en statistiques, 2 ans d'expérience, publications scientifiques, maîtrise de Python et R.\nCandidat B: Master en informatique, 5 ans d'expérience en entreprise, projets concrets en production, maîtrise de Python et SQL.",
            "output": "Candidat A présente un profil académique solide avec une formation théorique approfondie en statistiques, ce qui est précieux pour la recherche et le développement d'algorithmes complexes. Ses publications démontrent sa capacité à innover.\n\nCandidat B offre une expérience pratique plus importante en environnement d'entreprise avec des projets déployés en production, suggérant une meilleure compréhension des contraintes business et des cycles de développement.\n\nRecommandation: Si le poste nécessite des recherches avancées et le développement de nouveaux algorithmes, le Candidat A serait préférable. Si l'objectif est d'implémenter rapidement des solutions dans un environnement de production, le Candidat B semble plus adapté."
        }
    ]
    
    # Créer un dataframe puis le convertir en dataset
    df = pd.DataFrame(samples)
    return Dataset.from_pandas(df)

def format_instruction_dataset(dataset):
    """Formate le dataset pour l'apprentissage par instruction"""
    
    def format_instruction(example):
        # Format pour les modèles Mistral-Instruct ou similaires
        if example.get("instruction") and example.get("input") and example.get("output"):
            # Format: <s>[INST] instruction \n input [/INST] output </s>
            formatted = f"<s>[INST] {example['instruction']}\n{example['input']} [/INST] {example['output']} </s>"
        else:
            # Cas où nous avons seulement une paire question-réponse
            question = example.get("question", example.get("prompt", ""))
            answer = example.get("answer", example.get("response", ""))
            formatted = f"<s>[INST] {question} [/INST] {answer} </s>"
        
        return {"formatted_text": formatted}
    
    return dataset.map(format_instruction)

def tokenize_dataset(dataset, tokenizer, max_length):
    """Tokenize le dataset formatté"""
    
    def tokenize(example):
        return tokenizer(
            example["formatted_text"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
    
    return dataset.map(tokenize, remove_columns=["formatted_text"])

def main():
    args = parse_args()
    
    # Vérifier si le répertoire de sortie existe, sinon le créer
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configuration pour charger le modèle en 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16"
    )
    
    # Charger le tokenizer et le modèle
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Configurer le tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    
    # Préparer le modèle pour l'entraînement en 4-bit avec PEFT
    model = prepare_model_for_kbit_training(model)
    
    # Configuration LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # Appliquer PEFT au modèle
    model = get_peft_model(model, peft_config)
    
    # Charger le jeu de données d'entraînement
    try:
        train_dataset = load_training_data(args.dataset_path)
        print(f"Loaded dataset with {len(train_dataset)} examples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Creating sample dataset instead")
        train_dataset = create_sample_dataset()
    
    # Formater et tokeniser le jeu de données
    formatted_dataset = format_instruction_dataset(train_dataset)
    tokenized_dataset = tokenize_dataset(formatted_dataset, tokenizer, args.max_seq_length)
    
    # Split train/eval
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    
    # Préparer le collator de données
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Configuration d'entraînement
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        weight_decay=0.05,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        logging_steps=50,
        gradient_accumulation_steps=4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
        gradient_checkpointing=True
    )
    
    # Initialiser le Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Lancer l'entraînement
    print("Starting training...")
    trainer.train()
    
    # Sauvegarder le modèle et le tokenizer
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"Training complete. Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()

