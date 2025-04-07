#!/usr/bin/env python
# Système de feedback pour le matching CV-offres d'emploi

import os
import json
import argparse
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional
import httpx
import asyncio

# Configuration
DEFAULT_API_URL = "http://localhost:8000/api/v1"
DEFAULT_LLM_URL = "http://localhost:8001"
FEEDBACK_DIR = "data/feedback"
FEEDBACK_HISTORY = "data/feedback/history.json"

def parse_args():
    parser = argparse.ArgumentParser(description="Système de feedback pour le matching CV-offres d'emploi")
    parser.add_argument("--results", type=str, required=True, help="Chemin vers le fichier JSON des résultats de matching")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="URL de l'API backend")
    parser.add_argument("--llm-url", default=DEFAULT_LLM_URL, help="URL du service LLM")
    parser.add_argument("--view", action="store_true", help="Visualiser les résultats et donner un feedback")
    parser.add_argument("--stats", action="store_true", help="Afficher les statistiques de feedback")
    return parser.parse_args()

def load_results(results_path: str) -> Dict[str, Any]:
    """Charger les résultats depuis un fichier JSON"""
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Erreur lors du chargement des résultats: {e}")
        return {}

def generate_session_id(results: Dict[str, Any]) -> str:
    """Générer un ID de session unique basé sur le contenu"""
    if not results:
        return hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()
    
    # Utiliser les informations du CV et la date pour générer un ID unique
    cv_info = ""
    if "cv_analysis" in results:
        cv_domain = results["cv_analysis"].get("main_domain", "")
        cv_skills = ",".join(results["cv_analysis"].get("skills", []))
        cv_info = f"{cv_domain}|{cv_skills}"
    
    timestamp = results.get("timestamp", datetime.now().isoformat())
    
    content = f"{cv_info}|{timestamp}"
    return hashlib.md5(content.encode()).hexdigest()

def init_feedback_store():
    """Initialiser le stockage des feedbacks"""
    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    
    if not os.path.exists(FEEDBACK_HISTORY):
        with open(FEEDBACK_HISTORY, 'w', encoding='utf-8') as f:
            json.dump({"sessions": []}, f)

def save_feedback(session_id: str, results: Dict[str, Any], feedback: Dict[str, Any]):
    """Sauvegarder le feedback de l'utilisateur"""
    # Charger l'historique existant
    with open(FEEDBACK_HISTORY, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    # Ajouter le feedback à l'historique
    session_entry = {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "cv_domain": results.get("cv_analysis", {}).get("main_domain", "unknown"),
        "feedback": feedback
    }
    
    # Vérifier si cette session existe déjà
    existing_idx = None
    for i, session in enumerate(history["sessions"]):
        if session["session_id"] == session_id:
            existing_idx = i
            break
    
    if existing_idx is not None:
        # Mettre à jour la session existante
        history["sessions"][existing_idx] = session_entry
    else:
        # Ajouter une nouvelle session
        history["sessions"].append(session_entry)
    
    # Sauvegarder l'historique mis à jour
    with open(FEEDBACK_HISTORY, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=4)
    
    # Sauvegarder également une copie détaillée pour cette session
    session_file = os.path.join(FEEDBACK_DIR, f"{session_id}.json")
    with open(session_file, 'w', encoding='utf-8') as f:
        full_data = {
            "results": results,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat()
        }
        json.dump(full_data, f, ensure_ascii=False, indent=4)
    
    print(f"✅ Feedback enregistré avec succès (ID: {session_id})")

async def send_feedback_to_api(api_url: str, session_id: str, feedback: Dict[str, Any]):
    """Envoyer le feedback à l'API pour améliorer le modèle"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{api_url}/feedback/matching",
                json={
                    "session_id": session_id,
                    "feedback": feedback
                }
            )
            
            if response.status_code == 200:
                print("✅ Feedback envoyé à l'API avec succès")
                return True
            else:
                print(f"❌ Erreur lors de l'envoi du feedback à l'API: {response.status_code}")
                print(f"Détails: {response.text}")
                return False
    except Exception as e:
        print(f"Exception lors de l'envoi du feedback: {str(e)}")
        return False

def collect_feedback_for_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Collecter le feedback de l'utilisateur pour les résultats de matching"""
    feedback = {"job_ratings": [], "overall_rating": 0, "comments": ""}
    
    if not results or "results" not in results or not results["results"]:
        print("Aucun résultat à évaluer.")
        return feedback
    
    print("\n" + "=" * 60)
    print("SYSTÈME DE FEEDBACK - MATCHING CV-OFFRES")
    print("=" * 60)
    
    # Informations sur le CV analysé
    cv_analysis = results.get("cv_analysis", {})
    main_domain = cv_analysis.get("main_domain", "Non spécifié")
    skills = cv_analysis.get("skills", [])
    
    print(f"Domaine principal: {main_domain}")
    if skills:
        print(f"Compétences identifiées: {', '.join(skills)}")
    print("-" * 60)
    
    # Demander un feedback pour chaque offre
    job_results = results["results"]
    
    print("Veuillez évaluer la pertinence de chaque offre (1-5, où 5 est le plus pertinent):")
    
    for i, job in enumerate(job_results):
        print(f"\nOffre {i+1}: {job['title']} - {job['company']}")
        print(f"Score du système: {job['score']:.2f}")
        
        # Demander une note
        while True:
            try:
                rating = int(input(f"Votre note (1-5): "))
                if 1 <= rating <= 5:
                    break
                else:
                    print("La note doit être entre 1 et 5.")
            except ValueError:
                print("Veuillez entrer un nombre entre 1 et 5.")
        
        # Demander un commentaire optionnel
        comment = input("Commentaire (optionnel): ")
        
        # Ajouter le feedback pour cette offre
        job_feedback = {
            "job_id": job.get("id", f"job_{i}"),
            "title": job["title"],
            "system_score": job["score"],
            "user_rating": rating,
            "comment": comment
        }
        
        feedback["job_ratings"].append(job_feedback)
    
    # Demander une évaluation globale
    print("\n" + "-" * 60)
    while True:
        try:
            overall = int(input("Évaluation globale des résultats (1-5): "))
            if 1 <= overall <= 5:
                feedback["overall_rating"] = overall
                break
            else:
                print("L'évaluation doit être entre 1 et 5.")
        except ValueError:
            print("Veuillez entrer un nombre entre 1 et 5.")
    
    # Demander un commentaire général
    feedback["comments"] = input("Commentaires généraux (forces/faiblesses du matching): ")
    
    # Ajouter des métadonnées
    feedback["domain"] = main_domain
    feedback["skills"] = skills
    feedback["feedback_date"] = datetime.now().isoformat()
    
    return feedback

def display_feedback_stats():
    """Afficher les statistiques de feedback"""
    if not os.path.exists(FEEDBACK_HISTORY):
        print("Aucun historique de feedback disponible.")
        return
    
    with open(FEEDBACK_HISTORY, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    sessions = history.get("sessions", [])
    if not sessions:
        print("Aucun feedback enregistré.")
        return
    
    print("\n" + "=" * 60)
    print("STATISTIQUES DE FEEDBACK")
    print("=" * 60)
    
    print(f"Nombre total de sessions de feedback: {len(sessions)}")
    
    # Statistiques par domaine
    domain_stats = {}
    overall_ratings = []
    job_ratings = []
    
    for session in sessions:
        domain = session.get("cv_domain", "unknown")
        if domain not in domain_stats:
            domain_stats[domain] = {
                "count": 0,
                "overall_ratings": [],
                "job_ratings": []
            }
        
        domain_stats[domain]["count"] += 1
        
        feedback = session.get("feedback", {})
        overall_rating = feedback.get("overall_rating", 0)
        
        if overall_rating > 0:
            domain_stats[domain]["overall_ratings"].append(overall_rating)
            overall_ratings.append(overall_rating)
        
        for job_rating in feedback.get("job_ratings", []):
            user_rating = job_rating.get("user_rating", 0)
            system_score = job_rating.get("system_score", 0)
            
            if user_rating > 0:
                domain_stats[domain]["job_ratings"].append({
                    "user_rating": user_rating,
                    "system_score": system_score
                })
                job_ratings.append({
                    "user_rating": user_rating,
                    "system_score": system_score
                })
    
    # Afficher les statistiques globales
    if overall_ratings:
        avg_overall = sum(overall_ratings) / len(overall_ratings)
        print(f"\nNote globale moyenne: {avg_overall:.2f}/5")
    
    # Calculer la corrélation moyenne entre scores système et notes utilisateur
    if job_ratings:
        correlation_sum = 0
        correlation_count = 0
        
        for rating in job_ratings:
            # Normaliser les scores système (0-1) vers une échelle de 1-5
            norm_system_score = 1 + (rating["system_score"] * 4)  # Map 0->1, 1->5
            correlation_sum += abs(norm_system_score - rating["user_rating"])
            correlation_count += 1
        
        if correlation_count > 0:
            avg_difference = correlation_sum / correlation_count
            # Convertir en pourcentage d'accord (0 difference = 100%, 4 difference = 0%)
            agreement_pct = max(0, (1 - (avg_difference / 4))) * 100
            print(f"Taux d'accord moyen entre système et utilisateur: {agreement_pct:.1f}%")
    
    # Afficher les statistiques par domaine
    print("\nStatistiques par domaine:")
    for domain, stats in domain_stats.items():
        print(f"\n- {domain.upper()} ({stats['count']} sessions)")
        
        if stats["overall_ratings"]:
            domain_avg = sum(stats["overall_ratings"]) / len(stats["overall_ratings"])
            print(f"  Note globale moyenne: {domain_avg:.2f}/5")
        
        if stats["job_ratings"]:
            domain_corr_sum = 0
            domain_corr_count = 0
            
            for rating in stats["job_ratings"]:
                norm_system_score = 1 + (rating["system_score"] * 4)
                domain_corr_sum += abs(norm_system_score - rating["user_rating"])
                domain_corr_count += 1
            
            if domain_corr_count > 0:
                domain_avg_diff = domain_corr_sum / domain_corr_count
                domain_agreement = max(0, (1 - (domain_avg_diff / 4))) * 100
                print(f"  Taux d'accord: {domain_agreement:.1f}%")

async def main_async():
    args = parse_args()
    
    # Initialiser le stockage de feedback
    init_feedback_store()
    
    if args.stats:
        # Afficher les statistiques
        display_feedback_stats()
        return
    
    # Charger les résultats
    results = load_results(args.results)
    if not results:
        print("Impossible de charger les résultats.")
        return
    
    # Générer un ID de session
    session_id = generate_session_id(results)
    
    if args.view:
        # Collecter le feedback
        feedback = collect_feedback_for_results(results)
        
        # Sauvegarder le feedback localement
        save_feedback(session_id, results, feedback)
        
        # Envoyer le feedback à l'API
        await send_feedback_to_api(args.api_url, session_id, feedback)
    else:
        print("Utilisez l'option --view pour visualiser les résultats et donner un feedback.")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main() 