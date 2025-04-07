#!/usr/bin/env python
# Interface utilisateur Streamlit pour le matching CV-offres d'emploi

import streamlit as st
import os
import json
import pandas as pd
import tempfile
from datetime import datetime
from improve_matching import improved_cv_matching, DEFAULT_API_URL, DEFAULT_LLM_URL
import hashlib
import sys
import subprocess
import httpx  # Ajout de l'import httpx pour les appels API
from indeed_scraper import scrape_indeed_jobs, generate_indeed_query, match_job_with_cv
from pathlib import Path
import time

# Importer le service d'embeddings local
try:
    from local_embeddings import LocalEmbeddingService
    LOCAL_EMBEDDINGS_AVAILABLE = True
except ImportError:
    LOCAL_EMBEDDINGS_AVAILABLE = False

st.set_page_config(
    page_title="AI Recruiter - CV Matching",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fonction pour exécuter le matching asynchrone
def run_matching(cv_path=None, cv_text=None, api_url=None, llm_url=None, top_k=5, min_score=0.0, strict_threshold=False):
    """Lance le processus de matching et récupère les résultats"""
    try:
        # Afficher un message de progrès
        progress_placeholder = st.empty()
        progress_placeholder.info("Analyse du CV et recherche des meilleures correspondances...")
        
        # Mettre à jour le message pour indiquer la recherche dans toutes les sources
        time.sleep(0.5)  # Petit délai pour que l'utilisateur puisse lire le premier message
        progress_placeholder.info("Recherche en cours dans la base vectorielle et les offres scrapées récentes...")

        # Appeler la fonction de matching améliorée
        results = improved_cv_matching(
            cv_path=cv_path,
            cv_text=cv_text,
            api_url=api_url,
            llm_url=llm_url,
            top_k=top_k,
            min_score=min_score,
            strict_threshold=strict_threshold
        )
        
        # Effacer le message de progression
        progress_placeholder.empty()
        
        # Vérifier si des résultats ont été obtenus
        if results and "results" in results and results["results"]:
            st.success(f"✅ {len(results['results'])} offres correspondantes trouvées!")
        else:
            st.warning("❌ Aucune offre correspondante trouvée. Essayez d'ajuster les paramètres ou d'utiliser un autre CV.")
        
        return results
    
    except Exception as e:
        st.error(f"❌ Erreur lors de l'analyse: {str(e)}")
        return None

# Fonction pour afficher les résultats sous forme de tableau
def display_results(results, show_details=False):
    if not results or "results" not in results or not results["results"]:
        st.warning("Aucun résultat de matching trouvé.")
        return
    
    job_results = results["results"]
    
    # Créer un DataFrame pour affichage
    df_results = pd.DataFrame([
        {
            "Rang": job["rank"],
            "Titre": job["title"],
            "Entreprise": job["company"],
            "Domaine": job["domain"],
            "Score": job["score"],
            "Bonus Domaine": job["domain_bonus"],
            "Bonus Compétences": job["skills_bonus"],
            "Match": "✅" if job["domain_match"] == "✅" else "❌"
        }
        for job in job_results
    ])
    
    # Coloration conditionnelle
    def color_score(val):
        color = 'white'
        if val >= 0.7:
            color = 'lightgreen'
        elif val >= 0.5:
            color = 'lightyellow'
        elif val < 0.3:
            color = 'lightcoral'
        return f'background-color: {color}'
    
    # Appliquer le style
    styled_df = df_results.style.map(color_score, subset=['Score'])
    
    # Afficher le tableau de résultats
    st.dataframe(styled_df, use_container_width=True)
    
    # Afficher les détails si demandé
    if show_details and job_results:
        st.subheader("Détails des offres d'emploi")
        
        # Afficher les détails de chaque offre avec possibilité de donner un feedback
        feedback_data = {"job_ratings": [], "overall_rating": 0, "comments": ""}
        
        for job in job_results:
            with st.expander(f"{job['rank']}. {job['title']} - {job['company']} (Score: {job['score']:.2f})"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Entreprise:** {job['company']}")
                    st.markdown(f"**Domaine:** {job['domain']} {job['domain_match']}")
                    st.markdown(f"**Score brut:** {job['raw_score']:.2f}")
                    st.markdown(f"**Bonus domaine:** {job['domain_bonus']:.2f}")
                    st.markdown(f"**Bonus compétences:** {job['skills_bonus']:.2f} ({job['matched_skills']} compétences)")
                    st.markdown(f"**Score final:** {job['score']:.2f}")
                    st.markdown("**Description:**")
                    st.markdown(job['description'])
                
                with col2:
                    st.markdown("### Votre feedback")
                    rating = st.select_slider(
                        f"Pertinence",
                        options=[1, 2, 3, 4, 5],
                        value=3,
                        key=f"rating_{job['rank']}"
                    )
                    comment = st.text_area("Commentaire", key=f"comment_{job['rank']}", height=100)
                    
                    # Ajouter à la structure de feedback
                    feedback_data["job_ratings"].append({
                        "job_id": job.get("id", f"job_{job['rank']}"),
                        "title": job["title"],
                        "system_score": job["score"],
                        "user_rating": rating,
                        "comment": comment
                    })
        
        # Feedback global
        st.subheader("Votre feedback global")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            overall_rating = st.select_slider(
                "Qualité globale des résultats",
                options=[1, 2, 3, 4, 5],
                value=3,
                key="overall_rating"
            )
            feedback_data["overall_rating"] = overall_rating
        
        with col2:
            overall_comment = st.text_area("Commentaires ou suggestions d'amélioration", height=100, key="overall_comment")
            feedback_data["comments"] = overall_comment
        
        # Enregistrer le feedback
        if st.button("Envoyer mon feedback"):
            # Ajouter les métadonnées du feedback
            cv_analysis = results.get("cv_analysis", {})
            feedback_data["domain"] = cv_analysis.get("main_domain", "unknown")
            feedback_data["skills"] = cv_analysis.get("skills", [])
            feedback_data["feedback_date"] = datetime.now().isoformat()
            
            # Générer un ID de session
            session_id = hashlib.md5((str(results.get("timestamp", "")) + str(datetime.now().timestamp())).encode()).hexdigest()
            
            # Sauvegarder le feedback
            feedback_dir = "data/feedback"
            os.makedirs(feedback_dir, exist_ok=True)
            
            feedback_history_file = os.path.join(feedback_dir, "history.json")
            if not os.path.exists(feedback_history_file):
                with open(feedback_history_file, 'w', encoding='utf-8') as f:
                    json.dump({"sessions": []}, f)
            
            # Charger l'historique existant
            with open(feedback_history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            # Ajouter le feedback à l'historique
            session_entry = {
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "cv_domain": feedback_data["domain"],
                "feedback": feedback_data
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
            
            # Sauvegarder l'historique
            with open(feedback_history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=4)
            
            # Sauvegarder également une copie détaillée
            session_file = os.path.join(feedback_dir, f"{session_id}.json")
            with open(session_file, 'w', encoding='utf-8') as f:
                full_data = {
                    "results": results,
                    "feedback": feedback_data,
                    "timestamp": datetime.now().isoformat()
                }
                json.dump(full_data, f, ensure_ascii=False, indent=4)
            
            st.success("Merci ! Votre feedback a été enregistré et aidera à améliorer le système.")
        
        # Offrir la possibilité de télécharger les résultats
        results_json = json.dumps(results, ensure_ascii=False, indent=4)
        st.download_button(
            label="Télécharger les résultats (JSON)",
            data=results_json,
            file_name=f"cv_matching_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# Fonction pour afficher l'analyse du CV
def display_cv_analysis(results):
    if not results or "cv_analysis" not in results:
        return
    
    cv_analysis = results["cv_analysis"]
    
    st.subheader("Analyse du CV")
    
    # Afficher le domaine principal
    main_domain = cv_analysis.get("main_domain", "Non déterminé")
    st.markdown(f"**Domaine principal:** {main_domain}")
    
    # Afficher les scores de domaine
    domain_scores = cv_analysis.get("domain_scores", {})
    if domain_scores:
        st.markdown("**Scores par domaine:**")
        
        # Créer un graphique pour visualiser les scores de domaine
        domain_df = pd.DataFrame({
            "Domaine": list(domain_scores.keys()),
            "Score": list(domain_scores.values())
        })
        
        domain_df = domain_df.sort_values("Score", ascending=False)
        st.bar_chart(domain_df.set_index("Domaine"))
    
    # Afficher les compétences identifiées
    skills = cv_analysis.get("skills", [])
    if skills:
        st.markdown("**Compétences identifiées:**")
        # Afficher les compétences sous forme de "pills"
        cols = st.columns(4)
        for i, skill in enumerate(skills):
            col_idx = i % 4
            cols[col_idx].markdown(f"<div style='background-color: #f0f2f6; padding: 8px; border-radius: 16px; margin: 4px; display: inline-block;'>{skill}</div>", unsafe_allow_html=True)

# Fonction pour afficher les offres d'emploi récupérées depuis Indeed
def display_indeed_jobs(jobs, show_details=True):
    """Affiche les offres d'emploi récupérées depuis Indeed"""
    if not jobs:
        st.info("Aucune offre d'emploi trouvée sur Indeed.")
        return
    
    # Créer un DataFrame pour affichage
    df_results = pd.DataFrame([
        {
            "Titre": job["title"],
            "Entreprise": job["company"],
            "Lieu": job["location"],
            "Pays": job.get("country", "N/A"),
            "Score": job.get("match_score", 0.0),
            "Source": "Indeed"
        }
        for job in jobs
    ])
    
    # Fonction pour colorier les scores
    def color_score(val):
        if val >= 0.7:
            return f'background-color: #c6efce; color: #006100'  # Vert
        elif val >= 0.5:
            return f'background-color: #ffeb9c; color: #9c5700'  # Jaune
        else:
            return f'background-color: #ffc7ce; color: #9c0006'  # Rouge
    
    # Appliquer la coloration conditionnelle
    styled_df = df_results.style.applymap(
        color_score, 
        subset=['Score']
    ).format({
        'Score': '{:.2f}'
    })
    
    # Afficher le tableau
    st.dataframe(styled_df, use_container_width=True)
    
    # Afficher les détails
    if show_details and jobs:
        for job in jobs:
            score_color = "green" if job.get('match_score', 0) >= 0.7 else ("orange" if job.get('match_score', 0) >= 0.5 else "red")
            with st.expander(f"{job['title']} - {job['company']} | Score: {job.get('match_score', 0):.2f}"):
                st.markdown(f"**Entreprise:** {job['company']}")
                st.markdown(f"**Lieu:** {job['location']}")
                st.markdown(f"**Score de matching:** <span style='color:{score_color};font-weight:bold'>{job.get('match_score', 0):.2f}</span>", unsafe_allow_html=True)
                
                # Afficher les compétences correspondantes
                if job.get('skill_matches'):
                    st.markdown("**Compétences correspondantes:**")
                    for skill in job.get('skill_matches', []):
                        st.markdown(f"- {skill}")
                
                if job.get('salary'):
                    st.markdown(f"**Salaire:** {job['salary']}")
                st.markdown("**Description:**")
                st.markdown(job['description'])
                if job.get('url'):
                    st.markdown(f"[Voir l'offre sur Indeed]({job['url']})")

# Fonction pour récupérer des offres Indeed qui correspondent au CV
def fetch_indeed_jobs(cv_results, location, max_results=5, country_code="fr", min_match_score=0.4):
    try:
        # Extraire la requête de recherche à partir du CV
        skills = cv_results.get("skills", [])
        domain = cv_results.get("main_domain", "")
        
        # Générer une requête Indeed plus précise basée sur le domaine et les compétences principales
        from indeed_scraper import generate_indeed_query
        query = generate_indeed_query({"cv_analysis": cv_results})
        
        # Créer un fichier temporaire pour la sortie
        output_file = os.path.join(tempfile.gettempdir(), f"indeed_results_{int(time.time())}.json")
        
        cmd = [
            sys.executable, "indeed_scraper.py",
            "--query", query,
            "--location", location,
            "--max-results", str(int(max_results * 2)),  # Demander plus de résultats pour filtrage ultérieur
            "--output", output_file,
            "--verbose"
        ]
        
        # Ajouter le paramètre du pays si spécifié
        if country_code:
            cmd.extend(["--country", country_code])
            
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            st.error(f"Erreur lors de l'exécution du script de scraping: {stderr}")
            return []
        
        # Load results
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                results = json.load(f)
            
            # Delete temporary file
            os.remove(output_file)
            
            # Calculer les scores de compatibilité avec le CV et filtrer les résultats
            jobs = results.get("jobs", [])
            if jobs:
                from indeed_scraper import match_job_with_cv
                
                # Ajouter le score de matching à chaque offre
                for job in jobs:
                    job['match_score'] = match_job_with_cv(job, {"cv_analysis": cv_results})
                
                # Filtrer les offres dont le score est trop bas
                filtered_jobs = [job for job in jobs if job.get('match_score', 0) >= min_match_score]
                
                # Trier par score de matching, du plus élevé au plus faible
                filtered_jobs.sort(key=lambda x: x.get('match_score', 0), reverse=True)
                
                # Limiter au nombre de résultats demandé
                return filtered_jobs[:max_results]
            
            return []
        else:
            st.warning("Aucun résultat n'a été retourné par le script de scraping.")
            return []
            
    except Exception as e:
        st.error(f"Erreur lors de la récupération des offres d'emploi: {str(e)}")
        return []

@st.cache_resource
def get_embedding_service():
    """Initialiser et mettre en cache le service d'embeddings local"""
    try:
        from local_embeddings import LocalEmbeddingService
        return LocalEmbeddingService(
            model_name="paraphrase-multilingual-MiniLM-L12-v2",
            cache_dir="models/embeddings"
        )
    except (ImportError, Exception) as e:
        st.warning(f"Service d'embeddings local non disponible: {str(e)}")
        return None
        
def get_embeddings_local(text, domain=None):
    """Obtenir des embeddings via le service local"""
    service = get_embedding_service()
    if service:
        try:
            return service.get_embeddings(text, domain=domain)
        except Exception as e:
            st.warning(f"Erreur lors de la génération des embeddings: {str(e)}")
    return None

def get_embeddings(text, domain=None):
    """Get embeddings from API or local fallback, avec mise en cache"""
    # Essayer d'abord l'API distante
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                "http://localhost:8000/get-embeddings",
                json={"text": text, "domain": domain}
            )
            if response.status_code == 200:
                return response.json()["embeddings"]
    except Exception as e:
        st.warning(f"Service d'embeddings distant non disponible: {str(e)}")
        
    # Fallback vers le service local mis en cache
    return get_embeddings_local(text, domain)

# Sidebar pour les configurations
st.sidebar.title("Paramètres")

api_url = st.sidebar.text_input("URL de l'API Backend", DEFAULT_API_URL)
llm_url = st.sidebar.text_input("URL du service LLM", DEFAULT_LLM_URL)
top_k = st.sidebar.slider("Nombre d'offres à retourner", 1, 10, 5)
min_score = st.sidebar.slider("Score minimum", 0.0, 1.0, 0.3, 0.05)
strict_threshold = st.sidebar.checkbox("Seuil de pertinence strict", True)
show_details = st.sidebar.checkbox("Afficher les détails des offres", True)

# Option pour scraper Indeed
st.sidebar.markdown("---")
indeed_search = st.sidebar.checkbox("Rechercher aussi sur Indeed", True)

# Sélection du pays
indeed_country = st.sidebar.selectbox(
    "Pays de recherche Indeed",
    ["France", "Maroc"],
    index=0
)

# Code pays pour l'API
indeed_country_code = "fr" if indeed_country == "France" else "ma"

# Sélection de la ville en fonction du pays
if indeed_country == "France":
    indeed_cities = ["", "Paris", "Lyon", "Marseille", "Toulouse", "Nice", "Nantes", 
                     "Strasbourg", "Montpellier", "Bordeaux", "Lille"]
else:  # Maroc
    indeed_cities = ["", "Casablanca", "Rabat", "Marrakech", "Tanger", "Fès", "Meknès", 
                     "Agadir", "Tétouan", "Oujda", "Kénitra", "El Jadida", "Mohammedia"]

indeed_location = st.sidebar.selectbox(
    "Ville",
    indeed_cities,
    index=0
)

# Si aucune ville n'est sélectionnée, utiliser le pays comme localisation
if not indeed_location:
    indeed_location = indeed_country

indeed_max_results = st.sidebar.slider("Nombre d'offres Indeed", 1, 15, 5)
indeed_min_score = st.sidebar.slider("Score minimum de compatibilité", 0.3, 0.8, 0.4, 0.1)

# Gestion des offres locales
st.sidebar.markdown("---")
st.sidebar.subheader("Gestion des offres locales")

# Vérifier si des offres locales existent
jobs_dir = os.path.join("data", "indeed_jobs")
if os.path.exists(jobs_dir):
    json_files = [f for f in os.listdir(jobs_dir) if f.endswith('.json')]
    
    # Afficher le nombre d'offres locales
    individual_files = [f for f in json_files if not f.startswith('batch_') and not f.startswith('emergency_backup_')]
    batch_files = [f for f in json_files if f.startswith('batch_')]
    st.sidebar.info(f"{len(individual_files)} offres individuelles et {len(batch_files)} lots dans le stockage local")
    
    # Bouton pour réimporter les offres vers la base vectorielle
    if st.sidebar.button("Réimporter offres locales → Base vectorielle"):
        st.sidebar.info("Réimportation en cours...")
        
        try:
            # Appeler le script pour réimporter
            import subprocess
            import sys
            
            cmd = [sys.executable, "improve_matching.py", "--reimport-local"]
            if api_url != DEFAULT_API_URL:
                cmd.extend(["--api-url", api_url])
                
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                st.sidebar.success("Réimportation terminée")
                st.sidebar.text(result.stdout)
            else:
                st.sidebar.error(f"Erreur durant la réimportation: {result.returncode}")
                st.sidebar.text(result.stderr)
                
        except Exception as e:
            st.sidebar.error(f"Erreur lors de la réimportation: {str(e)}")
else:
    st.sidebar.warning("Aucune offre locale disponible")

# Ajouter un onglet pour les statistiques de feedback
st.sidebar.markdown("---")
if st.sidebar.checkbox("Afficher les statistiques de feedback"):
    feedback_dir = "data/feedback"
    feedback_history_file = os.path.join(feedback_dir, "history.json")
    
    if not os.path.exists(feedback_history_file):
        st.sidebar.warning("Aucun historique de feedback disponible.")
    else:
        try:
            with open(feedback_history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            sessions = history.get("sessions", [])
            if sessions:
                st.sidebar.success(f"{len(sessions)} sessions de feedback enregistrées.")
                
                # Afficher un mini tableau des statistiques
                domain_counts = {}
                for session in sessions:
                    domain = session.get("cv_domain", "unknown")
                    if domain not in domain_counts:
                        domain_counts[domain] = 0
                    domain_counts[domain] += 1
                
                st.sidebar.markdown("**Feedbacks par domaine:**")
                for domain, count in domain_counts.items():
                    st.sidebar.markdown(f"- {domain}: {count}")
            else:
                st.sidebar.info("Aucun feedback enregistré pour le moment.")
        except Exception as e:
            st.sidebar.error(f"Erreur lors de la lecture des statistiques: {str(e)}")

# Titre principal
st.title("Analyse de CV et Matching d'Offres")
st.markdown("Cet outil analyse votre CV pour identifier vos compétences et votre domaine, puis recherche des offres d'emploi correspondantes.")

# Information sur le status du service d'embeddings
embedding_service_status = st.empty()

# Options d'entrée: fichier ou texte
upload_option = st.radio("Comment souhaitez-vous fournir votre CV?", ["Uploader un fichier", "Saisir le texte directement"])

cv_path = None
cv_text = None

if upload_option == "Uploader un fichier":
    uploaded_file = st.file_uploader("Choisissez un fichier CV (PDF ou TXT)", type=["pdf", "txt"])
    if uploaded_file:
        # Créer un fichier temporaire pour le CV
        temp_dir = tempfile.mkdtemp()
        cv_path = os.path.join(temp_dir, uploaded_file.name)
        with open(cv_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Fichier téléchargé: {uploaded_file.name}")
else:
    cv_text = st.text_area("Entrez le texte de votre CV", height=300)

# Bouton pour lancer l'analyse
if st.button("Lancer le matching"):
    if not cv_path and not cv_text:
        st.error("Veuillez fournir un CV (fichier ou texte)")
    else:
        # Barre de progression
        progress_bar = st.progress(0)
        
        # Analyse en cours
        with st.spinner("Analyse du CV et recherche d'offres en cours..."):
            progress_bar.progress(25)
            
            # Lancer le matching
            try:
                results = run_matching(
                    cv_path=cv_path,
                    cv_text=cv_text,
                    api_url=api_url,
                    llm_url=llm_url,
                    top_k=top_k,
                    min_score=min_score,
                    strict_threshold=strict_threshold
                )
                
                progress_bar.progress(75)
                
                # Afficher les résultats
                if results:
                    st.subheader("Résultats du matching")
                    
                    # Afficher l'analyse du CV
                    display_cv_analysis(results)
                    
                    # Vérifier si les résultats contiennent des offres de la base vectorielle
                    if results.get("results") and any(job.get("source", "") != "Scraper" for job in results["results"]):
                        # Afficher les résultats de matching de la base vectorielle
                        st.markdown("### Offres d'emploi correspondantes (base de données)")
                        # Filtrer pour n'afficher que les offres de la base vectorielle
                        vector_results = {
                            "results": [job for job in results["results"] if job.get("source", "") != "Scraper"]
                        }
                        if vector_results["results"]:
                            display_results(vector_results, show_details)
                        else:
                            st.info("Aucune offre trouvée dans la base de données vectorielle.")
                    
                    # Vérifier si les résultats contiennent des offres scrapées
                    if results.get("results") and any(job.get("source", "") == "Scraper" for job in results["results"]):
                        # Afficher les offres scrapées
                        st.markdown("### Offres d'emploi correspondantes (récemment scrapées)")
                        # Filtrer pour n'afficher que les offres scrapées
                        scraped_results = {
                            "results": [job for job in results["results"] if job.get("source", "") == "Scraper"]
                        }
                        display_results(scraped_results, show_details)
                        
                        # Ajouter un bouton pour stocker les offres scrapées dans la base vectorielle
                        scraped_storage_col1, scraped_storage_col2 = st.columns([1, 2])
                        with scraped_storage_col1:
                            store_scraped = st.button("Stocker dans la base vectorielle")
                        with scraped_storage_col2:
                            domain_override = st.selectbox(
                                "Forcer le domaine (optionnel)",
                                ["", "informatique_reseaux", "automatismes_info_industrielle", "finance", "genie_civil_btp", "genie_industriel"],
                                index=0
                            )
                        
                        # Si l'utilisateur clique sur le bouton pour stocker les offres
                        if store_scraped:
                            with st.spinner("Stockage des offres scrapées dans la base vectorielle..."):
                                try:
                                    # Importer la fonction
                                    import asyncio
                                    from improve_matching import store_scraped_jobs_in_vector_db
                                    
                                    # Appeler la fonction de stockage de manière asynchrone
                                    domain = domain_override if domain_override else None
                                    total_jobs, stored_jobs = asyncio.run(store_scraped_jobs_in_vector_db(api_url=api_url, domain_override=domain))
                                    
                                    if stored_jobs > 0:
                                        st.success(f"✅ {stored_jobs} offres sur {total_jobs} ont été stockées avec succès dans la base vectorielle!")
                                    else:
                                        st.warning(f"⚠️ Aucune offre n'a pu être stockée. {total_jobs} offres trouvées.")
                                except Exception as e:
                                    st.error(f"❌ Erreur lors du stockage des offres: {str(e)}")
                    
                    # Rechercher des offres sur Indeed si l'option est activée
                    if indeed_search and results.get("cv_analysis"):
                        st.markdown("### Offres d'emploi récentes sur Indeed")
                        
                        # Expliquer le système de matching
                        with st.expander("📌 Comment fonctionne la recherche d'offres compatibles?"):
                            st.markdown("""
                            Le système utilise les informations extraites de votre CV pour rechercher des offres d'emploi récentes sur Indeed qui correspondent à votre profil:
                            
                            1. **Génération de requête intelligente**: Une requête de recherche est créée en combinant votre domaine professionnel et vos compétences clés
                            2. **Analyse du contenu des offres**: Chaque offre est analysée pour identifier les correspondances avec vos compétences
                            3. **Filtrage par score minimal**: Seules les offres dépassant le seuil de compatibilité défini sont affichées
                            4. **Classement par pertinence**: Les offres sont triées du meilleur match au moins bon
                            
                            **Signification des scores de compatibilité**:
                            - 🟢 **0.7-1.0**: Excellente correspondance - L'offre correspond très bien à votre profil
                            - 🟠 **0.5-0.7**: Bonne correspondance - L'offre correspond assez bien à votre profil
                            - 🔴 **< 0.5**: Correspondance faible - L'offre correspond peu à votre profil mais contient certains éléments pertinents
                            
                            Vous pouvez ajuster le score minimal de compatibilité dans les paramètres pour affiner les résultats.
                            """)
                        
                        progress_bar.progress(85)
                        # Option pour sauvegarder ou non dans la base vectorielle
                        save_to_vectordb = st.checkbox("Sauvegarder les offres dans la base vectorielle", value=True)
                        local_backup = st.checkbox("Activer la sauvegarde locale des offres", value=True)
                        
                        # Récupérer les offres Indeed
                        indeed_jobs = fetch_indeed_jobs(
                            results["cv_analysis"],
                            location=indeed_location,
                            max_results=indeed_max_results,
                            country_code=indeed_country_code,
                            min_match_score=indeed_min_score
                        )
                        
                        # Afficher les offres Indeed
                        display_indeed_jobs(indeed_jobs, show_details)
                        
                        # Ajouter l'option de télécharger les résultats Indeed
                        if indeed_jobs:
                            indeed_results = {
                                "query": generate_indeed_query({"cv_analysis": results["cv_analysis"]}),
                                "location": indeed_location,
                                "country": indeed_country,
                                "date": datetime.now().isoformat(),
                                "jobs": indeed_jobs
                            }
                            
                            indeed_json = json.dumps(indeed_results, ensure_ascii=False, indent=4)
                            st.download_button(
                                label="Télécharger les offres Indeed (JSON)",
                                data=indeed_json,
                                file_name=f"indeed_jobs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                            
                            # Si backup local activé et pas de stockage vectoriel, sauvegarder tout
                            if local_backup and not save_to_vectordb:
                                jobs_dir = os.path.join("data", "indeed_jobs")
                                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                batch_file = os.path.join(jobs_dir, f"batch_{timestamp}.json")
                                os.makedirs(jobs_dir, exist_ok=True)
                                
                                with open(batch_file, 'w', encoding='utf-8') as f:
                                    json.dump(indeed_results, f, ensure_ascii=False, indent=2)
                                
                                st.success(f"Toutes les offres sauvegardées localement: {batch_file}")
                    
                    progress_bar.progress(100)
                else:
                    st.error("Erreur lors de l'analyse du CV ou du matching.")
                    progress_bar.progress(100)
            
            except Exception as e:
                st.error(f"Une erreur s'est produite: {str(e)}")
                progress_bar.progress(100)

# Information sur les domaines supportés
with st.expander("Domaines professionnels supportés"):
    st.markdown("""
    L'outil supporte actuellement les domaines suivants:
    
    1. **informatique_reseaux** - Informatique, développement logiciel, réseaux, cybersécurité
    2. **automatismes_info_industrielle** - Automatisation industrielle, SCADA, PLC, robotique
    3. **finance** - Comptabilité, analyse financière, audit, contrôle de gestion
    4. **genie_civil_btp** - Génie civil, construction, architecture
    5. **genie_industriel** - Génie industriel, production, logistique, qualité
    """)

# En bas de la page, ajouter une section sur le service d'embeddings utilisé
st.sidebar.markdown("---")
st.sidebar.markdown("### Informations techniques")
if LOCAL_EMBEDDINGS_AVAILABLE:
    st.sidebar.success("Service d'embeddings local disponible ✅")
    try:
        from local_embeddings import LocalEmbeddingService
        service = LocalEmbeddingService(device="cpu")
        model_info = service.get_model_info()
        st.sidebar.info(f"Modèle: {model_info['name']} ({model_info['dimension']} dimensions)")
    except Exception as e:
        st.sidebar.warning(f"Erreur d'initialisation: {str(e)}")
else:
    st.sidebar.warning("Service d'embeddings local non disponible ❌")

# Pied de page
st.markdown("---")
st.markdown("© 2023 AI Recruiter LLM | Développé avec Streamlit")

# Exécuter l'application avec: streamlit run cv_matching_ui.py 