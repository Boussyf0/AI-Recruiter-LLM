from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
import httpx
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import os
import shutil
import tempfile
import uuid
import json
from pathlib import Path
from PIL import Image
import pytesseract
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

# Import config or use environment variables if config not found
try:
    from config import API_PREFIX, LLM_URL, VECTOR_DB_URL
except ImportError:
    print("Config module not found, using environment variables")
    API_PREFIX = os.getenv("API_PREFIX", "/api/v1")
    LLM_URL = os.getenv("LLM_URL", "http://llm:8001")
    VECTOR_DB_URL = os.getenv("VECTOR_DB_URL", "http://vector_db:6333")

app = FastAPI(title="AI Recruiter API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a temporary directory for uploads
UPLOAD_DIR = Path(tempfile.gettempdir()) / "ai_recruiter_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Import the vector database handler
from app.vector_db import vector_db

# Models
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    job_description_id: Optional[str] = None
    resume_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    
class DocumentUpload(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None

class EvaluationRequest(BaseModel):
    domain_id: str
    type: str
    content: str

class AnalysisResponse(BaseModel):
    domain_analysis: Dict[str, float]
    competences: Dict[str, List[Dict[str, Union[str, float]]]]
    summary: str

@app.get("/")
def read_root():
    return {"status": "ok", "message": "AI Recruiter API is running"}

@app.post(f"{API_PREFIX}/chat")
async def chat(request: ChatRequest):
    """Process a chat request and return the LLM response"""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(f"{LLM_URL}/generate", json=request.dict())
            response.raise_for_status()
            return response.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"LLM service error: {str(exc)}")

@app.post(f"{API_PREFIX}/upload/resume")
async def upload_resume(document: DocumentUpload):
    """Upload and index a resume"""
    try:
        # Ajouter le CV à la base de vecteurs
        cv_id = await vector_db.add_cv(document.content, document.metadata or {})
        return {"status": "success", "message": "Resume uploaded and indexed", "id": cv_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error indexing resume: {str(e)}")

@app.post(f"{API_PREFIX}/upload/job")
async def upload_job(document: DocumentUpload):
    """Upload and index a job description"""
    try:
        # Ajouter l'offre d'emploi à la base de vecteurs
        job_id = await vector_db.add_job(document.content, document.metadata or {})
        return {"status": "success", "message": "Job description uploaded and indexed", "id": job_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error indexing job description: {str(e)}")

@app.get(f"{API_PREFIX}/search/resume")
async def search_resume(query: str, limit: int = Query(5, ge=1, le=20)):
    """Rechercher des CV similaires à la requête"""
    try:
        results = await vector_db.search_similar_cvs(query, limit)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching resumes: {str(e)}")

@app.get(f"{API_PREFIX}/search/jobs")
async def search_jobs_for_resume(
    resume_id: str = None, 
    resume_text: str = None, 
    domain: str = None,
    limit: int = Query(5, ge=1, le=20)
):
    """Rechercher des offres d'emploi adaptées à un CV"""
    try:
        if resume_id:
            # Récupérer le CV depuis la base de vecteurs
            cv_data = vector_db.get_cv_by_id(resume_id)
            if not cv_data:
                raise HTTPException(status_code=404, detail=f"Resume with ID {resume_id} not found")
            cv_text = cv_data["text"]
            # Récupérer le domaine du CV si disponible
            if not domain and "metadata" in cv_data and "domain" in cv_data["metadata"]:
                domain = cv_data["metadata"]["domain"]
        elif resume_text:
            cv_text = resume_text
        else:
            raise HTTPException(status_code=400, detail="Either resume_id or resume_text must be provided")
        
        # Rechercher des offres similaires
        results = await vector_db.search_similar_jobs(cv_text, limit, domain)
        return {"results": results}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching jobs: {str(e)}")

@app.post(f"{API_PREFIX}/evaluate")
async def evaluate_document(request: EvaluationRequest):
    """Évalue un document pour un domaine spécifique"""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{LLM_URL}/evaluate", 
                json={
                    "domain_id": request.domain_id,
                    "type": request.type,
                    "content": request.content
                }
            )
            response.raise_for_status()
            return response.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"LLM service error: {str(exc)}")

def extract_text_from_pdf(file_path):
    """Extraire le texte d'un fichier PDF"""
    if PyPDF2 is None:
        raise ImportError("PyPDF2 is not installed. Please install it to process PDF files.")
    
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"
    return text

def extract_text_from_image(file_path):
    """Extraire le texte d'une image en utilisant OCR"""
    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image, lang='fra+eng')
        return text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'extraction du texte: {str(e)}")

@app.post(f"{API_PREFIX}/analyze-cv")
async def analyze_cv(file: UploadFile = File(...)):
    """Analyser un CV téléchargé (PDF ou PNG) et classifier dans les 5 domaines"""
    # Vérifier le type de fichier
    if file.content_type not in ["application/pdf", "image/png"]:
        raise HTTPException(status_code=400, detail="Seuls les fichiers PDF et PNG sont acceptés")
    
    # Enregistrer le fichier
    file_id = str(uuid.uuid4())
    file_extension = "pdf" if file.content_type == "application/pdf" else "png"
    file_path = UPLOAD_DIR / f"{file_id}.{file_extension}"
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()
    
    # Extraire le texte selon le type de fichier
    try:
        if file_extension == "pdf":
            text = extract_text_from_pdf(file_path)
        else:
            text = extract_text_from_image(file_path)
        
        # Si le texte est vide ou trop court
        if len(text) < 50:
            raise HTTPException(status_code=400, detail="Impossible d'extraire suffisamment de texte du document")
        
        # Analyser le CV dans tous les domaines
        analysis_results = {
            "domain_analysis": {},
            "competences": {},
            "summary": ""
        }
        
        domains = [
            "informatique_reseaux",
            "automatismes_info_industrielle",
            "finance",
            "genie_civil_btp",
            "genie_industriel"
        ]
        
        # Appeler le LLM pour chaque domaine
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Créer une analyse synthétique du CV
            summary_response = await client.post(
                f"{LLM_URL}/summarize", 
                json={"content": text}
            )
            if summary_response.status_code == 200:
                analysis_results["summary"] = summary_response.json().get("summary", "")
            
            # Analyser pour chaque domaine
            for domain in domains:
                response = await client.post(
                    f"{LLM_URL}/analyze", 
                    json={
                        "domain_id": domain,
                        "content": text
                    }
                )
                if response.status_code == 200:
                    result = response.json()
                    analysis_results["domain_analysis"][domain] = result.get("match_score", 0.0)
                    analysis_results["competences"][domain] = result.get("competences", [])
        
        # Indexer le CV dans la base de vecteurs
        cv_metadata = {
            "id": file_id,
            "filename": file.filename,
            "analysis": analysis_results["domain_analysis"],
            "summary": analysis_results["summary"]
        }
        await vector_db.add_cv(text, cv_metadata)
        
        # Supprimer le fichier temporaire
        if file_path.exists():
            file_path.unlink()
        
        return analysis_results
        
    except Exception as e:
        # Supprimer le fichier en cas d'erreur
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'analyse du CV: {str(e)}")

@app.get(f"{API_PREFIX}/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    # Try to import config values or use defaults
    try:
        from config import BACKEND_HOST, BACKEND_PORT
    except ImportError:
        BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
        BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
    
    uvicorn.run("main:app", host=BACKEND_HOST, port=BACKEND_PORT, reload=True)

