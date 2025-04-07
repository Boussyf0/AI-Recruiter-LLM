import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Backend configuration
BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))

# LLM service configuration
LLM_URL = os.getenv("LLM_URL", "http://llm:8001")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt2")

# Vector DB configuration
VECTOR_DB_URL = os.getenv("VECTOR_DB_URL", "http://vector_db:6333")
QDRANT_HOST = os.getenv("QDRANT_HOST", "vector_db")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# Collections
RESUME_COLLECTION = "resumes"
JOB_DESCRIPTION_COLLECTION = "job_descriptions"

# API configuration
API_PREFIX = "/api/v1"

