# AI Recruiter LLM

A suite of Python tools for improving CV-to-job matching using AI and vector embeddings.

## Overview

This project provides intelligent tools to scrape job postings, analyze CVs, and match candidates to the most relevant job opportunities using natural language processing and domain-specific matching.

## Components

The project consists of several Python scripts:

1. `improve_matching.py` - Core matching algorithm between CVs and job offers
2. `add_test_jobs.py` - Utility to add test job offers to the vector database
3. `job_scraper_morocco.py` - Scraper specifically for Morocco job markets
4. `job_scraper.py` - General job scraping utility
5. `match_cv_jobs.py` - Direct matching between CVs and jobs
6. `airflow_job_scraper.py` - Airflow DAG for scheduling job scraping

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt` (tabulate, httpx, beautifulsoup4, pandas, etc.)
- Local API server running on http://localhost:8000/api/v1 (for vector storage)
- LLM service running on http://localhost:8001 (for text analysis)

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ai-recruiter-llm.git
cd ai-recruiter-llm

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Matching CVs with Job Offers

```bash
python improve_matching.py --cv path/to/your_cv.pdf --verbose
```

Or using CV text directly:

```bash
python improve_matching.py --text "Your CV text here" --verbose
```

### Adding Test Jobs to the Database

```bash
python add_test_jobs.py
```

### Scraping Job Offers

```bash
python job_scraper.py --query "data engineer" --domain informatique_reseaux --add-to-db
```

For Morocco-specific jobs:

```bash
python job_scraper_morocco.py
```

### Running Tests

```bash
python test_ai_recruiter.py
```

Or test a specific component:

```bash
python test_improve_matching.py
```

## Domain Classification

The system supports 5 professional domains:

1. `informatique_reseaux` - IT, software development, networks, cybersecurity
2. `automatismes_info_industrielle` - Industrial automation, SCADA, PLC, robotics
3. `finance` - Accounting, financial analysis, auditing, management control
4. `genie_civil_btp` - Civil engineering, construction, architecture
5. `genie_industriel` - Industrial engineering, production, logistics, quality

## Nouvelles fonctionnalités

### Scraping d'offres d'emploi Indeed

Le système intègre désormais une fonctionnalité de recherche d'offres d'emploi récentes sur Indeed correspondant au CV analysé. Cette fonctionnalité :

- Génère automatiquement une requête de recherche pertinente basée sur le domaine et les compétences extraites du CV
- Permet de spécifier la localisation et le nombre d'offres à récupérer
- Calcule un score de correspondance entre chaque offre et le CV
- S'intègre parfaitement à l'interface utilisateur Streamlit

#### Utilisation

Pour utiliser cette fonctionnalité, vous pouvez :

1. Utiliser le script `improve_matching.py` en ligne de commande :
   ```
   python improve_matching.py --cv mon_cv.pdf --indeed --indeed-location "Paris" --indeed-max 5
   ```

2. Utiliser l'interface Streamlit et activer l'option "Rechercher aussi sur Indeed" dans les paramètres.

#### Remarques

La version actuelle utilise une simulation pour éviter les problèmes de blocage par Indeed. Dans un environnement de production, il faudrait implémenter :

- Des techniques d'anti-détection plus avancées (rotation de proxies, gestion des fingerprints de navigateur)
- Un système de cache pour éviter de répéter les requêtes identiques
- Un respect des conditions d'utilisation du site cible

## Architecture

- Backend API server: Handles vector storage and retrieval
- LLM service: Provides text analysis and domain classification
- Python scripts: Interface between users, data sources, and backend services

## Directory Structure

```
ai-recruiter-llm/
├── improve_matching.py
├── add_test_jobs.py
├── job_scraper_morocco.py
├── job_scraper.py
├── match_cv_jobs.py
├── airflow_job_scraper.py
├── test_ai_recruiter.py
├── test_improve_matching.py
├── data/
│   ├── scraped_jobs/
│   └── test/
└── logs/
```



## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
