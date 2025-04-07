import { useState } from 'react';
import Head from 'next/head';
import CVEvaluation from '../components/CVEvaluation';
import CVUpload from '../components/CVUpload';
import AnalysisResults from '../components/AnalysisResults';

export default function Home() {
  const [selectedDomain, setSelectedDomain] = useState(null);
  const [currentTab, setCurrentTab] = useState('upload'); // 'upload' ou 'manual'
  const [analysisResults, setAnalysisResults] = useState(null);
  
  const domains = [
    { id: 'informatique_reseaux', name: 'Informatique et Réseaux' },
    { id: 'automatismes_info_industrielle', name: 'Automatismes et Informatique Industrielle' },
    { id: 'finance', name: 'Finance' },
    { id: 'genie_civil_btp', name: 'Génie Civil et BTP' },
    { id: 'genie_industriel', name: 'Génie Industriel' }
  ];

  const selectedDomainObj = selectedDomain ? domains.find(d => d.id === selectedDomain) : null;

  const handleAnalysisComplete = (results) => {
    setAnalysisResults(results);
  };

  return (
    <div className="container">
      <Head>
        <title>AI Recruiter - Interface de recrutement</title>
        <meta name="description" content="Interface pour interagir avec l'IA de recrutement spécialisée" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className="main">
        <h1 className="title">
          AI Recruiter - Interface de recrutement
        </h1>

        <p className="description">
          Interface pour interagir avec l'IA de recrutement spécialisée dans 5 domaines
        </p>

        <div className="tabs">
          <button 
            className={`tab-button ${currentTab === 'upload' ? 'tab-active' : ''}`}
            onClick={() => setCurrentTab('upload')}
          >
            Analyse automatique de CV
          </button>
          <button 
            className={`tab-button ${currentTab === 'manual' ? 'tab-active' : ''}`}
            onClick={() => setCurrentTab('manual')}
          >
            Sélection manuelle de domaine
          </button>
        </div>

        {currentTab === 'upload' ? (
          <div className="upload-section">
            <CVUpload onAnalysisComplete={handleAnalysisComplete} />
            {analysisResults && <AnalysisResults results={analysisResults} />}
          </div>
        ) : (
          <>
            <div className="grid">
              {domains.map((domain) => (
                <div 
                  key={domain.id} 
                  className={`card ${selectedDomain === domain.id ? 'card-selected' : ''}`}
                  onClick={() => setSelectedDomain(domain.id)}
                >
                  <h2>{domain.name}</h2>
                  <p>Spécialistes en {domain.name.toLowerCase()}</p>
                </div>
              ))}
            </div>

            {selectedDomainObj && (
              <div className="selected-domain">
                <div className="domain-header">
                  <h3>Domaine sélectionné: {selectedDomainObj.name}</h3>
                  <button 
                    className="back-button"
                    onClick={() => setSelectedDomain(null)}
                  >
                    Retour à la liste
                  </button>
                </div>
                <CVEvaluation domain={selectedDomainObj} />
              </div>
            )}
          </>
        )}
      </main>

      <footer className="footer">
        <span>AI Recruiter - Projet en cours de développement</span>
      </footer>

      <style jsx>{`
        .container {
          min-height: 100vh;
          padding: 0 0.5rem;
          display: flex;
          flex-direction: column;
          justify-content: center;
          align-items: center;
          background-color: #f5f5f5;
        }
        
        .main {
          padding: 5rem 0;
          flex: 1;
          display: flex;
          flex-direction: column;
          justify-content: center;
          align-items: center;
          max-width: 900px;
        }
        
        .footer {
          width: 100%;
          height: 50px;
          border-top: 1px solid #eaeaea;
          display: flex;
          justify-content: center;
          align-items: center;
          background-color: #fff;
        }
        
        .title {
          margin: 0;
          line-height: 1.15;
          font-size: 3rem;
          text-align: center;
          color: #0070f3;
        }
        
        .description {
          text-align: center;
          line-height: 1.5;
          font-size: 1.5rem;
          margin: 20px 0 40px;
        }
        
        .tabs {
          display: flex;
          margin-bottom: 30px;
          border-bottom: 1px solid #ddd;
          width: 100%;
        }
        
        .tab-button {
          padding: 12px 24px;
          background: none;
          border: none;
          font-size: 16px;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.2s;
          border-bottom: 3px solid transparent;
          color: #666;
        }
        
        .tab-button:hover {
          color: #0070f3;
        }
        
        .tab-active {
          color: #0070f3;
          border-bottom-color: #0070f3;
        }
        
        .upload-section {
          width: 100%;
          max-width: 800px;
        }
        
        .grid {
          display: flex;
          flex-wrap: wrap;
          justify-content: center;
          max-width: 800px;
          margin-top: 2rem;
        }
        
        .card {
          margin: 1rem;
          flex-basis: 45%;
          padding: 1.5rem;
          text-align: left;
          color: inherit;
          text-decoration: none;
          border: 1px solid #eaeaea;
          border-radius: 10px;
          transition: color 0.15s ease, border-color 0.15s ease, transform 0.2s ease;
          background-color: white;
          cursor: pointer;
        }
        
        .card:hover,
        .card:focus,
        .card:active {
          color: #0070f3;
          border-color: #0070f3;
          transform: translateY(-5px);
        }
        
        .card-selected {
          color: #0070f3;
          border-color: #0070f3;
          background-color: #f0f7ff;
        }
        
        .card h2 {
          margin: 0 0 1rem 0;
          font-size: 1.5rem;
        }
        
        .card p {
          margin: 0;
          font-size: 1.25rem;
          line-height: 1.5;
        }
        
        .selected-domain {
          margin-top: 40px;
          padding: 20px;
          border: 1px solid #0070f3;
          border-radius: 10px;
          background-color: white;
          width: 100%;
          max-width: 800px;
        }
        
        .domain-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 20px;
        }
        
        .back-button {
          background-color: #f0f7ff;
          color: #0070f3;
          border: 1px solid #0070f3;
          border-radius: 4px;
          padding: 8px 16px;
          font-size: 14px;
          cursor: pointer;
          transition: background-color 0.2s;
        }
        
        .back-button:hover {
          background-color: #e6f0ff;
        }
      `}</style>

      <style jsx global>{`
        html,
        body {
          padding: 0;
          margin: 0;
          font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto,
            Oxygen, Ubuntu, Cantarell, Fira Sans, Droid Sans, Helvetica Neue,
            sans-serif;
        }

        * {
          box-sizing: border-box;
        }
      `}</style>
    </div>
  );
} 