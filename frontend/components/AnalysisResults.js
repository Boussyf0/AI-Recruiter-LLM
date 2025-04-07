import { useState } from 'react';

export default function AnalysisResults({ results }) {
  const [selectedDomain, setSelectedDomain] = useState(null);

  if (!results || !results.domain_analysis) {
    return null;
  }

  const { domain_analysis, competences } = results;
  
  // Trouver le meilleur domaine correspondant (le plus haut score)
  const bestMatch = Object.entries(domain_analysis).sort((a, b) => b[1] - a[1])[0];
  
  const domainLabels = {
    'informatique_reseaux': 'Informatique et Réseaux',
    'automatismes_info_industrielle': 'Automatismes et Informatique Industrielle',
    'finance': 'Finance',
    'genie_civil_btp': 'Génie Civil et BTP',
    'genie_industriel': 'Génie Industriel'
  };

  const handleDomainClick = (domain) => {
    setSelectedDomain(domain === selectedDomain ? null : domain);
  };

  return (
    <div className="analysis-results">
      <h2>Résultats de l'analyse</h2>

      <div className="match-box">
        <h3>Meilleure correspondance</h3>
        <div className="best-match">
          <div className="match-label">{domainLabels[bestMatch[0]] || bestMatch[0]}</div>
          <div className="match-score">{Math.round(bestMatch[1] * 100)}%</div>
        </div>
      </div>

      <h3>Compatibilité par domaine</h3>
      <div className="domains-grid">
        {Object.entries(domain_analysis).map(([domain, score]) => (
          <div 
            key={domain} 
            className={`domain-card ${selectedDomain === domain ? 'domain-selected' : ''}`}
            onClick={() => handleDomainClick(domain)}
          >
            <div className="domain-name">{domainLabels[domain] || domain}</div>
            <div className="domain-score-container">
              <div 
                className="domain-score-bar" 
                style={{ width: `${Math.round(score * 100)}%` }}
              />
              <span className="domain-score-text">{Math.round(score * 100)}%</span>
            </div>
          </div>
        ))}
      </div>

      {selectedDomain && competences && competences[selectedDomain] && (
        <div className="competences-section">
          <h3>Compétences en {domainLabels[selectedDomain] || selectedDomain}</h3>
          <div className="competences-list">
            {competences[selectedDomain].map((comp, index) => (
              <div key={index} className="competence-item">
                <div className="competence-name">{comp.name}</div>
                <div className="competence-level-container">
                  <div 
                    className="competence-level-bar" 
                    style={{ width: `${comp.level * 100}%` }}
                  />
                  <span className="competence-level-text">
                    {Math.round(comp.level * 100)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <style jsx>{`
        .analysis-results {
          width: 100%;
          margin-top: 30px;
          padding: 20px;
          background-color: white;
          border-radius: 8px;
          box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }

        .match-box {
          background-color: #f0f7ff;
          border: 1px solid #0070f3;
          border-radius: 8px;
          padding: 15px;
          margin-bottom: 20px;
        }

        .match-box h3 {
          margin-top: 0;
          color: #0070f3;
        }

        .best-match {
          display: flex;
          justify-content: space-between;
          align-items: center;
          font-size: 1.2rem;
          font-weight: bold;
        }

        .match-score {
          background-color: #0070f3;
          color: white;
          padding: 5px 10px;
          border-radius: 20px;
        }

        .domains-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
          gap: 15px;
          margin-bottom: 20px;
        }

        .domain-card {
          padding: 15px;
          border: 1px solid #ddd;
          border-radius: 8px;
          cursor: pointer;
          transition: all 0.2s;
        }

        .domain-card:hover {
          border-color: #0070f3;
          transform: translateY(-2px);
        }

        .domain-selected {
          border-color: #0070f3;
          background-color: #f0f7ff;
        }

        .domain-name {
          font-weight: bold;
          margin-bottom: 8px;
        }

        .domain-score-container {
          height: 20px;
          background-color: #eee;
          border-radius: 10px;
          position: relative;
          overflow: hidden;
        }

        .domain-score-bar {
          height: 100%;
          background-color: #0070f3;
          border-radius: 10px;
        }

        .domain-score-text {
          position: absolute;
          top: 0;
          right: 10px;
          line-height: 20px;
          font-size: 12px;
          font-weight: bold;
          color: white;
          mix-blend-mode: difference;
        }

        .competences-section {
          margin-top: 20px;
          padding: 15px;
          border: 1px solid #0070f3;
          border-radius: 8px;
          background-color: #f0f7ff;
        }

        .competences-list {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
          gap: 10px;
        }

        .competence-item {
          background-color: white;
          padding: 10px;
          border-radius: 5px;
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }

        .competence-name {
          font-weight: 500;
          margin-bottom: 5px;
        }

        .competence-level-container {
          height: 10px;
          background-color: #eee;
          border-radius: 5px;
          position: relative;
          overflow: hidden;
        }

        .competence-level-bar {
          height: 100%;
          background-color: #0070f3;
          border-radius: 5px;
        }

        .competence-level-text {
          position: absolute;
          top: -2px;
          right: 5px;
          font-size: 10px;
          font-weight: bold;
          color: white;
          mix-blend-mode: difference;
        }
      `}</style>
    </div>
  );
} 