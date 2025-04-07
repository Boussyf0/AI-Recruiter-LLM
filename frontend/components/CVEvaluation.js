import { useState } from 'react';
import axios from 'axios';

export default function CVEvaluation({ domain }) {
  const [cv, setCV] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.post('http://localhost:8000/api/evaluate', {
        domain_id: domain.id,
        type: 'cv_evaluation',
        content: cv
      });
      
      setResult(response.data);
    } catch (err) {
      console.error('Erreur lors de l\'évaluation du CV:', err);
      setError('Une erreur est survenue lors de l\'évaluation. Veuillez réessayer.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="cv-evaluation">
      <h2>Évaluation de CV - {domain.name}</h2>
      
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="cv-text">Contenu du CV</label>
          <textarea 
            id="cv-text"
            className="cv-textarea" 
            value={cv} 
            onChange={(e) => setCV(e.target.value)}
            placeholder="Collez le contenu du CV ici..."
            rows={10}
            required
          />
        </div>
        
        <button 
          type="submit" 
          className="submit-button"
          disabled={loading || !cv.trim()}
        >
          {loading ? 'Évaluation en cours...' : 'Évaluer le CV'}
        </button>
      </form>
      
      {error && (
        <div className="error-message">
          {error}
        </div>
      )}
      
      {result && (
        <div className="result-container">
          <h3>Résultat de l'évaluation</h3>
          <div className="result-content">
            <pre>{JSON.stringify(result, null, 2)}</pre>
          </div>
        </div>
      )}
      
      <style jsx>{`
        .cv-evaluation {
          width: 100%;
          max-width: 800px;
          margin-top: 20px;
        }
        
        .form-group {
          margin-bottom: 20px;
        }
        
        label {
          display: block;
          margin-bottom: 8px;
          font-weight: bold;
        }
        
        .cv-textarea {
          width: 100%;
          padding: 10px;
          border: 1px solid #ddd;
          border-radius: 4px;
          font-family: inherit;
          font-size: 14px;
          resize: vertical;
        }
        
        .submit-button {
          background-color: #0070f3;
          color: white;
          border: none;
          border-radius: 4px;
          padding: 10px 20px;
          font-size: 16px;
          cursor: pointer;
          transition: background-color 0.2s;
        }
        
        .submit-button:hover {
          background-color: #0051b3;
        }
        
        .submit-button:disabled {
          background-color: #cccccc;
          cursor: not-allowed;
        }
        
        .error-message {
          margin-top: 20px;
          padding: 10px;
          background-color: #ffebee;
          color: #c62828;
          border-radius: 4px;
          border-left: 4px solid #c62828;
        }
        
        .result-container {
          margin-top: 20px;
          padding: 15px;
          background-color: #f5f5f5;
          border-radius: 4px;
          border: 1px solid #ddd;
        }
        
        .result-content {
          background-color: white;
          padding: 15px;
          border-radius: 4px;
          overflow-x: auto;
        }
        
        .result-content pre {
          margin: 0;
          white-space: pre-wrap;
          word-wrap: break-word;
        }
      `}</style>
    </div>
  );
} 