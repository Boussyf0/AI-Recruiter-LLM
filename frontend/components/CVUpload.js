import { useState, useRef } from 'react';
import axios from 'axios';

export default function CVUpload({ onAnalysisComplete }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [preview, setPreview] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;
    
    // Vérifier le type de fichier
    if (!['application/pdf', 'image/png'].includes(selectedFile.type)) {
      setError('Format de fichier non supporté. Veuillez télécharger un fichier PDF ou PNG.');
      setFile(null);
      setPreview(null);
      return;
    }
    
    setFile(selectedFile);
    setError(null);
    
    // Générer une prévisualisation pour les images PNG
    if (selectedFile.type === 'image/png') {
      const reader = new FileReader();
      reader.onload = (e) => {
        setPreview(e.target.result);
      };
      reader.readAsDataURL(selectedFile);
    } else {
      setPreview(null);
    }
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) return;
    
    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const response = await axios.post('http://localhost:8000/api/v1/analyze-cv', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      if (onAnalysisComplete) {
        onAnalysisComplete(response.data);
      }
    } catch (err) {
      console.error('Erreur lors de l\'analyse du CV:', err);
      setError('Une erreur est survenue lors de l\'analyse. Veuillez réessayer.');
    } finally {
      setLoading(false);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (['application/pdf', 'image/png'].includes(droppedFile.type)) {
        setFile(droppedFile);
        setError(null);
        
        if (droppedFile.type === 'image/png') {
          const reader = new FileReader();
          reader.onload = (e) => {
            setPreview(e.target.result);
          };
          reader.readAsDataURL(droppedFile);
        } else {
          setPreview(null);
        }
      } else {
        setError('Format de fichier non supporté. Veuillez télécharger un fichier PDF ou PNG.');
      }
    }
  };

  return (
    <div className="cv-upload">
      <h2>Téléchargez votre CV</h2>
      <p>Formats acceptés: PDF, PNG</p>
      
      <form onSubmit={handleUpload}>
        <div 
          className="drop-area" 
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current.click()}
        >
          {preview ? (
            <img src={preview} alt="Aperçu du CV" className="preview-image" />
          ) : (
            <div className="drop-placeholder">
              <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" fill="currentColor" viewBox="0 0 16 16">
                <path d="M.5 9.9a.5.5 0 0 1 .5.5v2.5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-2.5a.5.5 0 0 1 1 0v2.5a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2v-2.5a.5.5 0 0 1 .5-.5z"/>
                <path d="M7.646 1.146a.5.5 0 0 1 .708 0l3 3a.5.5 0 0 1-.708.708L8.5 2.707V11.5a.5.5 0 0 1-1 0V2.707L5.354 4.854a.5.5 0 1 1-.708-.708l3-3z"/>
              </svg>
              <p>Cliquez ou glissez votre fichier ici</p>
              {file && <p className="file-name">{file.name}</p>}
            </div>
          )}
          <input 
            type="file" 
            accept=".pdf,.png" 
            onChange={handleFileChange} 
            ref={fileInputRef}
            style={{ display: 'none' }}
          />
        </div>
        
        <button 
          type="submit" 
          className="submit-button"
          disabled={loading || !file}
        >
          {loading ? 'Analyse en cours...' : 'Analyser le CV'}
        </button>
      </form>
      
      {error && (
        <div className="error-message">
          {error}
        </div>
      )}
      
      <style jsx>{`
        .cv-upload {
          width: 100%;
          max-width: 800px;
          margin-top: 20px;
        }
        
        .drop-area {
          border: 2px dashed #0070f3;
          border-radius: 8px;
          padding: 30px;
          text-align: center;
          cursor: pointer;
          margin-bottom: 20px;
          min-height: 200px;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: background-color 0.2s;
        }
        
        .drop-area:hover {
          background-color: #f0f7ff;
        }
        
        .drop-placeholder {
          display: flex;
          flex-direction: column;
          align-items: center;
          color: #666;
        }
        
        .preview-image {
          max-width: 100%;
          max-height: 300px;
          border-radius: 4px;
        }
        
        .file-name {
          margin-top: 10px;
          font-weight: bold;
          color: #0070f3;
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
          width: 100%;
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
      `}</style>
    </div>
  );
} 