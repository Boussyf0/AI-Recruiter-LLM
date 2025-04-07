import React, { useState, useEffect } from 'react';

export default function Home() {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputText.trim()) return;

    const userMessage = { role: 'user', content: inputText };
    setMessages((prev) => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);

    try {
      // Simulate API call to backend
      setTimeout(() => {
        const botMessage = { 
          role: 'assistant', 
          content: "Je suis l'assistant AI Recruiter. Notre backend n'est pas encore connecté, mais je serai bientôt en mesure de vous aider avec votre recherche d'emploi ou vos besoins de recrutement."
        };
        setMessages((prev) => [...prev, botMessage]);
        setIsLoading(false);
      }, 1000);
      
      // When backend is ready, replace with actual API call:
      /*
      const response = await fetch('/api/v1/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          messages: [...messages, userMessage]
        })
      });
      const data = await response.json();
      setMessages((prev) => [...prev, { role: 'assistant', content: data.response }]);
      */
    } catch (error) {
      console.error('Error:', error);
      setMessages((prev) => [...prev, { 
        role: 'assistant', 
        content: "Désolé, une erreur s'est produite. Veuillez réessayer."
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={{ 
      maxWidth: '800px', 
      margin: '0 auto', 
      padding: '20px',
      fontFamily: 'Arial, sans-serif'
    }}>
      <h1 style={{ textAlign: 'center', color: '#333' }}>
        AI Recruiter
      </h1>
      
      <div style={{ 
        border: '1px solid #eaeaea', 
        borderRadius: '10px', 
        height: '400px', 
        padding: '20px',
        overflowY: 'auto',
        marginBottom: '20px',
        backgroundColor: '#f9f9f9'
      }}>
        {messages.length === 0 ? (
          <div style={{ textAlign: 'center', color: '#666', marginTop: '160px' }}>
            Commencez une conversation avec AI Recruiter
          </div>
        ) : (
          messages.map((msg, index) => (
            <div 
              key={index} 
              style={{ 
                marginBottom: '10px',
                textAlign: msg.role === 'user' ? 'right' : 'left',
              }}
            >
              <div style={{ 
                display: 'inline-block',
                backgroundColor: msg.role === 'user' ? '#1E88E5' : '#E0E0E0',
                color: msg.role === 'user' ? 'white' : 'black',
                padding: '10px 15px',
                borderRadius: '18px',
                maxWidth: '70%',
                wordWrap: 'break-word'
              }}>
                {msg.content}
              </div>
            </div>
          ))
        )}
        {isLoading && (
          <div style={{ textAlign: 'left', marginTop: '10px' }}>
            <div style={{ 
              display: 'inline-block',
              backgroundColor: '#E0E0E0',
              padding: '10px 15px',
              borderRadius: '18px'
            }}>
              Réflexion...
            </div>
          </div>
        )}
      </div>
      
      <form onSubmit={handleSubmit} style={{ display: 'flex' }}>
        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Écrivez votre message ici..."
          style={{ 
            flex: 1, 
            padding: '12px 15px',
            borderRadius: '25px',
            border: '1px solid #ddd',
            fontSize: '16px',
            outline: 'none'
          }}
        />
        <button 
          type="submit" 
          disabled={!inputText.trim() || isLoading}
          style={{ 
            marginLeft: '10px',
            backgroundColor: '#1E88E5',
            color: 'white',
            border: 'none',
            borderRadius: '25px',
            padding: '0 20px',
            fontSize: '16px',
            cursor: 'pointer',
            opacity: !inputText.trim() || isLoading ? 0.7 : 1
          }}
        >
          Envoyer
        </button>
      </form>
    </div>
  );
} 