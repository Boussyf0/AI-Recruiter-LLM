import { useEffect } from 'react';
import axios from 'axios';

function MyApp({ Component, pageProps }) {
  useEffect(() => {
    // Configuration globale d'Axios
    axios.defaults.baseURL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    
    // Configuration des en-têtes par défaut
    axios.defaults.headers.common['Content-Type'] = 'application/json';
    axios.defaults.headers.common['Accept'] = 'application/json';
    
    // Intercepteur pour les réponses
    axios.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('Erreur API:', error);
        return Promise.reject(error);
      }
    );
  }, []);

  return <Component {...pageProps} />;
}

export default MyApp; 