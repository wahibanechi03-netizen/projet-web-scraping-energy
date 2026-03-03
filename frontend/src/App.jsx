import React, { useState, useEffect } from 'react';
import HealthCheck from './components/HealthCheck';
import ModelInfo from './components/ModelInfo';
import PredictForm from './components/PredictForm';
import './App.css';

function App() {
  const [health, setHealth] = useState(null);
  const [checking, setChecking] = useState(true);

  useEffect(() => {
    const checkBackend = async () => {
      try {
        const response = await fetch('http://localhost:8000/health');
        if (response.ok) {
          const data = await response.json();
          setHealth(data);
        } else {
          setHealth(null);
        }
      } catch (error) {
        setHealth(null);
      } finally {
        setChecking(false);
      }
    };
    
    checkBackend();
    const interval = setInterval(checkBackend, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div>
      <div className="container">
        {/* Header */}
        <div className="header">
          <h1>🌊 Prédiction Énergie des Vagues</h1>
          <p>Interface pour le modèle GradientBoosting (R² = 0.85)</p>
        </div>


        {/* Grille principale */}
        <div className="grid">
          {/* Colonne gauche */}
          <div>
            <HealthCheck />
            <ModelInfo />
          </div>

          {/* Colonne droite */}
          <div>
            <PredictForm />
          </div>
        </div>

        {/* Footer */}
        <div className="footer">
          <p>API FastAPI • Modèle GradientBoosting • R² = 0.85</p>
          <p style={{ marginTop: '10px', fontSize: '0.9em' }}>
            Backend: http://localhost:8000 | Frontend: http://localhost:5173
          </p>
          <p style={{ fontSize: '0.8em', marginTop: '10px', color: 'rgba(255,255,255,0.6)' }}>
            Pour tester: Lancez d'abord le backend (python main.py) puis actualisez cette page
          </p>
        </div>
      </div>
    </div>
  );
}

export default App;