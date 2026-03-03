import React, { useState, useEffect } from 'react';
import axios from 'axios';

const HealthCheck = () => {
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const checkHealth = async () => {
    setLoading(true);
    try {
      const response = await axios.get('http://127.0.0.1:8000/health');
      setHealth(response.data);
      setError(null);
    } catch (err) {
      setError('API non disponible - Vérifiez que le backend tourne');
      setHealth(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="card">
      <h2>🔍 État de l'API</h2>
      
      {loading && <p>Vérification...</p>}
      
      {error && (
        <div className="alert alert-error">
          ⚠️ {error}
        </div>
      )}
      
      {health && (
        <div>
          <div className="stat-item">
            <span className="stat-label">Statut:</span>
            <span className="stat-value">
              <span className={`badge ${health.status === 'healthy' ? 'badge-success' : 'badge-warning'}`}>
                {health.status}
              </span>
            </span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Modèle:</span>
            <span className="stat-value">{health.model_type}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Features:</span>
            <span className="stat-value">{health.features_count}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">R²:</span>
            <span className="stat-value">
              <span className="badge badge-success">{health.r2_score}</span>
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

export default HealthCheck;