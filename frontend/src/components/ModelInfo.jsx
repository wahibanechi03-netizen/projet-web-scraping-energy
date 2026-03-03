import React, { useState, useEffect } from 'react';
import axios from 'axios';

const ModelInfo = () => {
  const [info, setInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchInfo = async () => {
      try {
        const response = await axios.get('http://127.0.0.1:8000/model/info');
        setInfo(response.data);
        setError(null);
      } catch (err) {
        setError('Impossible de charger les informations du modèle');
      } finally {
        setLoading(false);
      }
    };
    fetchInfo();
  }, []);

  if (loading) return <div className="card">Chargement...</div>;
  if (error) return (
    <div className="card">
      <h2>ℹ️ Informations</h2>
      <div className="alert alert-error">{error}</div>
    </div>
  );
  if (!info) return null;

  return (
    <div className="card">
      <h2>📊 Performances du Modèle</h2>
      
      <div style={{ background: '#f8f9fa', padding: '15px', borderRadius: '8px', marginBottom: '15px' }}>
        <div className="stat-item">
          <span className="stat-label">Type:</span>
          <span className="stat-value"><span className="badge">{info.model_type}</span></span>
        </div>
        <div className="stat-item">
          <span className="stat-label">R² Score:</span>
          <span className="stat-value"><span className="badge badge-success">{info.r2_score}</span></span>
        </div>
      </div>

      <h3 style={{ marginBottom: '10px' }}>Métriques d'entraînement:</h3>
      <div className="stat-item">
        <span className="stat-label">Train R²:</span>
        <span className="stat-value">{info?.performance?.train_r2}</span>
      </div>
      <div className="stat-item">
        <span className="stat-label">Test R²:</span>
        <span className="stat-value">{info?.performance?.test_r2}</span>
      </div>
      <span className="stat-value">
  {info?.performance?.train_rmse != null
    ? info.performance.train_rmse.toFixed(2)
    : "--"}
</span>
      <span className="stat-value">
  {info?.performance?.test_rmse != null
    ? info.performance.test_rmse.toFixed(2)
    : "--"}
</span>
    </div>
  );
};

export default ModelInfo;