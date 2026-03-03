import React, { useState } from 'react';
import axios from 'axios';
import { Send, RefreshCw, AlertCircle, CheckCircle } from 'lucide-react';

const PredictForm = () => {
  const [features, setFeatures] = useState(Array(48).fill('0'));
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);

  const handleFeatureChange = (index, value) => {
    const newFeatures = [...features];
    newFeatures[index] = value;
    setFeatures(newFeatures);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setSuccess(false);
    setPrediction(null);

    try {
      // Convertir les strings en nombres
      const numericFeatures = features.map(v => parseFloat(v) || 0);
      
      const response = await axios.post('http://127.0.0.1:8000/predict', {
        features: numericFeatures
      });

      setPrediction(response.data.prediction);
      setSuccess(true);
    } catch (err) {
      setError(err.response?.data?.detail || 'Erreur lors de la prédiction');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const generateRandomFeatures = () => {
    const randomFeatures = Array(48).fill(0).map(() => 
      (Math.random() * 100).toFixed(2)
    );
    setFeatures(randomFeatures);
  };

  const resetForm = () => {
    setFeatures(Array(48).fill('0'));
    setPrediction(null);
    setError(null);
    setSuccess(false);
  };

  return (
    <div className="predict-form p-4 border rounded-lg shadow-md">
      <h2 className="text-xl font-bold mb-4">Faire une Prédiction</h2>

      <div className="mb-4 flex gap-2">
        <button
          onClick={generateRandomFeatures}
          className="bg-gray-500 hover:bg-gray-600 text-white px-3 py-1 rounded flex items-center text-sm"
        >
          <RefreshCw size={16} className="mr-1" /> Aléatoire
        </button>
        <button
          onClick={resetForm}
          className="bg-gray-300 hover:bg-gray-400 text-gray-800 px-3 py-1 rounded text-sm"
        >
          Réinitialiser
        </button>
      </div>

      <form onSubmit={handleSubmit}>
        <div className="grid grid-cols-4 gap-2 mb-4 max-h-60 overflow-y-auto p-2 border rounded">
          {features.map((value, index) => (
            <div key={index} className="flex flex-col">
              <label className="text-xs text-gray-500">f{index + 1}</label>
              <input
                type="number"
                step="0.01"
                value={value}
                onChange={(e) => handleFeatureChange(index, e.target.value)}
                className="border rounded px-2 py-1 text-sm w-full"
                required
              />
            </div>
          ))}
        </div>

        <br></br>

        <button
          type="submit"
          disabled={loading}
          className="w-full bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded flex items-center justify-center disabled:opacity-50"
        >
          {loading ? (
            <>
              <RefreshCw className="animate-spin mr-2" size={18} />
              Prédiction en cours...
            </>
          ) : (
            <>
              <Send className="mr-2" size={18} />
              Prédire
            </>
          )}
        </button>
      </form>

      {error && (
        <div className="mt-4 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded flex items-start">
          <AlertCircle className="mr-2 flex-shrink-0 mt-0.5" size={18} />
          <span>{error}</span>
        </div>
      )}

      {success && prediction !== null && (
        <div className="mt-4 bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded">
          <div className="flex items-center mb-2">
            <CheckCircle className="mr-2" size={18} />
            <span className="font-bold">Prédiction réussie!</span>
          </div>
          <div className="text-center">
            <span className="text-3xl font-bold">{prediction.toFixed(2)}</span>
            <span className="text-gray-600 ml-2">(valeur prédite)</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default PredictForm;