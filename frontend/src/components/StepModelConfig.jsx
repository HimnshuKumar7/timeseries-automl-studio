import React, { useState, useEffect } from 'react';
import { useStore } from '../store/useStore';

export default function StepModelConfig() {
  const { setModelsToTrain, setStep, modelParams, setModelParams } = useStore();
  
  const [selectedModels, setSelectedModels] = useState({
    arima: false,
    varma: false,
    boosting: false,
    lstm: false,
    linear: false,
    randomForest: false,
    gru: false,
    lstmGruHybrid: false,
    bilstm: false,
    transformer: false
  });
  
  const [deviceCapability, setDeviceCapability] = useState('Detecting...');

  useEffect(() => {
    // Basic device detection to set expectations
    const cores = navigator.hardwareConcurrency || 2;
    if (cores >= 8) {
      setDeviceCapability('High-end device detected. Training should be fast.');
    } else if (cores >= 4) {
      setDeviceCapability('Standard device detected. Deep learning might take a moment.');
    } else {
      setDeviceCapability('Low-power device detected. Consider disabling LSTM for faster results.');
    }
  }, []);

  const handleToggle = (model) => {
    setSelectedModels(prev => ({
      ...prev,
      [model]: !prev[model]
    }));
  };

  const handleContinue = () => {
    const toTrain = Object.keys(selectedModels).filter(k => selectedModels[k]);
    if (toTrain.length === 0) {
      alert("Please select at least one model to train.");
      return;
    }
    setModelsToTrain(toTrain);
    setStep('TRAIN');
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <div className="bg-white dark:bg-gray-900 p-6 rounded-xl shadow-sm border border-gray-100 dark:border-gray-800">
        <h2 className="text-2xl font-bold mb-2 text-gray-800 dark:text-gray-100">Model Selection</h2>
        <p className="text-gray-600 dark:text-gray-400 mb-6">
          All models will train locally on your device. {deviceCapability}
        </p>
        <div className="mb-6 p-4 rounded-lg bg-blue-50/50 dark:bg-blue-950/30 border border-blue-100 dark:border-blue-900/50 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h3 className="font-semibold text-gray-800 dark:text-gray-200 text-base">Forecast Horizon</h3>
            <p className="text-xs text-gray-500 dark:text-gray-400">Specify how many steps/days ahead to predict (e.g. 2 steps for short-term accuracy).</p>
          </div>
          <div className="flex items-center space-x-2">
            <input 
              type="number" 
              min="1" 
              max="60" 
              value={modelParams.forecastHorizon || 2} 
              onChange={e => setModelParams({ forecastHorizon: Math.max(1, parseInt(e.target.value) || 2) })} 
              className="w-24 text-sm font-bold text-center p-2 border rounded-md dark:bg-gray-800 dark:border-gray-700 dark:text-white"
            />
            <span className="text-sm font-medium text-gray-600 dark:text-gray-300">Steps Ahead</span>
          </div>
        </div>

        <div className="space-y-4 mb-8">
          {/* ARIMA */}
          <div 
            className={`p-4 rounded-lg border-2 cursor-pointer transition-colors ${selectedModels.arima ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' : 'border-gray-200 dark:border-gray-700 hover:border-blue-300'}`}
            onClick={() => handleToggle('arima')}
          >
            <div className="flex items-center justify-between">
              <div>
                <h3 className="font-semibold text-lg dark:text-white">ARIMA (Statsmodels)</h3>
                <p className="text-sm text-gray-500 dark:text-gray-400">Classical statistical method. Fast and reliable for simple trends.</p>
              </div>
              <input type="checkbox" checked={selectedModels.arima} readOnly className="w-5 h-5 text-blue-600 rounded" />
            </div>
          </div>

          {/* VARMA */}
          <div 
            className={`p-4 rounded-lg border-2 cursor-pointer transition-colors ${selectedModels.varma ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' : 'border-gray-200 dark:border-gray-700 hover:border-blue-300'}`}
            onClick={() => handleToggle('varma')}
          >
            <div className="flex items-center justify-between">
              <div>
                <h3 className="font-semibold text-lg dark:text-white">VARMA (Statsmodels)</h3>
                <p className="text-sm text-gray-500 dark:text-gray-400">Vector Autoregressive Moving Average model for joint temporal dynamics.</p>
              </div>
              <input type="checkbox" checked={selectedModels.varma} readOnly className="w-5 h-5 text-blue-600 rounded" />
            </div>
          </div>

          {/* Boosting */}
          <div 
            className={`p-4 rounded-lg border-2 cursor-pointer transition-colors ${selectedModels.boosting ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' : 'border-gray-200 dark:border-gray-700 hover:border-blue-300'}`}
            onClick={() => handleToggle('boosting')}
          >
            <div className="flex items-center justify-between">
              <div>
                <h3 className="font-semibold text-lg dark:text-white">Gradient Boosting (Scikit-Learn)</h3>
                <p className="text-sm text-gray-500 dark:text-gray-400">Machine learning approach (XGBoost alternative). Good for non-linear patterns.</p>
              </div>
              <input type="checkbox" checked={selectedModels.boosting} readOnly className="w-5 h-5 text-blue-600 rounded" />
            </div>
          </div>

          {/* LSTM */}
          <div 
            className={`p-4 rounded-lg border-2 cursor-pointer transition-colors ${selectedModels.lstm ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' : 'border-gray-200 dark:border-gray-700 hover:border-blue-300'}`}
            onClick={() => handleToggle('lstm')}
          >
            <div className="flex items-center justify-between">
              <div>
                <h3 className="font-semibold text-lg dark:text-white">LSTM (TensorFlow.js)</h3>
                <p className="text-sm text-gray-500 dark:text-gray-400">Deep learning model. Computationally intensive but catches complex sequences. Uses WebGL/WebGPU.</p>
              </div>
              <input type="checkbox" checked={selectedModels.lstm} readOnly className="w-5 h-5 text-blue-600 rounded" />
            </div>
          </div>

          {/* Linear Regression */}
          <div 
            className={`p-4 rounded-lg border-2 cursor-pointer transition-colors ${selectedModels.linear ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' : 'border-gray-200 dark:border-gray-700 hover:border-blue-300'}`}
            onClick={() => handleToggle('linear')}
          >
            <div className="flex items-center justify-between">
              <div>
                <h3 className="font-semibold text-lg dark:text-white">Linear Regression (Scikit-Learn)</h3>
                <p className="text-sm text-gray-500 dark:text-gray-400">Simple linear model. Fast, good baseline for simple trends.</p>
              </div>
              <input type="checkbox" checked={selectedModels.linear} readOnly className="w-5 h-5 text-blue-600 rounded" />
            </div>
          </div>

          {/* Random Forest Regressor */}
          <div 
            className={`p-4 rounded-lg border-2 cursor-pointer transition-colors ${selectedModels.randomForest ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' : 'border-gray-200 dark:border-gray-700 hover:border-blue-300'}`}
          >
            <div className="flex items-center justify-between" onClick={() => handleToggle('randomForest')}>
              <div>
                <h3 className="font-semibold text-lg dark:text-white">Random Forest (Scikit-Learn)</h3>
                <p className="text-sm text-gray-500 dark:text-gray-400">Ensemble learning method. Highly effective for complex non-linear patterns.</p>
              </div>
              <input type="checkbox" checked={selectedModels.randomForest} readOnly className="w-5 h-5 text-blue-600 rounded" />
            </div>
            
            {selectedModels.randomForest && (
              <div className="mt-4 pt-4 border-t border-blue-200 dark:border-blue-800 space-y-3" onClick={(e) => e.stopPropagation()}>
                <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">Hyperparameters</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">N Estimators</label>
                    <input type="number" min="10" max="1000" value={modelParams.rfEstimators} onChange={e => setModelParams({ rfEstimators: parseInt(e.target.value) || 100 })} className="w-full text-sm p-2 border rounded dark:bg-gray-800 dark:border-gray-700 dark:text-white" />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Max Depth (0 = None)</label>
                    <input type="number" min="0" max="100" value={modelParams.rfMaxDepth} onChange={e => setModelParams({ rfMaxDepth: parseInt(e.target.value) || 0 })} className="w-full text-sm p-2 border rounded dark:bg-gray-800 dark:border-gray-700 dark:text-white" />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Min Samples Split</label>
                    <input type="number" min="2" max="20" value={modelParams.rfMinSamplesSplit} onChange={e => setModelParams({ rfMinSamplesSplit: parseInt(e.target.value) || 2 })} className="w-full text-sm p-2 border rounded dark:bg-gray-800 dark:border-gray-700 dark:text-white" />
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* GRU */}
          <div 
            className={`p-4 rounded-lg border-2 cursor-pointer transition-colors ${selectedModels.gru ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' : 'border-gray-200 dark:border-gray-700 hover:border-blue-300'}`}
            onClick={() => handleToggle('gru')}
          >
            <div className="flex items-center justify-between">
              <div>
                <h3 className="font-semibold text-lg dark:text-white">GRU (TensorFlow.js)</h3>
                <p className="text-sm text-gray-500 dark:text-gray-400">Lighter, faster alternative to LSTM with fewer parameters.</p>
              </div>
              <input type="checkbox" checked={selectedModels.gru} readOnly className="w-5 h-5 text-blue-600 rounded" />
            </div>
          </div>

          {/* LSTM-GRU Hybrid */}
          <div 
            className={`p-4 rounded-lg border-2 cursor-pointer transition-colors ${selectedModels.lstmGruHybrid ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' : 'border-gray-200 dark:border-gray-700 hover:border-blue-300'}`}
            onClick={() => handleToggle('lstmGruHybrid')}
          >
            <div className="flex items-center justify-between">
              <div>
                <h3 className="font-semibold text-lg dark:text-white">LSTM-GRU Hybrid (TensorFlow.js)</h3>
                <p className="text-sm text-gray-500 dark:text-gray-400">Stacked LSTM + GRU layers. Heavier than either model alone.</p>
              </div>
              <input type="checkbox" checked={selectedModels.lstmGruHybrid} readOnly className="w-5 h-5 text-blue-600 rounded" />
            </div>
          </div>

          {/* Bi-LSTM */}
          <div 
            className={`p-4 rounded-lg border-2 cursor-pointer transition-colors ${selectedModels.bilstm ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' : 'border-gray-200 dark:border-gray-700 hover:border-blue-300'}`}
            onClick={() => handleToggle('bilstm')}
          >
            <div className="flex items-center justify-between">
              <div>
                <h3 className="font-semibold text-lg dark:text-white">Bi-LSTM (TensorFlow.js)</h3>
                <p className="text-sm text-gray-500 dark:text-gray-400">Reads the sequence forward and backward for richer context.</p>
              </div>
              <input type="checkbox" checked={selectedModels.bilstm} readOnly className="w-5 h-5 text-blue-600 rounded" />
            </div>
          </div>

          {/* Transformer */}
          <div 
            className={`p-4 rounded-lg border-2 cursor-pointer transition-colors ${selectedModels.transformer ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' : 'border-gray-200 dark:border-gray-700 hover:border-blue-300'}`}
            onClick={() => handleToggle('transformer')}
          >
            <div className="flex items-center justify-between">
              <div>
                <h3 className="font-semibold text-lg dark:text-white">Transformer (TensorFlow.js)</h3>
                <p className="text-sm text-gray-500 dark:text-gray-400">Self-attention model. Most computationally intensive option.</p>
              </div>
              <input type="checkbox" checked={selectedModels.transformer} readOnly className="w-5 h-5 text-blue-600 rounded" />
            </div>
          </div>
        </div>

        <div className="bg-gray-50 dark:bg-gray-800/50 p-4 rounded-lg border border-gray-200 dark:border-gray-700 mb-8 space-y-5">
          <h3 className="font-semibold text-gray-800 dark:text-gray-100 mb-2">Global Training Settings</h3>
          <div>
            <label className="block text-sm text-gray-700 dark:text-gray-300 mb-1">
              Train/Test Split Fraction (For Evaluation)
            </label>
            <div className="flex items-center space-x-4">
              <input 
                type="range" min="0.05" max="0.5" step="0.05" 
                value={modelParams.splitFraction} 
                onChange={e => setModelParams({ splitFraction: parseFloat(e.target.value) })}
                className="w-full max-w-xs"
              />
              <span className="text-sm font-medium text-gray-600 dark:text-gray-400">{Math.round(modelParams.splitFraction * 100)}% Test</span>
            </div>
          </div>
          <div>
            <label className="block text-sm text-gray-700 dark:text-gray-300 mb-1">
              Deep Learning Epochs (LSTM / GRU / Hybrid / Bi-LSTM / Transformer)
            </label>
            <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">Applies to all deep learning models trained in this run. More epochs can improve accuracy but takes longer.</p>
            <div className="flex items-center space-x-4">
              <input 
                type="range" min="1" max="50" step="1" 
                value={modelParams.epochs ?? 4} 
                onChange={e => setModelParams({ epochs: parseInt(e.target.value) || 1 })}
                className="w-full max-w-xs"
              />
              <span className="text-sm font-medium text-gray-600 dark:text-gray-400 w-24">{modelParams.epochs ?? 4} Epochs</span>
            </div>
          </div>
        </div>

        <div className="flex justify-between items-center pt-4 border-t border-gray-100 dark:border-gray-800">
          <button 
            onClick={() => setStep('EDA')}
            className="px-4 py-2 text-sm font-medium text-gray-600 bg-gray-100 rounded-md hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-300 dark:hover:bg-gray-700"
          >
            Back
          </button>
          <button 
            onClick={handleContinue}
            className="px-6 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 shadow-sm"
          >
            Start Training
          </button>
        </div>
      </div>
    </div>
  );
}
