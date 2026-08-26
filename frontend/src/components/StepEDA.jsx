import React, { useState } from 'react';
import { useStore } from '../store/useStore';
import Plot from 'react-plotly.js';
import { Target, AlertCircle, Settings2, ChevronDown, Image as ImageIcon } from 'lucide-react';

export default function StepEDA() {
  const { datasetPreview, setEDAOptions, setStep, targetColumn: storeTargetColumn, deepFeatureColumns: storedDeepFeatureColumns } = useStore();
  
  const columns = datasetPreview?.columns || [];
  const previewData = datasetPreview?.preview || [];
  const dtypes = datasetPreview?.dtypes || {};
  const numericColumns = columns.filter((c) => dtypes[c]?.includes('int') || dtypes[c]?.includes('float'));

  // Auto-select the first numeric column instead of just the first column (which is often a Date string)
  const defaultTarget = columns.find((c) => dtypes[c]?.includes('int') || dtypes[c]?.includes('float')) || columns[0] || '';

  const [targetColumn, setTargetColumn] = useState(storeTargetColumn || defaultTarget);
  const [deepFeatureColumns, setDeepFeatureColumns] = useState(() => (
    (storedDeepFeatureColumns || []).filter((column) => (
      column !== (storeTargetColumn || defaultTarget) && numericColumns.includes(column)
    ))
  ));
  const availableDeepFeatures = numericColumns.filter((column) => column !== targetColumn);
  const [missingValue, setMissingValue] = useState('ffill');
  const [outlier, setOutlier] = useState('none');
  const [activeTab, setActiveTab] = useState('line');
  const [graphDiv, setGraphDiv] = useState(null);

  const handleContinue = () => {
    const selectedDeepFeatures = deepFeatureColumns.filter((column) => (
      column !== targetColumn && numericColumns.includes(column)
    ));
    setEDAOptions(targetColumn, missingValue, outlier, selectedDeepFeatures);
    setStep('MODEL');
  };

  const handleTargetChange = (nextTarget) => {
    setTargetColumn(nextTarget);
    // The forecast target cannot also be an input feature.
    setDeepFeatureColumns((current) => current.filter((column) => column !== nextTarget));
  };

  const toggleDeepFeature = (column) => {
    setDeepFeatureColumns((current) => (
      current.includes(column)
        ? current.filter((feature) => feature !== column)
        : [...current, column]
    ));
  };

  const allDeepFeaturesSelected = availableDeepFeatures.length > 0
    && availableDeepFeatures.every((column) => deepFeatureColumns.includes(column));

  const toggleAllDeepFeatures = () => {
    setDeepFeatureColumns(allDeepFeaturesSelected ? [] : availableDeepFeatures);
  };

  const handleDownloadGraph = async () => {
    if (graphDiv) {
      try {
        const Plotly = (await import('plotly.js-dist-min')).default || await import('plotly.js-dist-min');
        Plotly.downloadImage(graphDiv, { format: 'png', filename: `eda_${activeTab}_${targetColumn}` });
      } catch (err) {
        console.error('Failed to save graph', err);
      }
    }
  };

  const plotData = previewData.map((row, i) => ({
    x: i,
    y: row[targetColumn]
  }));

  const renderPlot = () => {
    if (activeTab === 'heatmap') {
      const corr = datasetPreview?.correlation_matrix || {};
      const cols = Object.keys(corr);
      if (cols.length === 0) return <div className="flex items-center justify-center h-full text-gray-500">No numeric columns for correlation</div>;
      
      const zValues = cols.map(colY => cols.map(colX => corr[colY][colX]));
      
      return (
        <Plot
          data={[{
            z: zValues,
            x: cols,
            y: cols,
            type: 'heatmap',
            colorscale: 'RdBu',
            zmin: -1,
            zmax: 1
          }]}
          layout={{
            autosize: true,
            margin: { t: 20, r: 20, b: 80, l: 80 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
          }}
          config={{
            displayModeBar: true,
            toImageButtonOptions: { filename: `eda_heatmap_${targetColumn}`, format: 'png', scale: 2 }
          }}
          onInitialized={(figure, graphDiv) => setGraphDiv(graphDiv)}
          onUpdate={(figure, graphDiv) => setGraphDiv(graphDiv)}
          style={{width: '100%', height: '100%'}}
          useResizeHandler={true}
        />
      );
    }
    
    if (activeTab === 'histogram') {
      return (
        <Plot
          data={[{
            x: plotData.map((d) => d.y),
            type: 'histogram',
            marker: {color: '#3b82f6'},
            opacity: 0.7
          }]}
          layout={{
            autosize: true,
            margin: { t: 20, r: 20, b: 50, l: 60 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            xaxis: { title: { text: targetColumn || 'Value', font: { size: 14 } } },
            yaxis: { title: { text: 'Count', font: { size: 14 } } }
          }}
          config={{
            displayModeBar: true,
            toImageButtonOptions: { filename: `eda_histogram_${targetColumn}`, format: 'png', scale: 2 }
          }}
          onInitialized={(figure, graphDiv) => setGraphDiv(graphDiv)}
          onUpdate={(figure, graphDiv) => setGraphDiv(graphDiv)}
          style={{width: '100%', height: '100%'}}
          useResizeHandler={true}
        />
      );
    }
    
    if (activeTab === 'boxplot') {
      return (
        <Plot
          data={[{
            y: plotData.map((d) => d.y),
            type: 'box',
            name: targetColumn,
            marker: {color: '#3b82f6'}
          }]}
          layout={{
            autosize: true,
            margin: { t: 20, r: 20, b: 50, l: 60 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            yaxis: { title: { text: targetColumn || 'Value', font: { size: 14 } } }
          }}
          config={{
            displayModeBar: true,
            toImageButtonOptions: { filename: `eda_boxplot_${targetColumn}`, format: 'png', scale: 2 }
          }}
          onInitialized={(figure, graphDiv) => setGraphDiv(graphDiv)}
          onUpdate={(figure, graphDiv) => setGraphDiv(graphDiv)}
          style={{width: '100%', height: '100%'}}
          useResizeHandler={true}
        />
      );
    }

    // Default: Line Plot
    return (
      <Plot
        data={[
          {
            x: plotData.map((d) => d.x),
            y: plotData.map((d) => d.y),
            type: 'scatter',
            mode: 'lines+markers',
            marker: {color: '#3b82f6', size: 4},
            line: {color: '#60a5fa', width: 2}
          },
        ]}
        layout={{
          autosize: true, 
          margin: { t: 20, r: 20, b: 50, l: 60 },
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          xaxis: { 
            gridcolor: '#e2e8f0',
            title: { text: 'Time Step (Index)', font: { size: 14 } } 
          },
          yaxis: { 
            gridcolor: '#e2e8f0',
            title: { text: targetColumn || 'Value', font: { size: 14 } }
          }
        }}
        config={{
          displayModeBar: true,
          toImageButtonOptions: { filename: `eda_lineplot_${targetColumn}`, format: 'png', scale: 2 }
        }}
        onInitialized={(figure, graphDiv) => setGraphDiv(graphDiv)}
        onUpdate={(figure, graphDiv) => setGraphDiv(graphDiv)}
        style={{width: '100%', height: '100%'}}
        useResizeHandler={true}
      />
    );
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8 pb-12">
      <div className="bg-white dark:bg-gray-900 p-8 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-800">
        
        <div className="mb-8">
          <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-100 mb-2">Exploratory Data Analysis</h2>
          <p className="text-gray-500 dark:text-gray-400">Select the target column you want to forecast to preview the raw time-series data.</p>
        </div>
        
        <div className="bg-blue-50/50 dark:bg-blue-900/10 border border-blue-100 dark:border-blue-800/50 rounded-xl p-6 mb-8">
          <label className="flex items-center text-sm font-semibold text-gray-800 dark:text-gray-200 mb-3">
            <Target className="w-4 h-4 mr-2 text-blue-500" />
            Target Column to Forecast
          </label>
          <div className="relative max-w-sm">
            <select 
              className="w-full appearance-none bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 text-gray-900 dark:text-white rounded-lg pl-4 pr-10 py-3 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all font-medium cursor-pointer"
              value={targetColumn} 
              onChange={e => handleTargetChange(e.target.value)}
            >
              {columns.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
            <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-3 text-gray-500 dark:text-gray-400">
              <ChevronDown className="w-5 h-5" />
            </div>
          </div>
        </div>

        <div className="bg-purple-50/50 dark:bg-purple-900/10 border border-purple-100 dark:border-purple-800/50 rounded-xl p-6 mb-8">
          <h3 className="text-sm font-semibold text-gray-800 dark:text-gray-200 mb-1">Deep-learning input features</h3>
          <p className="text-xs text-gray-500 dark:text-gray-400 mb-4">
            Optional numeric columns for LSTM, GRU, Hybrid, Bi-LSTM, and Transformer. Their future values are forecast with the worker's VAR(1) feature model when a final forecast is generated.
          </p>
          {availableDeepFeatures.length > 0 ? (
            <div className="space-y-3">
              <label className="flex items-center gap-2 text-sm font-medium text-purple-700 dark:text-purple-300 cursor-pointer">
                <input
                  type="checkbox"
                  checked={allDeepFeaturesSelected}
                  onChange={toggleAllDeepFeatures}
                  className="rounded border-gray-300 text-purple-600 focus:ring-purple-500"
                />
                {allDeepFeaturesSelected ? 'Clear all features' : 'Select all features'}
              </label>
              <div className="flex flex-wrap gap-x-5 gap-y-3">
                {availableDeepFeatures.map((column) => (
                <label key={column} className="flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={deepFeatureColumns.includes(column)}
                    onChange={() => toggleDeepFeature(column)}
                    className="rounded border-gray-300 text-purple-600 focus:ring-purple-500"
                  />
                  {column}
                </label>
                ))}
              </div>
            </div>
          ) : (
            <p className="text-sm text-gray-500 dark:text-gray-400">No additional numeric columns are available.</p>
          )}
        </div>

        <div className="mb-10">
          <div className="flex flex-col md:flex-row md:items-center justify-between mb-4 gap-4">
            <div className="flex flex-wrap gap-2">
              <button 
                onClick={() => setActiveTab('line')}
                className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${activeTab === 'line' ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/50 dark:text-blue-300' : 'text-gray-600 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-800'}`}
              >
                Line Plot
              </button>
              <button 
                onClick={() => setActiveTab('heatmap')}
                className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${activeTab === 'heatmap' ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/50 dark:text-blue-300' : 'text-gray-600 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-800'}`}
              >
                Correlation
              </button>
              <button 
                onClick={() => setActiveTab('histogram')}
                className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${activeTab === 'histogram' ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/50 dark:text-blue-300' : 'text-gray-600 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-800'}`}
              >
                Histogram
              </button>
              <button 
                onClick={() => setActiveTab('boxplot')}
                className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${activeTab === 'boxplot' ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/50 dark:text-blue-300' : 'text-gray-600 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-800'}`}
              >
                Box Plot
              </button>
            </div>
            <span className="bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300 px-3 py-1 text-xs font-medium rounded-full border border-gray-200 dark:border-gray-700 whitespace-nowrap">
              {activeTab === 'heatmap' ? 'All numeric columns' : 'Showing first 100 rows'}
            </span>
          </div>
          <div id="eda-plot-container" className="border border-gray-200 dark:border-gray-700 rounded-xl overflow-hidden h-[400px] bg-gray-50 dark:bg-gray-800/30">
            {renderPlot()}
          </div>
          <div className="flex justify-end mt-4">
            <button 
              onClick={handleDownloadGraph}
              className="flex items-center space-x-2 px-4 py-2 text-sm font-medium text-blue-700 bg-blue-50 hover:bg-blue-100 rounded-lg dark:bg-blue-900/30 dark:text-blue-300 dark:hover:bg-blue-900/50 transition-colors border border-blue-200 dark:border-blue-800/50"
            >
              <ImageIcon className="w-4 h-4" />
              <span>Save Graph Image</span>
            </button>
          </div>
        </div>

        <div className="border-t border-gray-200 dark:border-gray-700 pt-8 mb-8">
          <div className="mb-6">
            <h3 className="text-lg font-semibold flex items-center text-gray-800 dark:text-gray-200">
              <Settings2 className="w-5 h-5 mr-2 text-gray-500" />
              Preprocessing Strategies
            </h3>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              Configure how the data will be cleaned before training. These strategies will be applied automatically by the pipeline.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="flex items-center text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                <AlertCircle className="w-4 h-4 mr-2 text-orange-500" />
                Missing Value Strategy
              </label>
              <div className="relative">
                <select 
                  className="w-full appearance-none bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 text-gray-900 dark:text-white rounded-lg pl-4 pr-10 py-2.5 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all cursor-pointer"
                  value={missingValue} 
                  onChange={e => setMissingValue(e.target.value)}
                >
                  <option value="ffill">Forward Fill (Recommended)</option>
                  <option value="bfill">Backward Fill</option>
                  <option value="mean">Mean Imputation</option>
                  <option value="drop">Drop Rows</option>
                </select>
                <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-3 text-gray-500 dark:text-gray-400">
                  <ChevronDown className="w-4 h-4" />
                </div>
              </div>
            </div>
            
            <div>
              <label className="flex items-center text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                <AlertCircle className="w-4 h-4 mr-2 text-red-500" />
                Outlier Strategy
              </label>
              <div className="relative">
                <select 
                  className="w-full appearance-none bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 text-gray-900 dark:text-white rounded-lg pl-4 pr-10 py-2.5 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all cursor-pointer"
                  value={outlier} 
                  onChange={e => setOutlier(e.target.value)}
                >
                  <option value="none">None (Keep All Data)</option>
                  <option value="clip">Clip (IQR Method)</option>
                  <option value="drop">Drop (IQR Method)</option>
                </select>
                <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-3 text-gray-500 dark:text-gray-400">
                  <ChevronDown className="w-4 h-4" />
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="flex justify-between items-center pt-6 border-t border-gray-100 dark:border-gray-800">
          <button 
            onClick={() => setStep('UPLOAD')}
            className="px-6 py-2.5 text-sm font-medium text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-300 dark:hover:bg-gray-700 transition-colors"
          >
            Back
          </button>
          <button 
            onClick={handleContinue}
            className="px-8 py-2.5 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 shadow-sm transition-colors"
          >
            Continue to Models
          </button>
        </div>
      </div>
    </div>
  );
}
