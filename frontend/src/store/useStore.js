import { create } from 'zustand';
import { persist } from 'zustand/middleware';

/**
 * @typedef {'UPLOAD' | 'EDA' | 'MODEL' | 'TRAIN' | 'RESULTS'} WizardStep
 *
 * @typedef {Object} DatasetPreview
 * @property {string[]} columns
 * @property {number[]} shape
 * @property {Record<string, string>} dtypes
 * @property {any[]} preview
 * @property {Record<string, Record<string, number>>} [correlation_matrix]
 * @property {Record<string, Record<string, number>>} [summary_statistics]
 */

export const useStore = create(
  persist(
    (set) => ({
      currentStep: 'UPLOAD',
      datasetId: null,
      datasetPreview: null,
      targetColumn: null,
      deepFeatureColumns: [],
      preprocessing: {
        missingValueStrategy: 'ffill',
        outlierStrategy: 'none'
      },
      modelsToTrain: [],
      modelResults: {},
      modelParams: {
        splitFraction: 0.2,
        forecastHorizon: 2,
        epochs: 4,
        rfEstimators: 100,
        rfMaxDepth: 0, // 0 means None
        rfMinSamplesSplit: 2,
      },
      
      setStep: (step) => set({ currentStep: step }),
      setDataset: (id, preview) => set({ datasetId: id, datasetPreview: preview }),
      setEDAOptions: (target, missing, outliers, deepFeatures = []) => set({
        targetColumn: target,
        deepFeatureColumns: deepFeatures,
        preprocessing: { missingValueStrategy: missing, outlierStrategy: outliers }
      }),
      setModelsToTrain: (models) => set({ modelsToTrain: models }),
      setModelResults: (results) => set({ modelResults: results }),
      setModelParams: (params) => set((state) => ({ modelParams: { ...state.modelParams, ...params } })),
      reset: () => set({
        currentStep: 'UPLOAD',
        datasetId: null,
        datasetPreview: null,
        targetColumn: null,
        deepFeatureColumns: [],
        preprocessing: {
          missingValueStrategy: 'ffill',
          outlierStrategy: 'none'
        },
        modelsToTrain: [],
        modelResults: {},
        modelParams: {
          splitFraction: 0.2,
          forecastHorizon: 2,
          epochs: 4,
          rfEstimators: 100,
          rfMaxDepth: 0,
          rfMinSamplesSplit: 2,
        }
      }),
    }),
    {
      name: 'automl-studio-state',
    }
  )
);
