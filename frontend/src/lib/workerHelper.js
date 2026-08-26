import * as Comlink from 'comlink';

let pyodideWorker = null;
let tfjsWorker = null;

export const getPyodideAPI = () => {
  if (typeof window === 'undefined') return null;
  if (!pyodideWorker) {
    pyodideWorker = new Worker(new URL('../workers/pyodide.worker.js', import.meta.url), { type: 'module' });
  }
  return Comlink.wrap(pyodideWorker);
};

export const resetPyodideWorker = () => {
  if (pyodideWorker) {
    pyodideWorker.terminate();
    pyodideWorker = null;
  }
};

export const getTfjsAPI = () => {
  if (typeof window === 'undefined') return null;
  if (!tfjsWorker) {
    tfjsWorker = new Worker(new URL('../workers/tfjs.worker.js', import.meta.url), { type: 'module' });
  }
  return Comlink.wrap(tfjsWorker);
};

// Each new training session terminates the prior TF.js worker and creates a fresh
// one on the next getTfjsAPI() call. This clears any retained WebGL/GPU/backend
// state from the previous run, which otherwise could leave the worker unable to
// train again until a full page reload.
export const resetTfjsWorker = () => {
  if (tfjsWorker) {
    tfjsWorker.terminate();
    tfjsWorker = null;
  }
};
