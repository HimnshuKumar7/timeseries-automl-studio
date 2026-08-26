import { openDB } from 'idb';

let dbPromise = null;

export const getDB = () => {
  if (!dbPromise) {
    dbPromise = openDB('automl-studio-db', 1, {
      upgrade(db) {
        db.createObjectStore('datasets', { keyPath: 'id' });
        db.createObjectStore('cleaned_data', { keyPath: 'id' });
      },
    });
  }
  return dbPromise;
};

export const storage = {
  async saveRawDataset(id, filename, rawContent) {
    const db = await getDB();
    await db.put('datasets', {
      id,
      filename,
      rawContent,
      timestamp: Date.now()
    });
  },

  async getRawDataset(id) {
    const db = await getDB();
    return db.get('datasets', id);
  },

  async saveCleanedData(id, content) {
    const db = await getDB();
    await db.put('cleaned_data', {
      id,
      content,
      timestamp: Date.now()
    });
  },

  async getCleanedData(id) {
    const db = await getDB();
    return db.get('cleaned_data', id);
  },

  async clearAll() {
    const db = await getDB();
    await db.clear('datasets');
    await db.clear('cleaned_data');
    // For TF.js models, they have their own indexedDB db "tensorflowjs"
    try {
      indexedDB.deleteDatabase('tensorflowjs');
    } catch (e) {
      console.warn("Could not delete TF.js database", e);
    }
  }
};
