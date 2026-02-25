import { EcgAnalysisResult, EcgRecord } from '../types';

/**
 * DB SERVICE / REPOSITORY
 * Simula um banco de dados usando LocalStorage com IDs únicos.
 */

const STORAGE_KEY = 'cardioai_history_v1';

export const saveRecord = (analysis: EcgAnalysisResult): EcgRecord => {
  const record: EcgRecord = {
    ...analysis,
    id: crypto.randomUUID(),
    timestamp: Date.now(),
    synced: false
  };

  try {
    const existingData = localStorage.getItem(STORAGE_KEY);
    const history: EcgRecord[] = existingData ? JSON.parse(existingData) : [];
    
    // Prepend new record
    const updatedHistory = [record, ...history].slice(0, 50); // Keep last 50
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updatedHistory));
    
    return record;
  } catch (error) {
    console.error("Database Error (Storage):", error);
    throw new Error("Failed to persist clinical record.");
  }
};

export const updateAdjudication = (id: string, adjudication: any): void => {
  try {
    const existingData = localStorage.getItem(STORAGE_KEY);
    if (!existingData) return;
    
    const history: any[] = JSON.parse(existingData);
    const index = history.findIndex(r => r.id === id);
    
    if (index !== -1) {
      history[index].adjudication = adjudication;
      localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
    } else {
      // If record not found (maybe not saved yet?), we can't update it easily without the full object.
      // Ideally the app should save on analysis completion.
      console.warn(`Record with ID ${id} not found for adjudication update.`);
    }
  } catch (error) {
    console.error("Database Error (Update):", error);
    throw new Error("Failed to update clinical record.");
  }
};

export const getHistory = (): EcgRecord[] => {
  try {
    const data = localStorage.getItem(STORAGE_KEY);
    return data ? JSON.parse(data) : [];
  } catch (error) {
    return [];
  }
};

export const clearHistory = (): void => {
  localStorage.removeItem(STORAGE_KEY);
};
