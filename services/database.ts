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
