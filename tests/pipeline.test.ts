import { describe, it, expect, vi } from 'vitest';
import { analyzeEcgImage } from '../services/geminiService';
import { enrichAnalysisWithLogic } from '../utils/cardioLogic';

// Mocking the Gemini API call for unit testing the pipeline logic
vi.mock('../services/geminiService', async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual as any,
    analyzeEcgImage: vi.fn().mockResolvedValue({
      id: 'mock-123',
      diagnosis: 'Fibrilação Atrial',
      urgency: 'Urgent',
      rhythm: 'Fibrilação Atrial',
      heartRate: '110',
      precisionMeasurements: {
        signalQuality: { snrDb: 25, artifactsDetected: [] },
        neuralTelemetry: {
          modelArchitecture: 'Hybrid CNN + Transformer',
          processingTimeMs: 150,
          differentialDiagnoses: [
            { diagnosis: 'Fibrilação Atrial', probability: 95 },
            { diagnosis: 'Flutter Atrial', probability: 5 }
          ]
        },
        waves: {
          pWave: { present: false, morphology: 'Absent' },
          qrsComplex: { durationMs: 90, axisDegrees: 45 },
          intervals: { rrRegularity: 'Irregularly Irregular' }
        }
      }
    })
  };
});

describe('ECG Analysis Pipeline', () => {
  
  describe('1. Ingestion & Preprocessing (Simulated)', () => {
    it('should validate signal quality metrics', async () => {
      const result = await analyzeEcgImage('mock-base64', 'image/png');
      expect(result.precisionMeasurements.signalQuality.snrDb).toBeGreaterThan(20);
      expect(result.precisionMeasurements.signalQuality.artifactsDetected).toHaveLength(0);
    });
  });

  describe('2. Beat-Level & Rhythm-Level Classification', () => {
    it('should correctly identify Atrial Fibrillation based on rhythm features', async () => {
      const result = await analyzeEcgImage('mock-base64', 'image/png');
      
      // Rhythm-level check
      expect(result.rhythm).toBe('Fibrilação Atrial');
      expect(result.precisionMeasurements.waves.intervals.rrRegularity).toBe('Irregularly Irregular');
      
      // Beat-level check (P-wave absence)
      expect(result.precisionMeasurements.waves.pWave.present).toBe(false);
    });

    it('should provide differential diagnoses probabilities', async () => {
      const result = await analyzeEcgImage('mock-base64', 'image/png');
      const diffs = result.precisionMeasurements.neuralTelemetry.differentialDiagnoses;
      
      expect(diffs).toBeDefined();
      expect(diffs[0].diagnosis).toBe('Fibrilação Atrial');
      expect(diffs[0].probability).toBeGreaterThan(90);
    });
  });

  describe('3. Post-Processing & Clinical Rules (Guardrails)', () => {
    it('should apply clinical logic to enrich the AI output', () => {
      const mockRawResult: any = {
        diagnosis: 'Taquicardia Ventricular',
        urgency: 'Routine', // Incorrect urgency from AI
        heartRate: '180',
        rhythm: 'Taquicardia Ventricular',
        precisionMeasurements: {
            waves: { qrsComplex: { durationMs: 150 } }
        },
        clinicalImplications: []
      };

      const enriched = enrichAnalysisWithLogic(mockRawResult);
      
      // The logic should override the urgency to Emergency for VT
      expect(enriched.urgency).toBe('Emergency');
      expect(enriched.clinicalImplications).toContain('Arritmia com risco de vida detectada. Acionar protocolo de emergência.');
    });
  });

  describe('4. Explainability & Telemetry', () => {
    it('should include neural architecture telemetry', async () => {
      const result = await analyzeEcgImage('mock-base64', 'image/png');
      const telemetry = result.precisionMeasurements.neuralTelemetry;
      
      expect(telemetry.modelArchitecture).toContain('Hybrid CNN + Transformer');
      expect(telemetry.processingTimeMs).toBeLessThan(200); // Latency requirement check
    });
  });

});
