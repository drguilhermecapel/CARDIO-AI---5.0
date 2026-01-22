
import { EcgAnalysisResult, PatientContext } from '../types';

export const parseHeartRate = (hrString: string): number => {
  if (!hrString) return 0;
  const matches = hrString.match(/(\d+)/g);
  return matches ? parseInt(matches[0], 10) : 0;
};

export const identifyEcgPattern = (
  arrhythmias: string[], 
  diagnosis: string = '',
  prIntervalMs?: number,
  qrsDurationMs?: number
): string => {
  const text = (arrhythmias.join(' ') + ' ' + diagnosis).toLowerCase();
  
  const isWideQRS = (qrsDurationMs && qrsDurationMs >= 120) || text.includes('qrs largo') || text.includes('wide qrs');
  const isTachy = text.includes('taquicardia') || text.includes('tachycardia');

  if (isWideQRS && isTachy) {
    if (text.includes('dissociação av') || text.includes('av dissociation')) return 'vt';
    if (text.includes('r inicial em avr')) return 'vt'; 
  }

  if (text.includes('wellens')) return 'wellens';
  if (text.includes('de winter')) return 'dewinter';
  if (text.includes('bayés') || text.includes('bayes') || text.includes('interatrial')) return 'bayessyndrome';
  if (text.includes('arvd') || text.includes('displasia') || text.includes('épsilon')) return 'arvd';
  if (text.includes('marcapasso') || text.includes('pacemaker')) return 'paced';
  if (text.includes('afib')) return 'afib';
  if (text.includes('vf')) return 'vf';
  if (text.includes('vt')) return 'vt';
  
  return 'normal';
};

export const enrichAnalysisWithLogic = (result: EcgAnalysisResult, patientCtx?: PatientContext): EcgAnalysisResult => {
  const measurements = result.precisionMeasurements;
  let implications = [...result.clinicalImplications];
  const diagnosisText = result.diagnosis.toLowerCase();
  
  // Ischemia & OMI Logic
  const ischemia = measurements.ischemiaAnalysis;
  if (ischemia) {
    if (ischemia.sgarbossaScore && ischemia.sgarbossaScore >= 3) {
      implications.push("🚨 CRITÉRIOS DE SGARBOSSA POSITIVOS: Alta probabilidade de IAM em vigência de BRE ou Marcapasso.");
    }
    if (ischemia.smithSgarbossaRatio && ischemia.smithSgarbossaRatio < -0.25) {
      implications.push("⚠️ SMITH-SGARBOSSA: Razão ST/S sugere Oclusão Coronariana Aguda (OMI).");
    }
    if (ischemia.wellensSyndrome !== 'None') {
      implications.push(`🎯 SÍNDROME DE WELLENS (${ischemia.wellensSyndrome}): Estenose crítica de ADA proximal detectada.`);
    }
    if (ischemia.deWinterPattern) {
      implications.push("⚡ PADRÃO DE DE WINTER: Equivalente de STEMI de parede anterior; oclusão total de ADA proximal.");
    }
  }

  // WCT Logic
  if (measurements.qrsComplex?.durationMs && measurements.qrsComplex.durationMs >= 120) {
    if (diagnosisText.includes('rs > 100ms')) implications.push("📏 BRUGADA: Intervalo RS > 100ms favorece Taquicardia Ventricular.");
  }
  
  // Baranchuk Logic
  const baran = measurements.baranchukAnalysis;
  if (baran && baran.iabType === 'Advanced (Bayes Syndrome)') {
    implications.push("🫀 SÍNDROME DE BAYÉS: Risco de FA e Stroke elevado.");
  }

  return {
    ...result,
    clinicalImplications: Array.from(new Set(implications))
  };
};
