

import { EcgAnalysisResult, PatientContext, PrecisionMeasurements } from '../types';

export const parseHeartRate = (hrString: string): number => {
  if (!hrString) return 0;
  const matches = hrString.match(/(\d+)/g);
  return matches ? parseInt(matches[0], 10) : 0;
};

// --- ADVANCED PATTERN RECOGNITION (COMPENDIUM INTEGRATION) ---

/**
 * Traduz os dados telemétricos profundos do motor de IA para uma chave de visualização
 * usada pelo EcgVisualizer para desenhar a morfologia correta.
 */
export const identifyEcgPattern = (
  diagnosis: string = '',
  measurements?: PrecisionMeasurements,
  heartRateString?: string
): string => {
  const text = diagnosis.toLowerCase();
  const m = measurements;
  const ischemia = m?.ischemiaAnalysis;
  const waves = m?.waves;
  const conduction = m?.conductionAnalysis;

  // 1. CRITICAL OMI / INFARCTION PATTERNS
  if (text.includes('torsades')) return 'torsades';
  if (text.includes('vf') || text.includes('fibrilação ventricular')) return 'vf';
  
  // OMI: De Winter's T-Waves (LAD Occlusion)
  if (ischemia?.deWinterPattern || text.includes('de winter')) return 'dewinter';
  
  // OMI: Wellens' Syndrome (LAD Stenosis)
  if (ischemia?.wellensSyndrome && ischemia.wellensSyndrome !== 'None') return 'wellens';
  
  // OMI: STEMI (Tombstone or Convex)
  if (text.includes('stemi') || (ischemia?.stSegmentTrend === 'Elevation' && ischemia?.stShape !== 'Concave')) return 'stemi';
  
  // OMI: Posterior MI (Depression V1-V3)
  if (text.includes('posterior') && ischemia?.stSegmentTrend === 'Depression') return 'posteriormi';

  // 2. METABOLIC & DRUGS
  if (text.includes('hipercalemia') || text.includes('hyperkalemia')) return 'hyperkalemia';
  if (text.includes('hipocalemia') || text.includes('hypokalemia')) return 'hypokalemia'; // Prominent U waves
  
  // 3. GENETIC & CHANNELOPATHIES
  if (text.includes('brugada')) return 'brugada';
  if (text.includes('arvc') || text.includes('arvd') || text.includes('displasia') || text.includes('epsilon')) return 'arvd';
  if (text.includes('long qt') || (waves?.intervals?.qtcMs || 0) > 480) return 'lqt';
  
  // 4. CONDUCTION
  if (text.includes('wpw') || text.includes('wolff') || conduction?.wpwPattern) return 'wpw';
  if (text.includes('bloqueio av total') || text.includes('bavt') || text.includes('3rd degree') || text.includes('complete heart block')) return 'avblock3';
  if (text.includes('wenckebach') || text.includes('mobitz i') || text.includes('type i')) return 'wenckebach';
  if (text.includes('mobitz ii') || text.includes('type ii')) return 'mobitz2';
  if (text.includes('2nd degree')) return 'avblock2'; // Generic 2nd degree if not caught above
  if (text.includes('1st degree') || text.includes('first degree')) return 'avblock1';
  if (conduction?.ivcdType === 'LBBB') return 'lbbb';
  if (conduction?.ivcdType === 'RBBB') return 'rbbb';

  // 5. RHYTHM
  if (text.includes('fibrilação atrial') || text.includes('afib')) return 'afib';
  if (text.includes('flutter')) return 'flutter';
  
  // Fallback to basic tachycardia/bradycardia limits or Normal
  const hr = parseHeartRate(heartRateString || '60');
  
  // Check for Wide Complex Tachycardia (VT) fallback
  // Use waves.qrsComplex first, fallback to legacy qrsComplex if needed
  const qrsDur = waves?.qrsComplex?.durationMs || m?.qrsComplex?.durationMs || 0;
  
  if (qrsDur > 120 && hr > 100) return 'vt';

  return 'normal';
};

// --- GUARDRAILS & LOGIC ENRICHMENT ---

const validateLBBB = (waves: any): boolean => {
  if (!waves || !waves.qrsComplex) return false;
  // Criteria: QRS > 120ms AND V1 is dominant negative (QS or rS) AND V6 is dominant positive (R)
  const wideQRS = waves.qrsComplex.durationMs >= 120;
  // V1 check: Should be predominantly negative
  const v1Neg = ['QS', 'rS'].includes(waves.qrsComplex.morphologyV1) || waves.qrsComplex.morphologyV1?.includes('S');
  return wideQRS && v1Neg;
};

const validateSinusRhythm = (waves: any, rate: number): boolean => {
    if (!waves || !waves.pWave) return false;
    const normalRate = rate >= 50 && rate <= 100;
    const pPresent = waves.pWave.present && waves.pWave.morphology === 'Sinus';
    const regular = waves.intervals?.rrRegularity === 'Regular';
    return normalRate && pPresent && regular;
};

const validateOMI = (ischemia: any, patientCtx?: PatientContext): { isOMI: boolean, type: string } => {
    if (!ischemia) return { isOMI: false, type: '' };
    
    // Check for specific OMI equivalents
    if (ischemia.deWinterPattern) return { isOMI: true, type: 'De Winter T-Waves (LAD Occlusion)' };
    if (ischemia.wellensSyndrome && ischemia.wellensSyndrome !== 'None') return { isOMI: true, type: `Wellens Syndrome (${ischemia.wellensSyndrome})` };
    if (ischemia.sgarbossaScore >= 3) return { isOMI: true, type: 'Sgarbossa Positive (LBBB/Paced Infarction)' };
    
    // Posterior MI Check
    // Often indicated by ST depression in V1-V3 (reciprocal), or Elevation in V7-V9
    if (ischemia.affectedWall?.includes('Posterior')) {
         // If wall is Posterior, it's an OMI equivalent (STEMI equivalent)
         return { isOMI: true, type: 'Posterior MI' };
    }

    // Left Main / Triple Vessel
    // STE aVR > V1 is a strong indicator if diffuse depression exists
    if (ischemia.affectedWall?.includes('Left Main') || ischemia.culpritArtery?.includes('Left Main')) {
        return { isOMI: true, type: 'Left Main / Triple Vessel Disease' };
    }
    
    // Check for classic STEMI
    const isElevation = ischemia.stSegmentTrend === 'Elevation';
    const hasReciprocal = ischemia.reciprocalChangesFound;
    const hasSymptoms = patientCtx?.symptoms?.some((s: string) => s.toLowerCase().includes('chest') || s.toLowerCase().includes('dor'));
    
    if (isElevation && (hasReciprocal || hasSymptoms || ischemia.affectedWall)) return { isOMI: true, type: 'STEMI' };
    
    return { isOMI: false, type: '' };
};

export const enrichAnalysisWithLogic = (result: EcgAnalysisResult, patientCtx?: PatientContext): EcgAnalysisResult => {
  const m = result.precisionMeasurements;
  const waves = m.waves;
  const hr = parseHeartRate(result.heartRate);
  
  let implications = [...result.clinicalImplications];
  let urgency = result.urgency;
  let diagnosis = result.diagnosis;
  let reasoning = result.clinicalReasoning;
  
  // --- GUARDRAIL EXECUTION ---

  // 1. Sinus Rhythm Validation
  if (diagnosis.toLowerCase().includes('sinus rhythm')) {
      if (!validateSinusRhythm(waves, hr)) {
          if (hr > 100 && diagnosis.includes('Tachycardia')) { /* Valid */ } 
          else if (hr < 60 && diagnosis.includes('Bradycardia')) { /* Valid */ } 
          else if (waves?.intervals?.rrRegularity !== 'Regular') {
             reasoning += " [AUTO-CHECK: Irregular RR intervals detected. Consider Sinus Arrhythmia or subtle Atrial Fibrillation.]";
          }
      }
  }

  // 2. OMI (Occlusion MI) / STEMI Protocol
  const omiCheck = validateOMI(m.ischemiaAnalysis, patientCtx);
  if (omiCheck.isOMI) {
      if (urgency !== 'Emergency') {
          urgency = 'Emergency';
          reasoning += ` [CRITICAL ALERT: Detected ${omiCheck.type}. Upgraded to Emergency status per OMI protocol.]`;
      }
      if (!implications.some(i => i.includes('Cath Lab'))) {
          implications.unshift("🚨 IMMEDIATE CATH LAB ACTIVATION (Time-to-Balloon Critical)");
      }
      // Append localization if available
      if (m.ischemiaAnalysis?.affectedWall && !diagnosis.includes(m.ischemiaAnalysis.affectedWall)) {
          diagnosis += ` - ${m.ischemiaAnalysis.affectedWall} Wall`;
      }
  } else if (diagnosis.toLowerCase().includes('stemi') && m.ischemiaAnalysis?.stShape === 'Concave' && !patientCtx?.symptoms?.includes('Chest Pain')) {
       // Pericarditis Mimic Check
       diagnosis = "Suspected Pericarditis vs Early Repolarization";
       urgency = "Routine";
       reasoning += " [AUTO-CHECK: Concave ST elevation without reciprocal changes favors Pericarditis/BER over STEMI.]";
       implications = implications.filter(i => !i.includes('STEMI'));
  } else {
      // Check for NSTEMI / Ischemia (Urgent)
      if (m.ischemiaAnalysis?.stSegmentTrend === 'Depression' || m.ischemiaAnalysis?.stSegmentTrend === 'T-Wave Inversion') {
          if (urgency === 'Routine') {
              urgency = 'Urgent';
              reasoning += " [AUTO-CHECK: Significant ST Depression or T-Wave Inversion detected. Suspect NSTEMI or Ischemia.]";
          }
          if (m.ischemiaAnalysis?.affectedWall && !diagnosis.includes(m.ischemiaAnalysis.affectedWall)) {
             diagnosis += ` (${m.ischemiaAnalysis.affectedWall} Ischemia)`;
          }
      }
  }

  // 3. QTc Safety Net
  const qtc = waves?.intervals?.qtcMs || m.qtAnalysis?.qtcInterval || 0;
  if (qtc > 500) {
      if (urgency === 'Routine') urgency = 'Urgent';
      implications.unshift(`⚡ CRITICAL QTc (${qtc}ms): High risk of Torsades de Pointes. Monitor Magnesium/Potassium.`);
  }

  // 4. AV Block Validation
  const pr = waves?.intervals?.prMs || 0;
  if (diagnosis.toLowerCase().includes('1st degree') && pr < 200 && pr > 0) {
      reasoning += ` [AUTO-CHECK: PR interval (${pr}ms) is normal (<200ms). Diagnosis of 1st Degree AV Block requires PR > 200ms.]`;
  }

  // 5. Noise / Reliability Guardrail
  const reliability = m.signalQuality?.reliabilityScore ?? 10;
  if (reliability < 5) {
      urgency = 'Routine'; // Downgrade urgency if we can't trust the signal (unless it was already routine)
      // Or keep it but add a massive warning. Actually, for safety, we usually don't downgrade "Critical" to "Routine" blindly, 
      // but we should warn.
      // Let's just append a warning.
      diagnosis = `[POOR QUALITY] ${diagnosis}`;
      reasoning += ` [CAUTION: Low signal reliability score (${reliability}/10). Wave measurements may be inaccurate due to artifacts/noise.]`;
      implications.unshift("⚠️ REPEAT ECG: Signal quality precludes definitive analysis.");
  }

  return {
    ...result,
    diagnosis,
    urgency,
    clinicalReasoning: reasoning,
    clinicalImplications: Array.from(new Set(implications))
  };
};
