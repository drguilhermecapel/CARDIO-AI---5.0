
export interface PatientContext {
  age: string;
  gender: 'Male' | 'Female' | 'Other';
  symptoms: string[]; 
  history?: string; 
}

export interface BaranchukAnalysis {
  pWaveDurationMs: number;
  iabType: 'None' | 'Partial' | 'Advanced (Bayes Syndrome)';
  pWaveMorphologyInferior: 'Normal' | 'Notched' | 'Biphasic (+/-)';
  afibRiskScore: 'Low' | 'Moderate' | 'High';
}

export interface ArvdAnalysis {
  epsilonWaveDetected: boolean;
  tWaveInversionV1V3: boolean;
  terminalActivationDelayMs: number;
  localizedQrsWidening: boolean;
  rvPrecordialEctopy: boolean;
}

export interface IschemiaAnalysis {
  sgarbossaScore?: number;
  smithSgarbossaRatio?: number;
  wellensSyndrome: 'None' | 'Type A (Biphasic)' | 'Type B (Deep Inversion)';
  deWinterPattern: boolean;
  stSegmentTrend: 'Elevation' | 'Depression' | 'Neutral';
  affectedWall?: 'Anterior' | 'Inferior' | 'Lateral' | 'Septal' | 'Posterior' | 'Global';
}

export interface PacemakerAnalysis {
  pacingMode: 'AAI' | 'VVI' | 'DDD' | 'VDD' | 'CRT-P/D' | 'None';
  pacingSite: 'Atrial' | 'Ventricular (RV)' | 'Ventricular (LV)' | 'Biventricular' | 'Dual Chamber' | 'None';
  captureIntegrity: 'Stable' | 'Failure to Capture' | 'Failure to Sense' | 'Oversensing' | 'Inconclusive';
  spikeAmplitude: 'Micro' | 'Prominent' | 'Bipolar (Low)';
  atrioventricularIntervalMs?: number;
}

export interface SndAnalysis {
  condition: 'Sinus Bradycardia' | 'Sinus Pause' | 'Sinoatrial Block' | 'Chronotropic Incompetence' | 'Tachy-Brady Syndrome' | 'Isorhythmic Dissociation' | 'None';
  pauseDurationMs?: number;
  bsaType?: 'Type I' | 'Type II';
  chronotropicIndex?: number;
  dissociationDetails?: string;
  pacemakerIndication: 'Class I' | 'Class IIa' | 'Class IIb' | 'Class III' | 'Unknown';
}

export interface PrecisionMeasurements {
  pWave: { 
    amplitudeMv: number; 
    durationMs: number;
    morphology?: 'Normal' | 'Biphasic (+/-)' | 'Notched' | 'Inverted' | 'Retrograde';
  };
  prIntervalMs: number;
  qrsComplex: { 
    amplitudeMv: number; 
    durationMs: number;
  };
  qtIntervalMs: number;
  qtcIntervalMs: number;
  stDeviationMv: number;
  pacemakerAnalysis?: PacemakerAnalysis;
  sndAnalysis?: SndAnalysis;
  arvdAnalysis?: ArvdAnalysis;
  baranchukAnalysis?: BaranchukAnalysis;
  ischemiaAnalysis?: IschemiaAnalysis;
}

export interface EcgAnalysisResult {
  id?: string;
  timestamp?: number;
  technicalQuality: {
    overallScore: number;
    calibrationFound: boolean;
    isInterpretabilityLimited: boolean;
    leadPlacementValidation: string;
  };
  heartRate: string;
  rhythm: string;
  diagnosis: string;
  clinicalReasoning: string;
  urgency: 'Emergency' | 'Urgent' | 'Routine' | 'Low';
  clinicalImplications: string[];
  precisionMeasurements: PrecisionMeasurements;
  confidenceLevel: 'Low' | 'Medium' | 'High';
}

export interface EcgRecord extends EcgAnalysisResult {
  id: string;
  timestamp: number;
  synced: boolean;
}

export enum AnalysisStatus {
  IDLE = 'IDLE',
  ANALYZING = 'ANALYZING',
  SUCCESS = 'SUCCESS',
  ERROR = 'ERROR'
}
