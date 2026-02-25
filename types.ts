
export interface PatientContext {
  age: string;
  gender: 'Male' | 'Female' | 'Other';
  ethnicity?: 'Caucasian' | 'Asian' | 'African' | 'Hispanic' | 'Other';
  symptoms: string[]; 
  history?: string; 
}

export interface SignalQualityMetrics {
  snrDb: number; // Signal-to-Noise Ratio estimate
  baselineWander: 'None' | 'Mild' | 'Severe';
  powerLineInterference: boolean; // 50/60Hz noise
  artifactsDetected: string[];
  reliabilityScore: number; // 0-100
}

export interface NeuralTelemetry {
  modelArchitecture: string; // e.g., "Hybrid EfficientNet-B7 + Temporal Transformer"
  processingTimeMs: number;
  attentionFocus: string[]; // Leads where the model focused most (simulated attention map)
  differentialDiagnoses: Array<{
    diagnosis: string;
    probability: number; // 0-100
    reasoning: string;
  }>;
  featureExtraction: {
    morphologicalFeatures: string[]; // Detected by CNN
    rhythmFeatures: string[]; // Detected by Transformer
  };
}

export interface WaveformFeatures {
  pWave: {
    present: boolean;
    morphology: 'Sinus' | 'Biphasic' | 'Peaked' | 'Notched' | 'Absent' | 'Flutter' | 'Fibrillatory';
    durationMs: number;
    amplitudeMv: number;
    axisDegrees?: number;
  };
  qrsComplex: {
    durationMs: number;
    amplitudeMv: number; // Max amplitude
    axisDegrees: number;
    morphologyV1: 'rS' | 'QS' | 'RSr' | 'Rs' | 'R' | 'Other';
    morphologyV6: 'qR' | 'Rs' | 'R' | 'Monophasic R' | 'Other';
    transitionZone: 'V1' | 'V2' | 'V3' | 'V4' | 'V5' | 'V6' | 'Clockwise' | 'Counter-Clockwise';
  };
  tWave: {
    morphology: 'Normal' | 'Peaked' | 'Inverted' | 'Biphasic' | 'Flat';
    axisDegrees?: number;
    symmetry: 'Symmetric' | 'Asymmetric';
  };
  intervals: {
    prMs: number;
    qtMs: number;
    qtcMs: number; // Fridericia
    rrRegularity: 'Regular' | 'Irregular' | 'Regularly Irregular';
  };
}

export interface IschemiaAnalysis {
  sgarbossaScore?: number;
  smithSgarbossaRatio?: number;
  wellensSyndrome: 'None' | 'Type A (Biphasic)' | 'Type B (Deep Inversion)';
  deWinterPattern: boolean;
  stSegmentTrend: 'Elevation' | 'Depression' | 'Neutral' | 'T-Wave Inversion';
  stSegmentDepression?: string;
  stShape?: 'Concave' | 'Convex' | 'Horizontal' | 'Downsloping' | 'Upsloping';
  affectedWall?: 'Anterior' | 'Inferior' | 'Lateral' | 'Septal' | 'Posterior' | 'Global' | 'Right Ventricle';
  reciprocalChangesFound: boolean;
  culpritArtery?: 'LAD' | 'RCA' | 'LCx' | 'Left Main' | 'Unknown';
}

export interface StructuralAnalysis {
  lvhDetected: boolean;
  lvhCriteria?: string;
  rvhDetected: boolean;
  atrialEnlargement: 'None' | 'Left (LAE)' | 'Right (RAE)' | 'Bi-atrial';
}

export interface ConductionAnalysis {
  blocks: string[];
  fascicularBlock: 'None' | 'LAFB' | 'LPFB' | 'Bifascicular' | 'Trifascicular';
  wpwPattern: boolean;
  ivcdType?: 'LBBB' | 'RBBB' | 'IVCD' | 'None';
}

// Unified Precision Measurements
export interface PrecisionMeasurements {
  signalQuality: SignalQualityMetrics;
  neuralTelemetry: NeuralTelemetry; // New field for AI architecture outputs
  waves: WaveformFeatures;
  
  // Specific Analyses
  ischemiaAnalysis?: IschemiaAnalysis;
  structuralAnalysis?: StructuralAnalysis;
  conductionAnalysis?: ConductionAnalysis;
  brugadaAnalysis?: any;
  pacemakerAnalysis?: any;
  
  // ECG Digitiser Integration
  digitizationMetrics?: DigitizationMetrics;

  // Backward compatibility helpers
  axis?: { qrsAxis: number; pAxis: number; tAxis: number; interpretation: string };
  pWave?: any; 
  qrsComplex?: any; 
  qtAnalysis?: any; 
}

export interface DigitizedBeat {
  lead: string;
  timeMs: number[]; // Relative time in ms (0 to ~1000ms)
  amplitudeMv: number[]; // Amplitude in mV
}

export interface DigitizationMetrics {
  method: string; // e.g., "Hough Transform + Deep Learning (Simulated)"
  gridDetection: 'Successful' | 'Failed' | 'Partial';
  segmentationConfidence: number; // 0-100
  representativeBeats: DigitizedBeat[]; // Extracted vector data for key leads
}

export interface HospitalGradeReport {
  diagnóstico_principal: string;
  confiança_principal: number;
  diagnósticos_diferenciais: Array<{
    condition: string;
    probability: number;
    severity: string;
  }>;
  regiões_críticas: { [lead: string]: number[] }; // Heatmap coordinates
  qualidade_sinal: number;
  alertas: string[];
  tempo_processamento: number;
}

export interface OptimizedReport {
  patient_id: string;
  ecg_date: string;
  acquisition_quality: 'adequate' | 'suboptimal' | 'poor';
  heart_rate: { value: number; unit: string; classification: string };
  rhythm: {
    primary: string;
    secondary: string[];
    regularity: string;
  };
  intervals: {
    PR_ms: number;
    QRS_ms: number;
    QT_ms: number;
    QTc_Bazett_ms: number;
    QTc_Fridericia_ms: number;
  };
  axis: {
    QRS_degrees: number;
    P_degrees: number;
    T_degrees: number;
    classification: string;
  };
  waveform_findings: string[];
  diagnoses: Array<{
    code: string;
    description: string;
    confidence: number;
    alert_level: 'CRITICAL' | 'URGENT' | 'ROUTINE';
    supporting_leads: string[];
    reciprocal_changes: string[];
  }>;
  drug_interactions_flagged: string[];
  comparison_with_prior_ecg: string;
  recommendations: string[];
  disclaimer: string;
}

export interface EcgAnalysisResult {
  id?: string;
  timestamp?: number;
  technicalQuality: { 
    overallScore: number;
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
  guidelineReferences: string[];
  regulatoryWarnings: string[];
  
  // New Hospital Grade Field
  hospitalGradeReport?: HospitalGradeReport;
  
  // Optimized Report (Etapa 9)
  optimizedReport?: OptimizedReport;
}

export interface AdjudicationData {
  status: 'pending' | 'approved' | 'rejected' | 'modified';
  user: string;
  timestamp: number;
  modifiedDiagnosis?: string;
  notes?: string;
}

export interface EcgRecord extends EcgAnalysisResult {
  id: string;
  timestamp: number;
  synced: boolean;
  adjudication?: AdjudicationData;
}

export enum AnalysisStatus {
  IDLE = 'IDLE',
  ANALYZING = 'ANALYZING',
  SUCCESS = 'SUCCESS',
  ERROR = 'ERROR'
}
