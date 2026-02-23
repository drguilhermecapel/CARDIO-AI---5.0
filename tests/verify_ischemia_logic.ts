import { enrichAnalysisWithLogic } from '../utils/cardioLogic';
import { EcgAnalysisResult, PatientContext } from '../types';

const mockResult: EcgAnalysisResult = {
    heartRate: "80 bpm",
    rhythm: "Sinus Rhythm",
    diagnosis: "Normal Sinus Rhythm",
    clinicalReasoning: "Normal ECG",
    urgency: "Routine",
    clinicalImplications: [],
    confidenceLevel: "High",
    guidelineReferences: [],
    regulatoryWarnings: [],
    technicalQuality: { overallScore: 10, leadPlacementValidation: "Correct" },
    precisionMeasurements: {
        signalQuality: { snrDb: 20, baselineWander: 'None', powerLineInterference: false, artifactsDetected: [], reliabilityScore: 10 },
        neuralTelemetry: { modelArchitecture: "Test", processingTimeMs: 100, attentionFocus: [], differentialDiagnoses: [], featureExtraction: { morphologicalFeatures: [], rhythmFeatures: [] } },
        waves: {
            pWave: { present: true, morphology: 'Sinus', durationMs: 100, amplitudeMv: 0.1 },
            qrsComplex: { durationMs: 80, amplitudeMv: 1.0, axisDegrees: 45, morphologyV1: 'rS', morphologyV6: 'qR', transitionZone: 'V3' },
            tWave: { morphology: 'Normal', symmetry: 'Symmetric' },
            intervals: { prMs: 160, qtMs: 360, qtcMs: 400, rrRegularity: 'Regular' }
        },
        ischemiaAnalysis: {
            wellensSyndrome: 'None',
            deWinterPattern: false,
            stSegmentTrend: 'Elevation',
            stShape: 'Convex',
            affectedWall: 'Anterior',
            reciprocalChangesFound: true,
            culpritArtery: 'LAD'
        }
    }
};

const patientCtx: PatientContext = {
    age: "55",
    gender: "Male",
    symptoms: ["Chest Pain"],
    history: "Hypertension"
};

console.log("Running Ischemia Logic Verification...");

// Test 1: STEMI Detection
const stemiResult = enrichAnalysisWithLogic({ ...mockResult }, patientCtx);
if (stemiResult.diagnosis.includes("STEMI") && stemiResult.urgency === "Emergency") {
    console.log("✅ Test 1 Passed: STEMI correctly identified and upgraded to Emergency.");
} else {
    console.error("❌ Test 1 Failed: STEMI not identified correctly.", stemiResult.diagnosis, stemiResult.urgency);
}

// Test 2: NSTEMI Detection (Depression)
const nstemiMock = JSON.parse(JSON.stringify(mockResult));
nstemiMock.precisionMeasurements.ischemiaAnalysis = {
    stSegmentTrend: 'Depression',
    stSegmentDepression: 'Horizontal',
    affectedWall: 'Lateral',
    reciprocalChangesFound: false
};
nstemiMock.diagnosis = "Sinus Rhythm";
nstemiMock.urgency = "Routine";

const nstemiResult = enrichAnalysisWithLogic(nstemiMock, patientCtx);
if (nstemiResult.urgency === "Urgent" && nstemiResult.clinicalReasoning.includes("NSTEMI")) {
    console.log("✅ Test 2 Passed: NSTEMI correctly identified and upgraded to Urgent.");
} else {
    console.error("❌ Test 2 Failed: NSTEMI not identified correctly.", nstemiResult.urgency, nstemiResult.clinicalReasoning);
}

// Test 3: Wellens Syndrome
const wellensMock = JSON.parse(JSON.stringify(mockResult));
wellensMock.precisionMeasurements.ischemiaAnalysis = {
    wellensSyndrome: 'Type A (Biphasic)',
    stSegmentTrend: 'Neutral'
};
wellensMock.diagnosis = "Sinus Rhythm";
wellensMock.urgency = "Routine";

const wellensResult = enrichAnalysisWithLogic(wellensMock, patientCtx);
if (wellensResult.urgency === "Emergency" && wellensResult.clinicalReasoning.includes("Wellens")) {
    console.log("✅ Test 3 Passed: Wellens Syndrome correctly identified.");
} else {
    console.error("❌ Test 3 Failed: Wellens not identified.", wellensResult.urgency, wellensResult.clinicalReasoning);
}
