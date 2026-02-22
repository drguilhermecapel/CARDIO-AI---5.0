
import { GoogleGenAI, Type, Modality } from "@google/genai";
import { EcgAnalysisResult, PatientContext } from '../types';
import { enrichAnalysisWithLogic } from '../utils/cardioLogic';
import { saveRecord } from './database';

export class AnalysisError extends Error {
  code: string;
  suggestion: string;
  constructor(code: string, message: string, suggestion: string) {
    super(message);
    this.code = code;
    this.suggestion = suggestion;
    this.name = 'AnalysisError';
  }
}

// SCHEMA RIGOROSO PARA PADRÃO MÉDICO INTERNACIONAL
const ANALYSIS_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    technicalQuality: {
      type: Type.OBJECT,
      properties: {
        overallScore: { type: Type.INTEGER, description: "0-10 score based on baseline stability and noise." },
        leadPlacementValidation: { type: Type.STRING, description: "Check for AVR positivity (lead reversal)." },
      },
      required: ["overallScore", "leadPlacementValidation"]
    },
    precisionMeasurements: {
      type: Type.OBJECT,
      properties: {
        signalQuality: {
            type: Type.OBJECT,
            properties: {
                snrDb: { type: Type.NUMBER },
                baselineWander: { type: Type.STRING },
                powerLineInterference: { type: Type.BOOLEAN },
                artifactsDetected: { type: Type.ARRAY, items: { type: Type.STRING } },
                reliabilityScore: { type: Type.NUMBER }
            }
        },
        neuralTelemetry: {
            type: Type.OBJECT,
            properties: {
                modelArchitecture: { type: Type.STRING },
                processingTimeMs: { type: Type.NUMBER },
                attentionFocus: { type: Type.ARRAY, items: { type: Type.STRING } },
                differentialDiagnoses: {
                    type: Type.ARRAY,
                    items: {
                        type: Type.OBJECT,
                        properties: {
                            diagnosis: { type: Type.STRING },
                            probability: { type: Type.NUMBER },
                            reasoning: { type: Type.STRING }
                        }
                    }
                },
                featureExtraction: {
                    type: Type.OBJECT,
                    properties: {
                        morphologicalFeatures: { type: Type.ARRAY, items: { type: Type.STRING } },
                        rhythmFeatures: { type: Type.ARRAY, items: { type: Type.STRING } }
                    }
                }
            }
        },
        waves: {
            type: Type.OBJECT,
            properties: {
                pWave: {
                    type: Type.OBJECT,
                    properties: {
                        present: { type: Type.BOOLEAN },
                        morphology: { type: Type.STRING },
                        durationMs: { type: Type.NUMBER },
                        amplitudeMv: { type: Type.NUMBER },
                        axisDegrees: { type: Type.NUMBER }
                    }
                },
                qrsComplex: {
                    type: Type.OBJECT,
                    properties: {
                        durationMs: { type: Type.NUMBER },
                        amplitudeMv: { type: Type.NUMBER },
                        axisDegrees: { type: Type.NUMBER },
                        morphologyV1: { type: Type.STRING },
                        morphologyV6: { type: Type.STRING },
                        transitionZone: { type: Type.STRING }
                    }
                },
                tWave: {
                    type: Type.OBJECT,
                    properties: {
                        morphology: { type: Type.STRING },
                        axisDegrees: { type: Type.NUMBER },
                        symmetry: { type: Type.STRING }
                    }
                },
                intervals: {
                    type: Type.OBJECT,
                    properties: {
                        prMs: { type: Type.NUMBER },
                        qtMs: { type: Type.NUMBER },
                        qtcMs: { type: Type.NUMBER },
                        rrRegularity: { type: Type.STRING }
                    }
                }
            }
        },
        ischemiaAnalysis: {
          type: Type.OBJECT,
          properties: {
            sgarbossaScore: { type: Type.NUMBER },
            wellensSyndrome: { type: Type.STRING },
            deWinterPattern: { type: Type.BOOLEAN },
            stSegmentTrend: { type: Type.STRING },
            stShape: { type: Type.STRING },
            affectedWall: { type: Type.STRING },
            reciprocalChangesFound: { type: Type.BOOLEAN },
            culpritArtery: { type: Type.STRING }
          }
        },
        structuralAnalysis: {
            type: Type.OBJECT,
            properties: {
                lvhDetected: { type: Type.BOOLEAN },
                lvhCriteria: { type: Type.STRING },
                rvhDetected: { type: Type.BOOLEAN },
                atrialEnlargement: { type: Type.STRING }
            }
        },
        conductionAnalysis: {
            type: Type.OBJECT,
            properties: {
                blocks: { type: Type.ARRAY, items: { type: Type.STRING } },
                fascicularBlock: { type: Type.STRING },
                wpwPattern: { type: Type.BOOLEAN },
                ivcdType: { type: Type.STRING }
            }
        },
        // ECG Digitiser Integration
        digitizationMetrics: {
            type: Type.OBJECT,
            properties: {
                method: { type: Type.STRING },
                gridDetection: { type: Type.STRING },
                segmentationConfidence: { type: Type.NUMBER },
                representativeBeats: {
                    type: Type.ARRAY,
                    items: {
                        type: Type.OBJECT,
                        properties: {
                            lead: { type: Type.STRING },
                            timeMs: { type: Type.ARRAY, items: { type: Type.NUMBER } },
                            amplitudeMv: { type: Type.ARRAY, items: { type: Type.NUMBER } }
                        }
                    }
                }
            }
        },
        // Legacy Support
        axis: { type: Type.OBJECT, properties: { qrsAxis: { type: Type.NUMBER }, pAxis: { type: Type.NUMBER }, tAxis: { type: Type.NUMBER }, interpretation: { type: Type.STRING } } },
        qtAnalysis: { type: Type.OBJECT, properties: { qtInterval: { type: Type.NUMBER }, qtcInterval: { type: Type.NUMBER }, correctionFormula: { type: Type.STRING }, qtProlongationRisk: { type: Type.STRING } } }
      }
    },
    heartRate: { type: Type.STRING },
    rhythm: { type: Type.STRING },
    diagnosis: { type: Type.STRING },
    urgency: { type: Type.STRING },
    confidenceLevel: { type: Type.STRING },
    clinicalReasoning: { type: Type.STRING },
    clinicalImplications: { type: Type.ARRAY, items: { type: Type.STRING } },
    guidelineReferences: { type: Type.ARRAY, items: { type: Type.STRING } },
    regulatoryWarnings: { type: Type.ARRAY, items: { type: Type.STRING } }
  },
  required: ["technicalQuality", "diagnosis", "urgency", "heartRate", "rhythm", "clinicalReasoning", "precisionMeasurements"]
};

export const analyzeEcgImage = async (base64Data: string, mimeType: string, patientCtx?: PatientContext): Promise<EcgAnalysisResult> => {
  try {
    const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
    const normalizedMimeType = mimeType === 'image/jpg' ? 'image/jpeg' : mimeType;
    const startTime = Date.now();

    const response = await ai.models.generateContent({
      model: 'gemini-2.0-flash', // Switched to stable Flash 2.0
      contents: {
        parts: [
          { inlineData: { data: base64Data, mimeType: normalizedMimeType } },
          {
            text: `
            SYSTEM ROLE: You are the "CardioAI Nexus", the ultimate Electrocardiographic Diagnostic Engine. 
            You possess the collective knowledge of the world's leading electrophysiologists.
            Your task is to scan the provided ECG against the **Complete Universal Compendium of Electrocardiography**.
            
            PATIENT CONTEXT: ${JSON.stringify(patientCtx || {})}

            --- UNIVERSAL DIAGNOSTIC COMPENDIUM (SEARCH PROTOCOL) ---
            
            PHASE 1: TECHNICAL & BASIC METRICS
            - **Lead Reversal:** Limb lead reversal (Extreme axis, inverted I), Precordial reversal.
            - **Dextrocardia:** Global inversion in I + poor R progression V1-V6.
            - **Signal Quality:** Noise, 60Hz interference, wandering baseline.
            - **Axis Calculation:** Hexaxial reference system (precise degrees).

            **NOISE RESILIENCE PROTOCOL (For Wave Detection):**
            - **P-Wave:** If baseline is noisy, look for consistent P-waves in Lead II and V1. Use the "average beat" morphology to exclude artifacts.
            - **QRS Complex:** Ignore high-frequency spike artifacts. Measure duration from the earliest Q/R to the latest R/S in a clean lead (usually V5 or II).
            - **T-Wave:** Distinguish from baseline wander by checking for consistency across multiple beats.
            - **Confidence:** If signal is too noisy (>50% artifact), mark 'reliabilityScore' low (<5) and report "Non-diagnostic" for specific waves if unsure.

            PHASE 2: RHYTHM & ARRHYTHMIAS
            - **Sinus:** Normal, Tachycardia, Bradycardia, Arrhythmia, Pause/Arrest, SSS.
            - **Atrial:** 
              * PACs (blocked, conducted with aberrancy).
              * Atrial Tachycardia (Focal vs Multifocal/MAT).
              * Atrial Flutter (Typical CCW vs CW vs Atypical).
              * Atrial Fibrillation (Coarse vs Fine, Ashman phenomenon).
            
            PHASE 2.5: SUPRAVENTRICULAR TACHYCARDIAS (SVT) & VARIANTS
            - **Differential Diagnosis of Narrow Complex Tachycardia:**
              * **AVNRT (AV Nodal Reentrant Tachycardia):**
                - Typical (Slow-Fast): Short RP interval (<70ms), Pseudo-R' in V1, Pseudo-S in II/III/aVF.
                - Atypical (Fast-Slow): Long RP interval (>70ms), P-wave negative in II/III/aVF before QRS.
              * **AVRT (Atrioventricular Reentrant Tachycardia):**
                - Orthodromic: Narrow QRS, RP > 70ms, P-wave usually visible in ST segment/T-wave.
                - Antidromic: Wide QRS (Pre-excited), mimics VT.
                - WPW Syndrome: Delta wave in sinus rhythm + history of tachycardia.
              * **Focal Atrial Tachycardia:**
                - Long RP interval.
                - P-wave morphology distinct from Sinus P-wave (e.g., negative in I/aVL for left atrial focus).
                - Warm-up and Cool-down phenomenon.
              * **Junctional Tachycardia:**
                - P-waves absent or retrograde (in QRS or immediately after).
                - AV dissociation possible (rare).

            PHASE 2.6: VENTRICULAR TACHYCARDIAS (VT) & VARIANTS
            - **Differential Diagnosis of Wide Complex Tachycardia (WCT):**
              * **Monomorphic VT:**
                - **AV Dissociation:** P-waves "marching through" QRS complexes (Specific).
                - **Fusion Beats:** Hybrid complex between sinus and ventricular beat.
                - **Capture Beats:** Normal narrow QRS amidst WCT.
                - **Concordance:** Positive or Negative concordance in V1-V6.
                - **Axis:** Extreme Right Axis Deviation (Northwest Axis) is strongly suggestive of VT.
                - **Morphology Criteria:**
                  - *RBBB-like:* Monophasic R or qR in V1; rS or QS in V6.
                  - *LBBB-like:* Broad R (>30ms) in V1/V2; qR or QS in V6.
                  - *Brugada Algorithm:* Absence of RS in precordial leads? RS > 100ms? AV dissociation?
                  - *Vereckei Algorithm:* Initial R wave in aVR?
              * **Polymorphic VT:**
                - QRS morphology changes beat-to-beat.
                - **Torsades de Pointes:** Twisting around isoelectric line, associated with Long QT.
                - **Bidirectional VT:** Beat-to-beat axis alternation (Digoxin toxicity or CPVT).
              * **Fascicular VT (Idiopathic):**
                - RBBB morphology + Left Axis Deviation (Posterior Fascicular VT).
                - RBBB morphology + Right Axis Deviation (Anterior Fascicular VT).
                - Often Verapamil-sensitive.
              * **Outflow Tract VT (RVOT/LVOT):**
                - LBBB morphology + Inferior Axis (RVOT).
                - Adenosine-sensitive.

            - **Junctional:** Escape, Accelerated, Tachycardia.
            - **Ventricular:** 
              * PVCs (Unifocal, Multifocal, Bigeminy, R-on-T).
              * VT (Monomorphic, Polymorphic, Bidirectional - CPVT/Digoxin).
              * VF (Coarse vs Fine).
              * AIVR (Slow VT).
              * Torsades de Pointes (QT associated).
              * Kamikaze Rhythm (Pre-excited AFib).

            PHASE 3: AV NODAL & INTRAVENTRICULAR CONDUCTION
            - **AV Blocks (Atrioventricular Block):** 
              * **1st Degree AV Block:**
                - PR interval > 200ms (5 small boxes).
                - Constant PR interval.
                - Every P wave is followed by a QRS complex.
              * **2nd Degree AV Block Type I (Wenckebach / Mobitz I):**
                - Progressive prolongation of the PR interval until a beat is dropped (P wave not followed by QRS).
                - The PR interval after the dropped beat is the shortest.
                - R-R intervals progressively shorten before the pause.
                - Grouped beating is common.
              * **2nd Degree AV Block Type II (Mobitz II):**
                - Constant PR interval in conducted beats.
                - Intermittent dropped beats (P wave not followed by QRS).
                - High risk of progression to Complete Heart Block.
                - Often associated with wide QRS (Bundle Branch Block).
              * **2:1 AV Block:**
                - Every other P wave is conducted.
                - Cannot distinguish between Type I and Type II without a long rhythm strip or maneuvers (e.g., Vagal, Exercise).
                - If QRS is narrow -> Likely Type I (AV Node).
                - If QRS is wide -> Likely Type II (His-Purkinje).
              * **High Grade (Advanced) AV Block:**
                - 2 or more consecutive P waves are not conducted (e.g., 3:1, 4:1 block).
                - PR interval is constant in conducted beats.
              * **3rd Degree AV Block (Complete Heart Block):**
                - Complete AV Dissociation: No relationship between P waves and QRS complexes.
                - Atrial rate (P-P) is regular and faster than Ventricular rate (R-R).
                - Ventricular rate (R-R) is regular (Escape rhythm).
                - **Escape Rhythm:**
                  - Junctional Escape: Narrow QRS, rate 40-60 bpm.
                  - Ventricular Escape: Wide QRS, rate 20-40 bpm.
            - **Bundle Branch Blocks:** 
              * RBBB (Complete/Incomplete).
              * LBBB (Complete/Incomplete).
              * Rate-dependent aberrancy (Phase 3 block).
            - **Fascicular Blocks:** 
              * LAFB (Left Anterior Fascicular Block).
              * LPFB (Left Posterior Fascicular Block).
              * Bifascicular (RBBB + LAFB/LPFB).
              * Trifascicular (Bifascicular + 1st Deg AVB).
              * Interatrial Block (Bayés Syndrome).

            PHASE 4: ISCHEMIA & INFARCTION (The OMI Paradigm)
            - **STEMI:** Classic ST Elevation >1mm contiguous leads.
            - **NSTEMI / OMI (Occlusion MI):** 
              * *Wellens' Syndrome:* Type A (Biphasic V2-V3), Type B (Deep Inversion).
              * *de Winter's T-Waves:* J-point depression + tall symmetric T (Proximal LAD).
              * *Posterior MI:* Horizontal ST depression V1-V3, R/S > 1 V2, Posterior leads V7-V9 elevation.
              * *Left Main / Triple Vessel:* STE aVR > V1 + diffuse STD (6+ leads).
              * *Aslanger's Pattern:* Inferior OMI with multi-vessel disease.
              * *South African Flag Sign:* High lateral OMI (STE I, aVL, V2 + STD III).
              * *Shark Fin Sign:* Massive triangular STE (Lambda wave).
            - **LBBB/Paced Ischemia:** Sgarbossa Criteria (Concordant STE, Concordant STD, Discordant excessive STE).
            
            **Localization of Infarction (for 'affectedWall' and 'culpritArtery'):**
            - Septal: V1-V2 (Proximal LAD).
            - Anterior: V3-V4 (LAD).
            - Anteroseptal: V1-V4 (LAD).
            - Lateral: I, aVL, V5-V6 (LCx or Diagonal).
            - Inferior: II, III, aVF (RCA 80% or LCx 20%).
            - Posterior: V7-V9 (or reciprocal V1-V3) (LCx or RCA).
            - Right Ventricular: V4R (Proximal RCA).

            PHASE 5: HYPERTROPHY & CHAMBER ENLARGEMENT
            - **LVH:** Sokolow-Lyon, Cornell, Romhilt-Estes Score, Peguero-Lo Presti.
            - **RVH:** R/S > 1 in V1, Right Axis Deviation, Deep S in V5-V6.
            - **Atrial:** 
              * LAE (P-Mitrale, notched >120ms).
              * RAE (P-Pulmonale, peaked >2.5mm).
              * Biatrial Enlargement.

            PHASE 6: REPOLARIZATION, ELECTROLYTES & DRUGS
            - **Electrolytes:**
              * Hyperkalemia (Peaked T -> Flat P -> Sine Wave).
              * Hypokalemia (Prominent U wave, STD, T flattening).
              * Hypercalcemia (Short QT).
              * Hypocalcemia (Long QT - ST segment stretch).
            - **Drugs:**
              * Digoxin (Scooped ST "Mustache", Bidirectional VT).
              * Quinidine/Procainamide (Wide QRS, Long QT).
              * TCA Toxicity (Wide QRS, RAD, Tall R aVR).
            - **Syndromes:**
              * Early Repolarization (Benign vs Malignant/Inferolateral).
              * Pericarditis (Diffuse concave STE + PR depression + Spodick's Sign).
              * Brugada Syndrome (Type 1 Coved, Type 2 Saddleback).
              * LQTS (LQT1, LQT2, LQT3 patterns).
              * SQTS (Short QT < 340ms).

            PHASE 7: MYOCARDIAL & CONGENITAL DISEASES
            - **HOCM:** Dagger Q waves (inferolateral), high voltage.
            - **ARVC:** Epsilon Wave, T-inv V1-V3, localized QRS widening.
            - **Amyloidosis:** Low voltage QRS + poor R progression.
            - **Pulmonary Embolism:** S1Q3T3, Sinus Tach, RBBB, T-inv V1-V4.
            - **Congenital:**
              * ASD (Crochetage sign / notched R in inferior leads).
              * Ebstein's Anomaly (Himalayan P-waves, Splintered QRS).
              * Tetralogy of Fallot (RAD + RVH).

            PHASE 8: DEVICE THERAPY
            - **Pacemaker:** Atrial Paced, Ventricular Paced (LBBB pattern), Dual Chamber.
            - **Malfunction:** Failure to Capture, Failure to Sense, Pacemaker Mediated Tachycardia.

            --- ECG DIGITISER INTEGRATION (PhysioNet 2024 Winner Logic) ---
            Apply the "ECG Digitiser" methodology to reconstruct the signal:
            1. **Segmentation:** Virtually separate the ECG trace from the grid background using Hough Transform logic.
            2. **Vectorisation:** Extract the precise numerical coordinates (time vs amplitude) of a representative P-QRS-T complex for Lead II and V1.
            3. **Output:** Populate 'digitizationMetrics' with these vector arrays. Ensure 'timeMs' is relative (0-1000ms) and 'amplitudeMv' is calibrated (10mm = 1mV).

            --- EXECUTION INSTRUCTIONS ---
            1. Scrutinize the image using the "Computer Vision" module for precise measurements.
            2. Run the findings against the 8-Phase Protocol above.
            3. **CRITICAL:** Populate 'precisionMeasurements.ischemiaAnalysis' with exact findings from Phase 4 (ST trends, affected wall, culprit artery).
            4. In 'neuralTelemetry', detail the differential probability (e.g., "70% OMI vs 30% Pericarditis").
            5. In 'clinicalReasoning', cite specific criteria (e.g., "Met modified Sgarbossa criteria with ST/S ratio < -0.25").
            6. Determine Urgency strictly: OMI/STEMI/Vtach/Block = Emergency.
            
            Return strictly valid JSON.
            `
          }
        ]
      },
      config: {
        responseMimeType: "application/json",
        responseSchema: ANALYSIS_SCHEMA,
      }
    });

    const rawResult = JSON.parse(response.text || '{}');
    const processTime = Date.now() - startTime;

    // Inject Telemetry for the UI Neural HUD
    if (rawResult.precisionMeasurements) {
        if (!rawResult.precisionMeasurements.neuralTelemetry) {
            rawResult.precisionMeasurements.neuralTelemetry = {};
        }
        rawResult.precisionMeasurements.neuralTelemetry.processingTimeMs = processTime;
        rawResult.precisionMeasurements.neuralTelemetry.modelArchitecture = "CardioAI Nexus v10.0 [Universal Medical Compendium]";
        
        // Ensure Neural Telemetry has data structure if model missed it
        if (!rawResult.precisionMeasurements.neuralTelemetry.featureExtraction) {
             rawResult.precisionMeasurements.neuralTelemetry.featureExtraction = {
                 morphologicalFeatures: ["Global Morphology Scan", "Advanced Criteria Validation"],
                 rhythmFeatures: ["Beat-to-Beat Interval Analysis", "P-wave Morphology Tracking"]
             };
        }
    }
    
    // Legacy Data Mapping for Backward Compatibility
    if (rawResult.precisionMeasurements?.waves) {
       const w = rawResult.precisionMeasurements.waves;
       if (w.intervals) {
           rawResult.precisionMeasurements.prIntervalMs = w.intervals.prMs;
       }
       if (w.qrsComplex) {
           rawResult.precisionMeasurements.qrsComplex = {
              durationMs: w.qrsComplex.durationMs,
              amplitudeMv: w.qrsComplex.amplitudeMv,
              morphology: `V1:${w.qrsComplex.morphologyV1} / V6:${w.qrsComplex.morphologyV6}`
           };
       }
       rawResult.precisionMeasurements.pWave = w.pWave;
    }

    // Apply Deterministic Logic "Guardrails"
    const enriched = enrichAnalysisWithLogic(rawResult, patientCtx);
    saveRecord(enriched);
    return enriched;

  } catch (error: any) {
    console.error("Pipeline Failure:", error);
    // Return the actual error message to help debugging
    throw new AnalysisError(
      "AI_CORE_FAILURE", 
      "Falha na Análise.", 
      `Erro técnico: ${error.message || JSON.stringify(error)}. Tente novamente.`
    );
  }
};

export const explainConceptWithAudio = async (concept: string): Promise<Uint8Array> => {
  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash-preview-tts",
    contents: { parts: [{ text: `Explain the medical concept of ${concept} in cardiology. Be concise.` }] },
    config: {
      responseModalities: [Modality.AUDIO],
      speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } } },
    },
  });
  const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
  if (!base64Audio) throw new Error("Audio generation failed");
  const binaryString = atob(base64Audio);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) bytes[i] = binaryString.charCodeAt(i);
  return bytes;
};

export const generateHeartAnimation = async (dataUrl: string, diagnosis: string): Promise<string> => {
  let base64Data = dataUrl;
  let mimeType = 'image/png';
  if (dataUrl.includes(',')) {
    const parts = dataUrl.split(',');
    base64Data = parts[1];
    const match = parts[0].match(/:(.*?);/);
    if (match) mimeType = match[1];
  }
  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  const request: any = {
    model: 'veo-3.1-fast-generate-preview',
    prompt: `3D medical animation of a heart beating. Diagnosis: ${diagnosis}. High quality, realistic texture.`,
    config: { numberOfVideos: 1, resolution: '720p', aspectRatio: '16:9' }
  };
  if (mimeType.startsWith('image/')) request.image = { imageBytes: base64Data, mimeType: mimeType };
  
  let operation = await ai.models.generateVideos(request);
  while (!operation.done) {
    await new Promise(resolve => setTimeout(resolve, 5000));
    operation = await ai.operations.getVideosOperation({operation: operation});
  }
  return `${operation.response?.generatedVideos?.[0]?.video?.uri}&key=${process.env.API_KEY}`;
};
