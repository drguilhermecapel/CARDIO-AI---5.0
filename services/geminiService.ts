
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

const ANALYSIS_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    technicalQuality: {
      type: Type.OBJECT,
      properties: {
        overallScore: { type: Type.INTEGER },
        calibrationFound: { type: Type.BOOLEAN },
        isInterpretabilityLimited: { type: Type.BOOLEAN },
        leadPlacementValidation: { type: Type.STRING }
      },
      required: ["overallScore", "calibrationFound", "leadPlacementValidation"]
    },
    precisionMeasurements: {
      type: Type.OBJECT,
      properties: {
        ischemiaAnalysis: {
          type: Type.OBJECT,
          properties: {
            sgarbossaScore: { type: Type.NUMBER },
            smithSgarbossaRatio: { type: Type.NUMBER },
            wellensSyndrome: { type: Type.STRING, enum: ["None", "Type A (Biphasic)", "Type B (Deep Inversion)"] },
            deWinterPattern: { type: Type.BOOLEAN },
            stSegmentTrend: { type: Type.STRING, enum: ["Elevation", "Depression", "Neutral"] },
            affectedWall: { type: Type.STRING, enum: ["Anterior", "Inferior", "Lateral", "Septal", "Posterior", "Global"] }
          }
        },
        pacemakerAnalysis: {
          type: Type.OBJECT,
          properties: {
            pacingMode: { type: Type.STRING, enum: ["AAI", "VVI", "DDD", "VDD", "CRT-P/D", "None"] },
            pacingSite: { type: Type.STRING, enum: ["Atrial", "Ventricular (RV)", "Ventricular (LV)", "Biventricular", "Dual Chamber", "None"] },
            captureIntegrity: { type: Type.STRING, enum: ["Stable", "Failure to Capture", "Failure to Sense", "Oversensing", "Inconclusive"] },
            spikeAmplitude: { type: Type.STRING, enum: ["Micro", "Prominent", "Bipolar (Low)"] },
            atrioventricularIntervalMs: { type: Type.NUMBER }
          }
        },
        baranchukAnalysis: {
          type: Type.OBJECT,
          properties: {
            pWaveDurationMs: { type: Type.NUMBER },
            iabType: { type: Type.STRING, enum: ["None", "Partial", "Advanced (Bayes Syndrome)"] },
            pWaveMorphologyInferior: { type: Type.STRING, enum: ["Normal", "Notched", "Biphasic (+/-)"] },
            afibRiskScore: { type: Type.STRING, enum: ["Low", "Moderate", "High"] }
          }
        },
        arvdAnalysis: {
          type: Type.OBJECT,
          properties: {
            epsilonWaveDetected: { type: Type.BOOLEAN },
            tWaveInversionV1V3: { type: Type.BOOLEAN },
            terminalActivationDelayMs: { type: Type.NUMBER }
          }
        },
        qrsComplex: {
          type: Type.OBJECT,
          properties: { durationMs: { type: Type.NUMBER } }
        },
        prIntervalMs: { type: Type.NUMBER },
        qtIntervalMs: { type: Type.NUMBER },
        qtcIntervalMs: { type: Type.NUMBER }
      }
    },
    heartRate: { type: Type.STRING },
    rhythm: { type: Type.STRING },
    diagnosis: { type: Type.STRING },
    urgency: { type: Type.STRING, enum: ["Emergency", "Urgent", "Routine", "Low"] },
    confidenceLevel: { type: Type.STRING, enum: ["Low", "Medium", "High"] },
    clinicalReasoning: { type: Type.STRING },
    clinicalImplications: {
      type: Type.ARRAY,
      items: { type: Type.STRING }
    }
  },
  required: ["technicalQuality", "diagnosis", "urgency", "confidenceLevel", "heartRate", "rhythm", "clinicalReasoning", "clinicalImplications"]
};

export const analyzeEcgImage = async (base64Data: string, mimeType: string, patientCtx?: PatientContext): Promise<EcgAnalysisResult> => {
  try {
    const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
    const normalizedMimeType = mimeType === 'image/jpg' ? 'image/jpeg' : mimeType;

    const response = await ai.models.generateContent({
      model: 'gemini-3-pro-preview',
      contents: {
        parts: [
          { inlineData: { data: base64Data, mimeType: normalizedMimeType } },
          {
            text: `ATUE COMO UM CARDIOLOGISTA ESPECIALISTA EM ELETROFISIOLOGIA E OMI (INFARTO COM OCLUSÃO):
            
            Analise este ECG em busca de padrões de alto risco, mesmo que sutis:
            
            1. ISQUEMIA EM RITMO DE MARCAPASSO/BRE: Use os Critérios de Sgarbossa (Elevação concordante ≥ 1mm, Depressão concordante V1-V3 ≥ 1mm) e Smith-Sgarbossa (razão ST/S < -0.25).
            2. EQUIVALENTES DE STEMI:
               - Wellens (T bifásica ou inversão profunda em V2-V3);
               - de Winter (Depressão do ponto J com T alta e simétrica em precordiais);
               - Infarto de Parede Posterior (R alta e depressão de ST em V1-V3).
            3. ARRRITMIAS E BLOQUEIOS:
               - Algoritmo de Baranchuk para IAB;
               - Task Force para ARVD (Onda Épsilon);
               - Brugada (Tipos 1, 2, 3).
            
            Contexto: ${patientCtx ? JSON.stringify(patientCtx) : 'Sem contexto'}.
            Priorize a detecção de Infarto com Oclusão (OMI).`
          }
        ]
      },
      config: {
        responseMimeType: "application/json",
        responseSchema: ANALYSIS_SCHEMA,
        thinkingConfig: { thinkingBudget: 32768 }
      }
    });

    const rawResult = JSON.parse(response.text || '{}');
    const enriched = enrichAnalysisWithLogic(rawResult, patientCtx);
    saveRecord(enriched);
    return enriched;
  } catch (error: any) {
    throw new AnalysisError(
      "AI_DECODE_FAILURE", 
      "Erro no processamento neural do sinal.", 
      "A resolução da imagem ou o contraste do papel milimetrado podem estar insuficientes para medições de alta precisão (ex: Smith-Sgarbossa)."
    );
  }
};

export const generateHeartAnimation = async (base64Image: string, diagnosis: string): Promise<string> => {
  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  let operation = await ai.models.generateVideos({
    model: 'veo-3.1-fast-generate-preview',
    prompt: `Medical 3D heart animation showing specialized activity for: ${diagnosis}. If ischemic, show localized wall motion abnormality and red glowing areas in the affected wall.`,
    image: { imageBytes: base64Image, mimeType: 'image/png' },
    config: { numberOfVideos: 1, resolution: '720p', aspectRatio: '16:9' }
  });
  while (!operation.done) {
    await new Promise(r => setTimeout(r, 10000));
    operation = await ai.operations.getVideosOperation({ operation: operation });
  }
  return `${operation.response?.generatedVideos?.[0]?.video?.uri}&key=${process.env.API_KEY}`;
};

export const explainConceptWithAudio = async (concept: string): Promise<Uint8Array> => {
  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash-preview-tts",
    contents: [{ parts: [{ text: `Explique a importância clínica de ${concept} no diagnóstico de infarto agudo do miocárdio.` }] }],
    config: {
      responseModalities: [Modality.AUDIO],
      speechConfig: {
        voiceConfig: {
          prebuiltVoiceConfig: { voiceName: 'Kore' },
        },
      },
    },
  });

  const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
  if (!base64Audio) throw new Error("Audio generation failed");

  const binaryString = atob(base64Audio);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
};
