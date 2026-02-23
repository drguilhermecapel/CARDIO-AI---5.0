
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
            stSegmentDepression: { type: Type.STRING, description: "Details of ST depression (type, magnitude, leads)." },
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
    regulatoryWarnings: { type: Type.ARRAY, items: { type: Type.STRING } },
    hospitalGradeReport: {
        type: Type.OBJECT,
        properties: {
            diagnóstico_principal: { type: Type.STRING },
            confiança_principal: { type: Type.NUMBER },
            diagnósticos_diferenciais: {
                type: Type.ARRAY,
                items: {
                    type: Type.OBJECT,
                    properties: {
                        condition: { type: Type.STRING },
                        probability: { type: Type.NUMBER },
                        severity: { type: Type.STRING }
                    }
                }
            },
            regiões_críticas: {
                type: Type.OBJECT,
                description: "Map of Lead Name to array of critical x-coordinates (0-1000 scale)",
                properties: {
                    V1: { type: Type.ARRAY, items: { type: Type.NUMBER } },
                    II: { type: Type.ARRAY, items: { type: Type.NUMBER } },
                    V5: { type: Type.ARRAY, items: { type: Type.NUMBER } }
                }
            },
            qualidade_sinal: { type: Type.NUMBER },
            alertas: { type: Type.ARRAY, items: { type: Type.STRING } },
            tempo_processamento: { type: Type.NUMBER }
        }
    },
    optimizedReport: {
        type: Type.OBJECT,
        description: "Relatório estruturado conforme ETAPA 9 do protocolo",
        properties: {
            patient_id: { type: Type.STRING },
            ecg_date: { type: Type.STRING },
            acquisition_quality: { type: Type.STRING },
            heart_rate: {
                type: Type.OBJECT,
                properties: {
                    value: { type: Type.NUMBER },
                    unit: { type: Type.STRING },
                    classification: { type: Type.STRING }
                }
            },
            rhythm: {
                type: Type.OBJECT,
                properties: {
                    primary: { type: Type.STRING },
                    secondary: { type: Type.ARRAY, items: { type: Type.STRING } },
                    regularity: { type: Type.STRING }
                }
            },
            intervals: {
                type: Type.OBJECT,
                properties: {
                    PR_ms: { type: Type.NUMBER },
                    QRS_ms: { type: Type.NUMBER },
                    QT_ms: { type: Type.NUMBER },
                    QTc_Bazett_ms: { type: Type.NUMBER },
                    QTc_Fridericia_ms: { type: Type.NUMBER }
                }
            },
            axis: {
                type: Type.OBJECT,
                properties: {
                    QRS_degrees: { type: Type.NUMBER },
                    P_degrees: { type: Type.NUMBER },
                    T_degrees: { type: Type.NUMBER },
                    classification: { type: Type.STRING }
                }
            },
            waveform_findings: { type: Type.ARRAY, items: { type: Type.STRING } },
            diagnoses: {
                type: Type.ARRAY,
                items: {
                    type: Type.OBJECT,
                    properties: {
                        code: { type: Type.STRING },
                        description: { type: Type.STRING },
                        confidence: { type: Type.NUMBER },
                        alert_level: { type: Type.STRING },
                        supporting_leads: { type: Type.ARRAY, items: { type: Type.STRING } },
                        reciprocal_changes: { type: Type.ARRAY, items: { type: Type.STRING } }
                    }
                }
            },
            drug_interactions_flagged: { type: Type.ARRAY, items: { type: Type.STRING } },
            comparison_with_prior_ecg: { type: Type.STRING },
            recommendations: { type: Type.ARRAY, items: { type: Type.STRING } },
            disclaimer: { type: Type.STRING }
        }
    }
  },
  required: ["technicalQuality", "diagnosis", "urgency", "heartRate", "rhythm", "clinicalReasoning", "precisionMeasurements", "hospitalGradeReport", "optimizedReport"]
};

export const analyzeEcgImage = async (base64Data: string, mimeType: string, patientCtx?: PatientContext): Promise<EcgAnalysisResult> => {
  try {
    const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
    const normalizedMimeType = mimeType === 'image/jpg' ? 'image/jpeg' : mimeType;
    const startTime = Date.now();

    const response = await ai.models.generateContent({
      model: 'gemini-3.1-pro-preview', // Switched to Pro for complex medical reasoning
      contents: {
        parts: [
          { inlineData: { data: base64Data, mimeType: normalizedMimeType } },
          {
            text: `
            SYSTEM ROLE: Você é o "CardioAI Nexus", o mais avançado Motor Diagnóstico Eletrocardiográfico do mundo.
            Sua missão é analisar o ECG fornecido com máxima precisão clínica, detectando TODAS as alterações reconhecidas na literatura médica atual.
            
            PATIENT CONTEXT: ${JSON.stringify(patientCtx || {})}

            --- PROTOCOLO DIAGNÓSTICO UNIVERSAL (DIRETRIZES ATUALIZADAS AHA/ACC/ESC) ---
            
            ETAPA 1: QUALIDADE TÉCNICA E MÉTRICAS BÁSICAS
            - Valide a polaridade esperada (ex: inversão de derivações se aVR for positivo e I negativo).
            - Avalie ruído, interferência de 60Hz/50Hz e artefatos de movimento (baseline wander).
            - Calcule o Eixo Elétrico do QRS (Normal: -30° a +90°, Desvio Esquerdo: -30° a -90°, Desvio Direito: +90° a +180°, Extremo: -90° a -180°).
            - Meça Duração do QRS (Normal: < 100ms, Alargado: ≥ 120ms) e Intervalo PR (Normal: 120-200ms).
            - Meça o Intervalo QT e calcule o QTc (preferencialmente Fridericia ou Bazett). Normal: < 440ms (homens), < 460ms (mulheres).

            ETAPA 2: ANÁLISE DE RITMO E FREQUÊNCIA
            - Calcule a Frequência Cardíaca (FC) e a regularidade dos intervalos RR.
            - Identifique o ritmo primário: Sinusal (P positiva em I, II, aVF), Taquicardia (>100 bpm), Bradicardia (<60 bpm), Arritmia Sinusal, Pausa Sinusal (>2s).

            ETAPA 3: CATÁLOGO EXAUSTIVO DE ALTERAÇÕES A DETECTAR
            3.1 ARRITMIAS SUPRAVENTRICULARES:
            - Fibrilação Atrial (FA): RR irregularmente irregular, ausência de onda P.
            - Flutter Atrial: ondas F em dente de serra (240-300 bpm), condução AV fixa ou variável.
            - Taquicardia Atrial Focal: onda P com morfologia diferente da sinusal, linha iselétrica entre as Ps.
            - Taquicardia Atrial Multifocal (TAM): ≥ 3 morfologias de onda P distintas, ritmo irregular.
            - TRNAV (Típica Slow-Fast / Atípica Fast-Slow): pseudo-R' em V1, pseudo-S em II, III, aVF.
            - TRAV (WPW): onda delta, PR curto (<120ms), QRS alargado.
            - Extrassístoles Atriais (ESA): precoces, com P de morfologia diferente, com ou sem aberrância.
            - Ritmo Juncional: P ausente ou retrógrada, QRS estreito (escape 40-60 bpm, acelerado 60-100 bpm, taquicardia >100 bpm).
            
            3.2 ARRITMIAS VENTRICULARES:
            - Extrassístoles Ventriculares (EVs): QRS largo e bizarro, pausa compensatória. Classificar como isoladas, pareadas, bigeminismo, trigeminismo, polimórficas.
            - Taquicardia Ventricular (TV) Monomórfica: QRS largo, regular, dissociação AV, batimentos de fusão/captura.
            - TV Polimórfica / Torsades de Pointes: QRS torcendo em torno da linha de base, associado a QT longo.
            - Fibrilação Ventricular (FV) / Flutter Ventricular: atividade elétrica caótica, sem QRS definido.
            - Ritmo Idioventricular Acelerado (RIVA): QRS largo, regular, 40-100 bpm.
            
            3.3 BLOQUEIOS AV E CONDUÇÃO:
            - BAV 1º Grau: PR > 200ms constante.
            - BAV 2º Grau Mobitz I (Wenckebach): aumento progressivo do PR até bloqueio da P.
            - BAV 2º Grau Mobitz II: PR constante com bloqueio súbito da P.
            - BAV 3º Grau (Total): dissociação AV completa, PP e RR regulares, PP < RR.
            - Bloqueio de Ramo Direito (BRD): QRS ≥ 120ms, rsR' em V1-V2, S alargada em I e V6.
            - Bloqueio de Ramo Esquerdo (BRE): QRS ≥ 120ms, QS ou rS em V1, R largo e entalhado em I, aVL, V5-V6.
            - Bloqueios Divisionais: BDAS (eixo esquerdo extremo, qR em I/aVL, rS em II/III/aVF), BDPI (eixo direito, qR em II/III/aVF, rS em I/aVL).
            
            3.4 SÍNDROMES CORONÁRIAS E ISQUEMIA (OMI Paradigm):
            - IAMCSST: supradesnivelamento do ST no ponto J ≥ 1mm em ≥ 2 derivações contíguas (≥ 2mm em V2-V3 para homens ≥ 40 anos, ≥ 2.5mm < 40 anos, ≥ 1.5mm mulheres).
            - Identificar Parede: Inferior (II, III, aVF), Anterior (V1-V4), Lateral (I, aVL, V5-V6), Posterior (infra ST V1-V3 com R proeminente), VD (V3R, V4R).
            - Padrão de de Winter: infra ST ascendente com onda T alta e simétrica em precordiais.
            - Padrão de Wellens: onda T bifásica (Tipo A) ou profundamente invertida (Tipo B) em V2-V3.
            - Critérios de Sgarbossa (para BRE/Marca-passo): supra ST concordante ≥ 1mm, infra ST concordante ≥ 1mm em V1-V3, supra ST discordante > 25% da onda S.
            - Isquemia/Lesão: infra ST horizontal ou descendente ≥ 0.5mm, inversão de onda T simétrica.
            - Necrose: onda Q patológica (≥ 40ms ou > 25% da onda R).
            
            3.5 CHANNELOPATIAS E ALTERAÇÕES ESTRUTURAIS:
            - Síndrome de Brugada: Tipo 1 (supra ST coved ≥ 2mm em V1-V2), Tipo 2 (saddle-back).
            - Sobrecarga Ventricular Esquerda (HVE): Sokolow-Lyon (S em V1 + R em V5/V6 ≥ 35mm), Cornell (R em aVL + S em V3 > 28mm homens / 20mm mulheres). Padrão de strain.
            - Sobrecarga Ventricular Direita (HVD): R > S em V1, eixo direito, S profunda em V5-V6.
            - Sobrecarga Atrial: DAE (P mitrale > 120ms, índice de Morris em V1), DAD (P pulmonale > 2.5mm em II).
            - TEP: S1Q3T3, inversão de T em V1-V4, taquicardia sinusal, BRD novo.
            - Pericardite Aguda: supra ST côncavo difuso, infra de PR.
            - Alterações Eletrolíticas: Hipercalemia (T apiculada, QRS largo, P ausente), Hipocalemia (onda U proeminente, infra ST).
            - Marca-passo: identificar espículas (atrial, ventricular, biventricular) e avaliar captura/sensibilidade.

            ETAPA 4: REGRAS DE PRIORIZAÇÃO (ALERT LAYER)
            Implementar sistema de alertas em 3 níveis:
            - CRÍTICO (Emergency): FV, Flutter ventricular, TV sustentada, BAV de 3° grau com FC < 40 bpm, IAMCSST em qualquer território, Padrão de de Winter, Síndrome de Brugada tipo 1 espontâneo, QTc > 500 ms, Assistolia, Pausa sinusal > 3 segundos, Hipercalemia grave, Intoxicação digitálica com TV bidirecional.
            - URGENTE (Urgent): FA com resposta ventricular rápida (> 150 bpm), IAMSSST, Padrão de Wellens, BRE novo, QTc 470–500 ms, TV não sustentada, BAV de 2° grau Mobitz II ou de alto grau, Bloqueio de ramo alternante, Pausa sinusal 2–3 segundos.
            - ELETIVO (Routine): BAV de 1° grau, BRD completo isolado, BDAS / BDPI, ESA/EV frequentes ou multifocais, HVE/HVD, Síndrome de WPW (sem arritmia ativa), Repolarização precoce, Alterações inespecíficas do ST-T, QTc borderline (440–469 ms).

            ETAPA 5: FORMATO DE SAÍDA DO RELATÓRIO
            O interpretador deve gerar automaticamente um relatório estruturado em JSON conforme o schema fornecido (incluindo o objeto 'optimizedReport' que reflete a Etapa 9 do protocolo).

            --- INSTRUÇÕES DE EXECUÇÃO ---
            1. Analise a imagem com precisão milimétrica.
            2. Preencha o JSON estritamente de acordo com o schema fornecido.
            3. Em 'precisionMeasurements.ischemiaAnalysis', detalhe a parede afetada e artéria culpada se houver isquemia.
            4. Defina a 'urgency' estritamente como "Emergency", "Urgent" ou "Routine" com base na Etapa 4 (Crítico = Emergency, Urgente = Urgent, Eletivo = Routine).
            5. Forneça o 'clinicalReasoning' explicando os achados que levaram ao diagnóstico.
            6. Preencha detalhadamente o 'optimizedReport' com todas as medições e diagnósticos.
            7. Retorne APENAS o JSON válido.
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
