
import React, { useState, useRef } from 'react';
import { EcgAnalysisResult, PatientContext } from '../types';
import EcgVisualizer from './EcgVisualizer';
import Heart3D from './Heart3D';
import MedicalReport from './MedicalReport';
import { parseHeartRate } from '../utils/cardioLogic';
import { generateHeartAnimation } from '../services/geminiService';

// Declaration for the global html2pdf library loaded via script tag
declare var html2pdf: any;

interface AnalysisViewProps {
  result: EcgAnalysisResult;
  imagePreview: string;
  onReset: () => void;
  patientContext?: PatientContext;
}

const ALL_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'];

// --- COMPONENTES AUXILIARES ---

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { DigitizationMetrics } from '../types';

// ... (existing imports)

const DigitizedSignalView: React.FC<{ metrics?: DigitizationMetrics }> = ({ metrics }) => {
  if (!metrics || !metrics.representativeBeats || metrics.representativeBeats.length === 0) return null;

  return (
    <div className="glass-card p-6 rounded-[2rem] border border-emerald-500/20 bg-emerald-900/5 mb-8 animate-fade-in-up">
      <div className="flex justify-between items-center mb-6">
        <div className="flex items-center gap-3">
           <div className="w-10 h-10 rounded-xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center">
              <svg className="w-5 h-5 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                 <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
           </div>
           <div>
              <h3 className="text-sm font-black text-emerald-400 uppercase tracking-widest">ECG Digitiser Reconstruction</h3>
              <p className="text-[9px] text-emerald-200/60 font-mono">PhysioNet 2024 Winner Methodology</p>
           </div>
        </div>
        <div className="text-right">
           <span className="text-[9px] text-slate-500 font-bold uppercase block">Segmentation Confidence</span>
           <span className="text-xs font-mono text-emerald-300">{metrics.segmentationConfidence}%</span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {metrics.representativeBeats.map((beat, idx) => {
           const data = beat.timeMs.map((t, i) => ({ time: t, amp: beat.amplitudeMv[i] }));
           return (
             <div key={idx} className="h-48 bg-black/20 rounded-xl border border-white/5 p-2 relative">
                <span className="absolute top-2 left-2 text-[10px] font-black text-slate-500 uppercase bg-black/50 px-2 py-0.5 rounded">Lead {beat.lead}</span>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={data}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
                    <XAxis dataKey="time" hide />
                    <YAxis hide domain={['auto', 'auto']} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#000', border: '1px solid #333', borderRadius: '8px', fontSize: '10px' }}
                      itemStyle={{ color: '#10b981' }}
                      formatter={(val: number) => [val.toFixed(2) + ' mV', 'Amplitude']}
                      labelFormatter={(label) => label + ' ms'}
                    />
                    <ReferenceLine y={0} stroke="#666" strokeDasharray="3 3" />
                    <Line type="monotone" dataKey="amp" stroke="#10b981" strokeWidth={2} dot={false} activeDot={{ r: 4, fill: '#fff' }} />
                  </LineChart>
                </ResponsiveContainer>
             </div>
           );
        })}
      </div>
      
      <div className="mt-4 flex gap-4 text-[9px] text-slate-500 font-mono uppercase">
         <div className="flex items-center gap-1">
            <div className="w-2 h-2 bg-emerald-500 rounded-full"></div>
            <span>Vectorized Signal</span>
         </div>
         <div className="flex items-center gap-1">
            <div className="w-2 h-2 border border-slate-600 border-dashed"></div>
            <span>Isoelectric Line</span>
         </div>
      </div>
    </div>
  );
};

const NeuralInsightsHUD: React.FC<{ telemetry: any; onExplain?: () => void }> = ({ telemetry, onExplain }) => {
    if (!telemetry) return null;

    return (
        <div className="glass-card p-6 rounded-[2rem] border border-cyan-500/20 bg-cyan-900/10 mb-8 animate-fade-in-up relative">
            <div className="flex justify-between items-start mb-6">
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center">
                        <svg className="w-5 h-5 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                           <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.384-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                        </svg>
                    </div>
                    <div>
                        <h3 className="text-sm font-black text-cyan-400 uppercase tracking-widest">Neural Architecture</h3>
                        <p className="text-[9px] text-cyan-200/60 font-mono">{telemetry.modelArchitecture || "Hybrid CNN + Transformer"}</p>
                    </div>
                </div>
                <div className="text-right flex flex-col items-end gap-1">
                    <div>
                        <span className="text-[9px] text-slate-500 font-bold uppercase block">Inference Time</span>
                        <span className="text-xs font-mono text-cyan-300">{telemetry.processingTimeMs}ms</span>
                    </div>
                    {onExplain && (
                        <button 
                            onClick={onExplain}
                            className="mt-2 px-3 py-1 bg-cyan-500/20 hover:bg-cyan-500/40 border border-cyan-500/50 rounded text-[9px] text-cyan-300 font-black uppercase tracking-widest transition-all"
                        >
                            Explain AI Decision
                        </button>
                    )}
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Differential Diagnoses Probabilities */}
                <div>
                    <h4 className="text-[9px] font-black text-slate-500 uppercase tracking-widest mb-3 border-b border-white/5 pb-1">Probabilistic Differential</h4>
                    <div className="space-y-3">
                        {telemetry.differentialDiagnoses?.map((diag: any, idx: number) => (
                            <div key={idx} className="relative group">
                                <div className="flex justify-between text-[9px] font-bold uppercase text-slate-300 mb-1">
                                    <span>{diag.diagnosis}</span>
                                    <span>{diag.probability}%</span>
                                </div>
                                <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                                    <div 
                                        className={`h-full rounded-full transition-all duration-1000 ${idx === 0 ? 'bg-cyan-500 shadow-[0_0_10px_rgba(6,182,212,0.5)]' : 'bg-slate-600'}`} 
                                        style={{ width: `${diag.probability}%` }}
                                    ></div>
                                </div>
                                <p className="text-[8px] text-slate-500 mt-1 opacity-0 group-hover:opacity-100 transition-opacity absolute top-full left-0 bg-black/90 p-2 rounded z-20 w-full">
                                    {diag.reasoning}
                                </p>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Feature Attention */}
                <div>
                    <h4 className="text-[9px] font-black text-slate-500 uppercase tracking-widest mb-3 border-b border-white/5 pb-1">Attention Mechanism Focus</h4>
                    <div className="flex flex-wrap gap-2 mb-4">
                        {telemetry.attentionFocus?.map((lead: string) => (
                            <span key={lead} className="px-2 py-1 rounded bg-cyan-500/10 border border-cyan-500/20 text-[9px] font-black text-cyan-400 uppercase">
                                {lead}
                            </span>
                        ))}
                        {(!telemetry.attentionFocus || telemetry.attentionFocus.length === 0) && (
                            <span className="text-[9px] text-slate-600 italic">Global Attention</span>
                        )}
                    </div>

                    <div className="grid grid-cols-2 gap-2">
                        <div className="p-2 bg-white/5 rounded border border-white/5">
                            <span className="text-[8px] text-slate-500 uppercase block mb-1">CNN (Morphology)</span>
                            <div className="flex flex-wrap gap-1">
                                {telemetry.featureExtraction?.morphologicalFeatures?.slice(0,3).map((f: string, i: number) => (
                                    <span key={i} className="text-[8px] text-slate-300 bg-black/40 px-1 rounded truncate max-w-full">{f}</span>
                                ))}
                            </div>
                        </div>
                        <div className="p-2 bg-white/5 rounded border border-white/5">
                            <span className="text-[8px] text-slate-500 uppercase block mb-1">Transformer (Rhythm)</span>
                            <div className="flex flex-wrap gap-1">
                                {telemetry.featureExtraction?.rhythmFeatures?.slice(0,3).map((f: string, i: number) => (
                                    <span key={i} className="text-[8px] text-slate-300 bg-black/40 px-1 rounded truncate max-w-full">{f}</span>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

const SignalQualityHUD: React.FC<{ quality: any }> = ({ quality }) => {
    if (!quality) return null;
    const snr = quality.snrDb || 0;
    const isGood = snr > 20;
    const isBad = snr < 10;
    const color = isGood ? 'text-emerald-400' : (isBad ? 'text-rose-400' : 'text-amber-400');
    const barColor = isGood ? 'bg-emerald-500' : (isBad ? 'bg-rose-500' : 'bg-amber-500');

    return (
        <div className="glass-card p-4 rounded-2xl border border-white/5 bg-white/5 mb-8 flex items-center justify-between gap-4">
            <div className="flex flex-col">
                <span className="text-[9px] font-black text-slate-500 uppercase tracking-widest">Signal Quality (DSP)</span>
                <div className="flex items-center gap-2 mt-1">
                    <div className="flex gap-0.5">
                        {[1,2,3,4,5].map(i => (
                            <div key={i} className={`w-1 h-3 rounded-sm ${snr >= i*5 ? barColor : 'bg-white/10'}`}></div>
                        ))}
                    </div>
                    <span className={`text-xs font-mono font-bold ${color}`}>
                        {snr} dB ({quality.baselineWander === 'None' ? 'Clean' : quality.baselineWander + ' Wander'})
                    </span>
                </div>
            </div>
            {quality.artifactsDetected?.length > 0 && (
                <div className="text-right">
                    <span className="text-[8px] font-black text-rose-500 uppercase">Artifacts</span>
                    <p className="text-[10px] text-slate-400">{quality.artifactsDetected.join(', ')}</p>
                </div>
            )}
        </div>
    );
};

const LeadReversalWarning: React.FC<{ validation: string }> = ({ validation }) => {
  const isSuspicious = /reversal|troca|invert|swap|misplaced|errado|check|verificar|malpositioned/i.test(validation);
  if (!isSuspicious) return null;
  return (
    <div className="glass-card p-4 rounded-xl border border-orange-500/40 bg-orange-950/20 mb-4 flex items-center gap-4">
        <svg className="w-6 h-6 text-orange-500" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
        <div>
           <h4 className="text-sm font-bold text-orange-400 uppercase">Lead Reversal Detected</h4>
           <p className="text-xs text-orange-200/60">{validation}</p>
        </div>
    </div>
  );
};

const RegulatoryHUD: React.FC<{ guidelines: string[] }> = ({ guidelines }) => {
    if (!guidelines || guidelines.length === 0) return null;
    return (
        <div className="mb-4 flex gap-2 flex-wrap">
            {guidelines.map((g, idx) => (
                <span key={idx} className="px-2 py-1 rounded border border-white/10 bg-white/5 text-[9px] text-slate-400 font-mono uppercase">
                    REF: {g}
                </span>
            ))}
        </div>
    );
};

const IschemiaHUD: React.FC<{ analysis: any }> = ({ analysis }) => {
  if (!analysis) return null;
  const hasSTTrend = analysis.stSegmentTrend && analysis.stSegmentTrend !== 'Neutral';
  if (!hasSTTrend && !analysis.sgarbossaScore) return null;
  
  return (
      <div className="glass-card p-6 rounded-2xl border border-indigo-500/20 bg-indigo-900/10 mb-8">
          <h4 className="text-indigo-400 font-black uppercase tracking-widest text-xs mb-4">Ischemia & Infarction Analysis</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div><span className="block text-[9px] text-slate-500">ST Trend</span><span className="text-sm text-white font-bold">{analysis.stSegmentTrend}</span></div>
              <div><span className="block text-[9px] text-slate-500">Wall</span><span className="text-sm text-white font-bold">{analysis.affectedWall || 'N/A'}</span></div>
              <div><span className="block text-[9px] text-slate-500">Culprit</span><span className="text-sm text-white font-bold">{analysis.culpritArtery || 'N/A'}</span></div>
              <div><span className="block text-[9px] text-slate-500">Reciprocal</span><span className="text-sm text-white font-bold">{analysis.reciprocalChangesFound ? 'Yes' : 'No'}</span></div>
          </div>
      </div>
  );
};

const PacemakerHUD: React.FC<{ analysis: any }> = ({ analysis }) => {
    if (!analysis || analysis.pacingMode === 'None') return null;
    return (
        <div className="glass-card p-4 rounded-xl border border-emerald-500/30 bg-emerald-900/10 mb-8">
            <span className="text-emerald-400 font-black uppercase text-xs">Pacemaker Detected</span>
            <div className="text-white text-sm mt-1">{analysis.pacingMode} Mode at {analysis.pacingSite}</div>
        </div>
    );
};

const ReasoningPathway: React.FC<{ reasoning: string }> = ({ reasoning }) => (
    <div className="mt-6 p-6 bg-black/20 rounded-2xl border border-white/5">
        <h4 className="text-[10px] text-cyan-500 font-black uppercase tracking-widest mb-2">Diagnostic Logic & Reasoning</h4>
        <p className="text-sm text-slate-300 leading-relaxed font-mono">{reasoning}</p>
    </div>
);


// --- COMPONENTE PRINCIPAL ---

const AnalysisView: React.FC<AnalysisViewProps> = ({ result, imagePreview, onReset, patientContext }) => {
  const [isVideoLoading, setIsVideoLoading] = useState(false);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const reportRef = useRef<HTMLDivElement>(null);
  
  const m = result.precisionMeasurements;
  
  const handleAnimate = async () => {
    setIsVideoLoading(true);
    try {
      const url = await generateHeartAnimation(imagePreview, result.diagnosis);
      setVideoUrl(url);
    } catch (err) { alert("Erro na simulação."); } finally { setIsVideoLoading(false); }
  };

  const handleExplain = async () => {
    try {
        const res = await fetch('/api/explain', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ analysisId: result.id, features: m })
        });
        const data = await res.json();
        alert(`Explainable AI Analysis:\nMethod: ${data.method}\n\nTop Features:\n${data.attribution.map((a: any) => `- ${a.feature}: ${a.description} (Score: ${a.score})`).join('\n')}`);
    } catch (e) {
        alert("Failed to fetch explanation.");
    }
  };

  const handleDownloadPDF = () => {
    const element = reportRef.current;
    if (!element) return;
    const opt = {
      margin: 0,
      filename: `CardioAI_Laudo_${result.id || Date.now()}.pdf`,
      image: { type: 'jpeg', quality: 0.98 },
      html2canvas: { scale: 2, useCORS: true, backgroundColor: '#ffffff', logging: false },
      jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' }
    };
    html2pdf().set(opt).from(element).save();
  };

  // Safe accessors for nested wave data
  const qrs = m.waves?.qrsComplex || { durationMs: m.qrsComplex?.durationMs || 0, axisDegrees: 0 };
  const pr = m.waves?.intervals?.prMs || m.prIntervalMs || 0;

  return (
    <div className="max-w-7xl mx-auto pb-20 px-4 animate-fade-in-up">
      
      {/* Hidden Report for PDF */}
      <div style={{ position: 'fixed', left: '-5000px', top: 0 }}>
        <MedicalReport ref={reportRef} result={result} imagePreview={imagePreview} patientContext={patientContext} />
      </div>

      <RegulatoryHUD guidelines={result.guidelineReferences} />
      <LeadReversalWarning validation={result.technicalQuality.leadPlacementValidation} />
      
      {/* NEW Neural Insights Panel */}
      <NeuralInsightsHUD telemetry={m.neuralTelemetry} onExplain={handleExplain} />
      
      {/* ECG Digitiser View */}
      <DigitizedSignalView metrics={m.digitizationMetrics} />
      
      <SignalQualityHUD quality={m.signalQuality} />
      
      <IschemiaHUD analysis={m.ischemiaAnalysis} />
      <PacemakerHUD analysis={m.pacemakerAnalysis} />

      <div className="mb-10 grid grid-cols-1 lg:grid-cols-2 gap-8">
        <Heart3D 
          heartRate={parseHeartRate(result.heartRate)} 
          urgency={result.urgency} 
          diagnosis={result.diagnosis}
          rhythm={result.rhythm}
          numericAxis={qrs.axisDegrees}
          structural={m.structuralAnalysis}
          ischemia={m.ischemiaAnalysis}
          conduction={m.conductionAnalysis}
          qrsDurationMs={qrs.durationMs}
          prIntervalMs={pr}
        />
        
        <div className="glass-card rounded-[3rem] p-8 flex flex-col justify-center relative overflow-hidden">
          {videoUrl ? (
            <video src={videoUrl} controls autoPlay loop className="w-full h-full rounded-3xl" />
          ) : (
            <div className="relative z-10 w-full">
              <h4 className="text-white font-black uppercase italic text-xl tracking-tighter mb-6">Síntese Diagnóstica</h4>
              <div className="space-y-4 mb-8">
                 {result.clinicalImplications.map((imp, idx) => (
                    <div key={idx} className="p-4 rounded-2xl border bg-white/5 border-white/10 text-slate-300 text-[11px] font-medium flex gap-3">
                       <div className="w-1.5 h-1.5 bg-cyan-500 rounded-full mt-1.5 flex-shrink-0"></div>
                       {imp}
                    </div>
                 ))}
              </div>
              <button 
                data-html2canvas-ignore="true"
                onClick={handleAnimate} 
                disabled={isVideoLoading} 
                className="w-full py-4 bg-cyan-600 text-white font-black rounded-2xl text-[10px] uppercase tracking-widest hover:bg-white transition-all shadow-lg shadow-cyan-500/20"
              >
                {isVideoLoading ? 'Gerando Modelo 3D...' : 'Visualizar Fisiopatologia'}
              </button>
            </div>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        <div className="lg:col-span-4 space-y-6">
          <div className={`glass-card p-8 rounded-[2.5rem] border-l-[12px] shadow-2xl ${result.urgency === 'Emergency' ? 'border-rose-500' : 'border-cyan-500'}`}>
             <h2 className="text-white text-3xl font-black uppercase italic leading-tight mb-4">{result.diagnosis}</h2>
             <ReasoningPathway reasoning={result.clinicalReasoning} />
          </div>
          
          {/* Detailed Wave Metrics Table */}
          {m.waves && (
              <div className="glass-card p-6 rounded-2xl">
                  <h4 className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-4">Precision Wave Metrics</h4>
                  <div className="space-y-3 text-xs">
                      <div className="flex justify-between border-b border-white/5 pb-1">
                          <span className="text-slate-400">P-Wave Duration</span>
                          <span className="font-mono text-white">{m.waves.pWave?.durationMs}ms</span>
                      </div>
                      <div className="flex justify-between border-b border-white/5 pb-1">
                          <span className="text-slate-400">QRS Duration</span>
                          <span className="font-mono text-white">{m.waves.qrsComplex?.durationMs}ms</span>
                      </div>
                      <div className="flex justify-between border-b border-white/5 pb-1">
                          <span className="text-slate-400">QRS Axis</span>
                          <span className="font-mono text-white">{m.waves.qrsComplex?.axisDegrees}°</span>
                      </div>
                      <div className="flex justify-between border-b border-white/5 pb-1">
                          <span className="text-slate-400">QTc (Fridericia)</span>
                          <span className="font-mono text-white">{m.waves.intervals?.qtcMs}ms</span>
                      </div>
                  </div>
              </div>
          )}
        </div>

        <div className="lg:col-span-8">
          <EcgVisualizer 
            arrhythmias={[result.rhythm]}
            heartRate={result.heartRate}
            diagnosis={result.diagnosis}
            fullResult={result}
          />
        </div>
      </div>

      {/* HUMAN-IN-THE-LOOP ADJUDICATION PANEL */}
      <div className="glass-card border border-slate-700 rounded-3xl p-8 max-w-4xl mx-auto mt-10 mb-20 animate-fade-in-up">
        <div className="flex items-center gap-4 mb-8 border-b border-white/10 pb-6">
          <div className="w-12 h-12 rounded-full bg-indigo-500/20 flex items-center justify-center shrink-0">
            <svg className="w-6 h-6 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <div>
            <h2 className="text-2xl font-black text-white uppercase italic tracking-tighter">Human-in-the-Loop Adjudication</h2>
            <p className="text-slate-400 text-[11px] font-mono mt-1">Review AI findings, validate beat-level classifications, and provide clinical consensus.</p>
          </div>
        </div>

        <div className="space-y-8">
          {/* Adjudication Actions */}
          <div>
            <h3 className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-4">Clinical Decision</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <button 
                onClick={() => alert('Approved: Added to validated dataset for continuous learning.')}
                className="flex flex-col items-center justify-center gap-3 p-6 bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-500/30 hover:border-emerald-500/50 rounded-2xl transition-all group"
              >
                <div className="w-12 h-12 rounded-full bg-emerald-500/20 flex items-center justify-center group-hover:scale-110 transition-transform">
                  <svg className="w-6 h-6 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                </div>
                <div className="text-center">
                  <div className="font-bold text-emerald-400">Approve AI Findings</div>
                  <div className="text-[10px] text-slate-400 mt-1">Add to validated dataset</div>
                </div>
              </button>

              <button 
                onClick={() => alert('Modify: Open interface to correct false positives/negatives.')}
                className="flex flex-col items-center justify-center gap-3 p-6 bg-amber-500/10 hover:bg-amber-500/20 border border-amber-500/30 hover:border-amber-500/50 rounded-2xl transition-all group"
              >
                <div className="w-12 h-12 rounded-full bg-amber-500/20 flex items-center justify-center group-hover:scale-110 transition-transform">
                  <svg className="w-6 h-6 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                  </svg>
                </div>
                <div className="text-center">
                  <div className="font-bold text-amber-400">Modify Diagnosis</div>
                  <div className="text-[10px] text-slate-400 mt-1">Correct false positives/negatives</div>
                </div>
              </button>

              <button 
                onClick={() => alert('Rejected: Flagged as artifact or non-diagnostic.')}
                className="flex flex-col items-center justify-center gap-3 p-6 bg-rose-500/10 hover:bg-rose-500/20 border border-rose-500/30 hover:border-rose-500/50 rounded-2xl transition-all group"
              >
                <div className="w-12 h-12 rounded-full bg-rose-500/20 flex items-center justify-center group-hover:scale-110 transition-transform">
                  <svg className="w-6 h-6 text-rose-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </div>
                <div className="text-center">
                  <div className="font-bold text-rose-400">Reject / Artifact</div>
                  <div className="text-[10px] text-slate-400 mt-1">Flag as non-diagnostic</div>
                </div>
              </button>
            </div>
          </div>
          
          {/* Audit Log */}
          <div className="pt-6 border-t border-white/5">
            <div className="text-[9px] text-slate-500 font-mono uppercase">
              <p>Pipeline: Hybrid CNN + Temporal Transformer</p>
              <p>Dataset Target: MIT-BIH Arrhythmia / PTB-XL</p>
              <p>Compliance: Ready for ISO 14971 Risk Analysis logging</p>
            </div>
          </div>
        </div>
      </div>

      <div className="fixed bottom-10 left-1/2 -translate-x-1/2 no-print z-[100] flex gap-4 w-full justify-center" data-html2canvas-ignore="true">
        <button onClick={handleDownloadPDF} className="px-8 py-5 bg-cyan-500 text-white rounded-2xl font-black text-[10px] uppercase tracking-widest hover:bg-cyan-400 transition-all shadow-2xl flex items-center gap-2 group">
          Exportar Laudo Oficial
        </button>
        <button onClick={onReset} className="px-8 py-5 bg-white text-slate-950 rounded-2xl font-black text-[10px] uppercase tracking-widest hover:bg-slate-200 transition-all shadow-2xl">
          NOVA ANÁLISE
        </button>
      </div>
    </div>
  );
};

export default AnalysisView;
