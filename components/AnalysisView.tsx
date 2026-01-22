
import React, { useState } from 'react';
import { EcgAnalysisResult } from '../types';
import EcgVisualizer from './EcgVisualizer';
import Heart3D from './Heart3D';
import { parseHeartRate } from '../utils/cardioLogic';
import { generateHeartAnimation } from '../services/geminiService';

interface AnalysisViewProps {
  result: EcgAnalysisResult;
  imagePreview: string;
  onReset: () => void;
}

const IschemiaHUD: React.FC<{ analysis: any }> = ({ analysis }) => {
  if (!analysis || (analysis.sgarbossaScore === 0 && !analysis.deWinterPattern && analysis.wellensSyndrome === 'None')) return null;

  return (
    <div className="glass-card p-8 rounded-[2.5rem] border border-rose-500/20 bg-rose-950/10 mb-8 animate-fade-in-up">
      <div className="flex items-center gap-3 mb-6">
        <div className="bg-rose-500 p-2.5 rounded-xl shadow-lg shadow-rose-500/20">
           <svg className="w-5 h-5 text-slate-950" fill="none" stroke="currentColor" viewBox="0 0 24 24">
             <path d="M13 10V3L4 14h7v7l9-11h-7z" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
           </svg>
        </div>
        <div>
          <h3 className="text-xl font-black text-white uppercase tracking-tighter italic">Isquemia Crítica (OMI)</h3>
          <p className="text-[10px] text-rose-400 font-mono uppercase font-black">Detecção de equivalentes de STEMI</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="p-4 rounded-2xl bg-white/5 border border-white/10">
          <span className="text-[9px] text-slate-500 uppercase font-black tracking-widest block mb-1">Sgarbossa Score</span>
          <div className={`text-sm font-black italic ${analysis.sgarbossaScore >= 3 ? 'text-rose-500' : 'text-white'}`}>
            {analysis.sgarbossaScore || 0} PTS
          </div>
        </div>
        <div className="p-4 rounded-2xl bg-white/5 border border-white/10">
          <span className="text-[9px] text-slate-500 uppercase font-black tracking-widest block mb-1">Ratio ST/S</span>
          <div className={`text-sm font-black italic ${analysis.smithSgarbossaRatio < -0.25 ? 'text-rose-500 animate-pulse' : 'text-white'}`}>
            {analysis.smithSgarbossaRatio || 'N/A'}
          </div>
        </div>
        <div className="p-4 rounded-2xl bg-white/5 border border-white/10">
          <span className="text-[9px] text-slate-500 uppercase font-black tracking-widest block mb-1">Síndrome de Wellens</span>
          <div className="text-sm font-black text-white italic uppercase">{analysis.wellensSyndrome}</div>
        </div>
        <div className="p-4 rounded-2xl bg-white/5 border border-white/10">
          <span className="text-[9px] text-slate-500 uppercase font-black tracking-widest block mb-1">Padrão de Winter</span>
          <div className={`text-xs font-black px-4 py-1 rounded-full ${analysis.deWinterPattern ? 'bg-rose-500 text-white' : 'bg-white/5 text-slate-500'}`}>
            {analysis.deWinterPattern ? 'DETECTADO' : 'AUSENTE'}
          </div>
        </div>
      </div>
    </div>
  );
};

const ReasoningPathway: React.FC<{ reasoning: string; measurements: any }> = ({ reasoning, measurements }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const getActiveFrameworks = () => {
    const frameworks = [];
    if (measurements.baranchukAnalysis && measurements.baranchukAnalysis.iabType !== 'None') frameworks.push('Baranchuk Algorithm');
    if (measurements.arvdAnalysis && measurements.arvdAnalysis.epsilonWaveDetected) frameworks.push('ARVD Task Force');
    if (measurements.ischemiaAnalysis && measurements.ischemiaAnalysis.sgarbossaScore > 0) frameworks.push('Smith-Sgarbossa');
    return frameworks;
  };

  const frameworks = getActiveFrameworks();

  return (
    <div className="mt-8 space-y-4">
      <div className="flex items-center justify-between mb-2">
        <h5 className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em]">Neural Reasoning Pathway</h5>
        <div className="flex gap-2">
          {frameworks.map(f => (
            <span key={f} className="px-2 py-0.5 rounded bg-cyan-500/10 border border-cyan-500/20 text-[8px] font-bold text-cyan-400 uppercase">
              {f}
            </span>
          ))}
        </div>
      </div>

      <div className="glass-card rounded-2xl border border-white/5 bg-black/40 overflow-hidden">
        <div className="p-5 border-b border-white/5 flex items-center gap-4">
          <div className="flex flex-col gap-1">
             <div className="flex gap-1">
               <div className="w-1 h-1 bg-cyan-400 rounded-full animate-pulse"></div>
               <div className="w-1 h-1 bg-cyan-400/40 rounded-full"></div>
             </div>
             <span className="text-[8px] font-mono text-cyan-500/60 uppercase">Engine v6.1</span>
          </div>
          <div className="flex-1">
            <p className="text-[10px] text-slate-400 font-medium leading-relaxed italic">
              {isExpanded ? reasoning : `${reasoning.substring(0, 160)}...`}
            </p>
          </div>
          <button onClick={() => setIsExpanded(!isExpanded)} className="text-[9px] font-black text-cyan-400 uppercase tracking-widest">
            {isExpanded ? '[COLLAPSE]' : '[EXPAND]'}
          </button>
        </div>
      </div>
    </div>
  );
};

const PacemakerHUD: React.FC<{ analysis: any }> = ({ analysis }) => {
  if (!analysis || analysis.pacingMode === 'None') return null;
  return (
    <div className="glass-card p-8 rounded-[2.5rem] border border-emerald-500/20 bg-emerald-950/10 mb-8">
      <h3 className="text-xl font-black text-white uppercase italic mb-6">Eletrofisiologia Digital</h3>
      <div className="grid grid-cols-4 gap-6">
        <div className="p-4 rounded-2xl bg-white/5 border border-white/10">
          <span className="text-[9px] text-slate-500 uppercase block mb-1">Mode</span>
          <div className="text-sm font-black text-white">{analysis.pacingMode}</div>
        </div>
        <div className="p-4 rounded-2xl bg-white/5 border border-white/10">
          <span className="text-[9px] text-slate-500 uppercase block mb-1">Site</span>
          <div className="text-sm font-black text-emerald-400">{analysis.pacingSite}</div>
        </div>
        <div className="p-4 rounded-2xl bg-white/5 border border-white/10">
          <span className="text-[9px] text-slate-500 uppercase block mb-1">Integrity</span>
          <div className="text-sm font-black text-white">{analysis.captureIntegrity}</div>
        </div>
        <div className="p-4 rounded-2xl bg-white/5 border border-white/10">
          <span className="text-[9px] text-slate-500 uppercase block mb-1">Spike</span>
          <div className="text-sm font-black text-white">{analysis.spikeAmplitude}</div>
        </div>
      </div>
    </div>
  );
};

const AnalysisView: React.FC<AnalysisViewProps> = ({ result, imagePreview, onReset }) => {
  const [isVideoLoading, setIsVideoLoading] = useState(false);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  
  const handleAnimate = async () => {
    setIsVideoLoading(true);
    try {
      const url = await generateHeartAnimation(imagePreview.split(',')[1], result.diagnosis);
      setVideoUrl(url);
    } catch (err) { alert("Erro na simulação."); } finally { setIsVideoLoading(false); }
  };

  return (
    <div className="max-w-7xl mx-auto pb-20 px-4 animate-fade-in-up">
      <IschemiaHUD analysis={result.precisionMeasurements.ischemiaAnalysis} />
      <PacemakerHUD analysis={result.precisionMeasurements.pacemakerAnalysis} />

      <div className="mb-10 grid grid-cols-1 lg:grid-cols-2 gap-8">
        <Heart3D 
          heartRate={parseHeartRate(result.heartRate)} 
          urgency={result.urgency} 
          diagnosis={result.diagnosis}
          rhythm={result.rhythm}
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
              <button onClick={handleAnimate} disabled={isVideoLoading} className="w-full py-4 bg-cyan-600 text-white font-black rounded-2xl text-[10px] uppercase tracking-widest hover:bg-white transition-all shadow-lg shadow-cyan-500/20">
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
             <ReasoningPathway reasoning={result.clinicalReasoning} measurements={result.precisionMeasurements} />
          </div>
        </div>

        <div className="lg:col-span-8">
          <EcgVisualizer 
            arrhythmias={[result.rhythm]}
            heartRate={result.heartRate}
            diagnosis={result.diagnosis}
          />
        </div>
      </div>

      <div className="fixed bottom-10 left-1/2 -translate-x-1/2 no-print z-[100]">
        <button onClick={onReset} className="px-12 py-5 bg-white text-slate-950 rounded-2xl font-black text-[10px] uppercase tracking-widest hover:bg-cyan-400 transition-all shadow-2xl">
          NOVA ANÁLISE BIO-ELÉTRICA
        </button>
      </div>
    </div>
  );
};

export default AnalysisView;
