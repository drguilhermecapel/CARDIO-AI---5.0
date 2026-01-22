
import React, { useState, useCallback, useEffect } from 'react';
import Header from './components/Header';
import Disclaimer from './components/Disclaimer';
import FileUpload from './components/FileUpload';
import AnalysisView from './components/AnalysisView';
import HistoryView from './components/HistoryView';
import VoiceAssistant from './components/VoiceAssistant'; 
import PatientForm from './components/PatientForm';
import EcgGlossary from './components/EcgGlossary';
import { EcgAnalysisResult, AnalysisStatus, EcgRecord, PatientContext } from './types';
import { analyzeEcgImage, AnalysisError } from './services/geminiService';
import { getHistory } from './services/database';

interface ErrorState {
  code: string;
  message: string;
  suggestion: string;
}

function App() {
  const [status, setStatus] = useState<AnalysisStatus>(AnalysisStatus.IDLE);
  const [result, setResult] = useState<EcgAnalysisResult | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [error, setError] = useState<ErrorState | null>(null);
  const [patientContext, setPatientContext] = useState<PatientContext | undefined>(undefined);
  const [showHistory, setShowHistory] = useState(false);
  const [historyRecords, setHistoryRecords] = useState<EcgRecord[]>([]);

  useEffect(() => {
    setHistoryRecords(getHistory());
  }, [status]); 

  const handlePatientContextConfirm = (ctx: PatientContext) => {
      setPatientContext(ctx);
  };

  const handleValidationError = (msg: string) => {
    const [code, ...rest] = msg.split(': ');
    setError({
      code: code || 'VALIDATION_ERROR',
      message: rest.join(': ') || msg,
      suggestion: 'Verifique se o arquivo é um ECG válido, com todas as derivações nítidas e boa iluminação.'
    });
    setStatus(AnalysisStatus.ERROR);
  };

  const handleFileSelect = useCallback(async (file: File) => {
    setStatus(AnalysisStatus.ANALYZING);
    setError(null);
    setResult(null);

    const reader = new FileReader();
    reader.onload = async (e) => {
      const base64String = e.target?.result as string;
      setImagePreview(base64String);
      const base64Data = base64String.split(',')[1];
      const mimeType = file.type;

      try {
        const analysisResult = await analyzeEcgImage(base64Data, mimeType, patientContext);
        setResult(analysisResult);
        setStatus(AnalysisStatus.SUCCESS);
      } catch (err: any) {
        if (err instanceof AnalysisError) {
            setError({ code: err.code, message: err.message, suggestion: err.suggestion });
        } else {
            setError({
                code: 'SYS_LINK_FAILURE',
                message: err.message || 'Falha na rede neural de processamento.',
                suggestion: 'O sistema encontrou um ruído inesperado. Tente fazer o upload novamente com uma captura em maior resolução.'
            });
        }
        setStatus(AnalysisStatus.ERROR);
      }
    };
    reader.readAsDataURL(file);
  }, [patientContext]);

  const handleReset = useCallback(() => {
    setStatus(AnalysisStatus.IDLE);
    setResult(null);
    setImagePreview(null);
    setError(null);
    setPatientContext(undefined);
  }, []);

  return (
    <div className="min-h-screen flex flex-col font-sans text-slate-100 relative selection:bg-cyan-500/30 overflow-x-hidden bg-[#020617]">
      <Header />
      <Disclaimer />
      <VoiceAssistant />

      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute inset-0 opacity-[0.03]" style={{ backgroundImage: 'linear-gradient(rgba(255,255,255,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.05) 1px, transparent 1px)', backgroundSize: '40px 40px' }}></div>
        <div className="absolute top-0 left-0 w-full h-1/2 bg-gradient-to-b from-cyan-500/5 to-transparent"></div>
        <div className="absolute top-[-10%] right-[-10%] w-[800px] h-[800px] bg-cyan-500/5 rounded-full blur-[160px]"></div>
        <div className="absolute bottom-[-10%] left-[-10%] w-[800px] h-[800px] bg-magenta-500/5 rounded-full blur-[160px]"></div>
      </div>

      {showHistory && (
        <HistoryView 
          records={historyRecords} 
          onSelect={(rec) => { setResult(rec); setStatus(AnalysisStatus.SUCCESS); setShowHistory(false); }} 
          onClose={() => setShowHistory(false)} 
        />
      )}

      <main className="flex-grow container mx-auto px-4 py-16 relative z-10">
        {status === AnalysisStatus.IDLE && (
          <div className="flex flex-col items-center justify-center animate-fade-in">
            <div className="text-center mb-16 max-w-5xl relative">
              <div className="relative mb-16 flex justify-center">
                <div className="relative w-48 h-48">
                  <div className="absolute inset-0 border-[3px] border-cyan-500/20 rounded-full animate-spin-slow"></div>
                  <div className="absolute inset-4 border border-dashed border-magenta-500/30 rounded-full animate-spin-reverse"></div>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <svg className="w-20 h-20 text-cyan-400 drop-shadow-[0_0_15px_rgba(6,182,212,0.8)]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                      <path d="M22 12h-4l-3 9L9 3l-3 9H2" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  </div>
                </div>
              </div>

              <h2 className="text-8xl md:text-9xl font-black text-white tracking-tighter mb-8 uppercase italic leading-[0.85]">
                NEURAL<br/><span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-magenta-400 neon-text-glow">CARDIAC</span><br/>LINK
              </h2>
              
              <div className="max-w-2xl mx-auto mb-16 p-4 border-y border-white/5 bg-white/5 backdrop-blur-sm">
                <p className="text-xs text-slate-400 font-mono tracking-[0.3em] uppercase opacity-70">
                  Precision Bio-Signal Analysis // Real-Time Diagnostic Synthesis // OMI-Aware Engine v6.1
                </p>
              </div>
              
              <div className="flex justify-center gap-6">
                 <button 
                   onClick={() => setShowHistory(true)}
                   className="group relative px-10 py-5 bg-transparent border border-white/10 rounded-xl overflow-hidden transition-all hover:border-cyan-500/50"
                 >
                   <div className="absolute inset-0 bg-white/5 translate-y-full group-hover:translate-y-0 transition-transform"></div>
                   <div className="relative z-10 flex items-center gap-3">
                     <div className="w-1.5 h-1.5 rounded-full bg-cyan-400 group-hover:animate-ping"></div>
                     <span className="text-[10px] font-black text-white uppercase tracking-[0.3em]">Access Clinical Vault</span>
                   </div>
                 </button>
              </div>
            </div>
            
            <div className="w-full flex flex-col items-center gap-12">
              <PatientForm onConfirm={handlePatientContextConfirm} />
              <FileUpload onFileSelect={handleFileSelect} isLoading={false} onValidationError={handleValidationError} />
              <EcgGlossary />
            </div>
          </div>
        )}

        {status === AnalysisStatus.ANALYZING && (
          <div className="flex flex-col items-center justify-center py-32 min-h-[60vh]">
             <div className="relative w-80 h-80 mb-20 flex items-center justify-center">
               <div className="absolute inset-0 border border-cyan-500/10 rounded-full animate-ping"></div>
               <div className="absolute inset-10 border-2 border-cyan-500 rounded-full border-t-transparent animate-spin"></div>
               <div className="absolute inset-20 border border-dashed border-magenta-500/40 rounded-full animate-spin-reverse"></div>
               
               <div className="text-center relative z-10">
                 <div className="text-3xl font-black text-white italic animate-pulse tracking-tighter uppercase">Scanning</div>
                 <div className="text-[10px] text-cyan-400 font-mono font-black mt-2 tracking-widest uppercase">Syncing Neural Nodes...</div>
               </div>
             </div>
             
             <div className="max-w-md w-full glass-card p-6 rounded-2xl border-white/5 border-l-cyan-500 border-l-4">
               <div className="flex justify-between items-center mb-4">
                 <span className="text-[10px] text-slate-500 font-mono font-black uppercase">Process Log</span>
                 <span className="text-[10px] text-cyan-400 font-mono animate-pulse">Running...</span>
               </div>
               <div className="space-y-3 font-mono text-[9px] uppercase tracking-widest text-slate-400">
                  <div className="flex items-center gap-3"><div className="w-1 h-1 bg-green-500 rounded-full"></div> Identifying Isoelectric Line</div>
                  <div className="flex items-center gap-3"><div className="w-1 h-1 bg-green-500 rounded-full"></div> Filtering Motion Artifacts</div>
                  <div className="flex items-center gap-3 animate-pulse"><div className="w-1 h-1 bg-cyan-500 rounded-full"></div> Mapping Voltage Territories</div>
               </div>
             </div>
          </div>
        )}

        {status === AnalysisStatus.SUCCESS && result && (
          <AnalysisView result={result} imagePreview={imagePreview || ''} onReset={handleReset} />
        )}

        {status === AnalysisStatus.ERROR && error && (
          <div className="max-w-4xl mx-auto animate-fade-in-up">
            <div className="glass-card rounded-[3rem] border-red-500/30 relative overflow-hidden shadow-[0_0_100px_rgba(239,68,68,0.1)]">
              {/* Scanline error effect */}
              <div className="absolute top-0 left-0 w-full h-1 bg-red-500/50 blur-[2px] animate-scanner-fast z-30 opacity-40"></div>
              
              <div className="p-10 md:p-16 relative z-10">
                <div className="flex flex-col md:flex-row gap-10 items-start mb-14">
                  <div className="w-28 h-28 bg-red-500/10 rounded-[2.5rem] flex items-center justify-center flex-shrink-0 border border-red-500/20 shadow-[0_0_40px_rgba(239,68,68,0.2)]">
                    <svg className="w-14 h-14 text-red-500 animate-pulse" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path d="M12 9v2m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" strokeWidth="1.5" strokeLinecap="round" />
                    </svg>
                  </div>
                  
                  <div className="flex-1">
                    <div className="inline-flex items-center gap-2 px-3 py-1 bg-red-500/10 border border-red-500/20 rounded-full mb-6">
                      <div className="w-1.5 h-1.5 bg-red-500 rounded-full animate-ping"></div>
                      <span className="text-[10px] font-mono text-red-400 font-black uppercase tracking-widest">
                        ERROR_LOG :: {error.code}
                      </span>
                    </div>
                    <h3 className="text-5xl font-black text-white mb-6 uppercase italic tracking-tighter leading-tight">
                      ANOMALIA NO <span className="text-red-500">SINAL BIO-ELÉTRICO</span>
                    </h3>
                    <p className="text-slate-400 text-xl leading-relaxed font-medium">
                      O motor neural falhou ao sintetizar o registro. <span className="text-slate-500 italic">{error.message}</span>
                    </p>
                  </div>
                </div>

                {/* THE PROMINENT SUGGESTION BOX */}
                <div className="mb-14 group">
                  <div className="p-8 rounded-[2.5rem] bg-gradient-to-br from-red-500/20 to-transparent border-2 border-red-500/40 shadow-[0_0_50px_rgba(239,68,68,0.1)] transition-all group-hover:border-red-500/60">
                    <div className="flex items-center gap-4 mb-5">
                      <div className="w-10 h-10 rounded-xl bg-red-500 flex items-center justify-center shadow-lg shadow-red-500/20">
                        <svg className="w-5 h-5 text-slate-950" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path d="M13 10V3L4 14h7v7l9-11h-7z" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
                        </svg>
                      </div>
                      <h4 className="text-xs font-black text-red-500 uppercase tracking-[0.4em]">Protocolo de Resolução</h4>
                    </div>
                    <p className="text-3xl text-white font-black leading-[1.1] italic tracking-tighter">
                      {error.suggestion}
                    </p>
                    <div className="mt-6 flex gap-2">
                       {[1,2,3,4].map(i => <div key={i} className="h-1 w-12 bg-red-500/30 rounded-full overflow-hidden relative"><div className="absolute inset-0 bg-red-500 animate-scan-line" style={{animationDelay: `${i*0.5}s`}}></div></div>)}
                    </div>
                  </div>
                </div>

                <div className="flex flex-col md:flex-row gap-4">
                  <button 
                    onClick={handleReset} 
                    className="flex-1 py-7 bg-white text-slate-950 font-black text-[11px] uppercase tracking-[0.4em] rounded-[1.5rem] transition-all hover:bg-red-500 hover:text-white shadow-2xl flex items-center justify-center gap-4 group"
                  >
                    <svg className="w-5 h-5 transition-transform group-hover:rotate-180" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                    REINICIALIZAR MÓDULO NEURAL
                  </button>
                  <button 
                    onClick={() => window.location.reload()}
                    className="px-10 py-7 bg-white/5 text-white font-black text-[11px] uppercase tracking-[0.4em] rounded-[1.5rem] border border-white/10 hover:bg-white/10 transition-all"
                  >
                    SYNC STATUS
                  </button>
                </div>
              </div>
              
              <div className="bg-red-500/5 border-t border-white/5 p-5 flex justify-between items-center overflow-hidden">
                <div className="flex gap-2">
                    {[1,2,3,4,5,6,7,8].map(i => (
                      <div key={i} className={`w-1 h-1 rounded-full ${i % 2 === 0 ? 'bg-red-500' : 'bg-red-500/20'} animate-pulse`} style={{animationDelay: `${i*0.1}s`}}></div>
                    ))}
                </div>
                <span className="text-[8px] font-mono text-red-500/40 uppercase tracking-[1.5em] font-black whitespace-nowrap">
                  CORE_RECOVERY_PROTOCOL_STANDBY_V6
                </span>
              </div>
            </div>
          </div>
        )}
      </main>

      <style>{`
        @keyframes spin-slow { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        .animate-spin-slow { animation: spin-slow 20s linear infinite; }
        @keyframes spin-reverse { from { transform: rotate(360deg); } to { transform: rotate(0deg); } }
        .animate-spin-reverse { animation: spin-reverse 15s linear infinite; }
        @keyframes scan-line {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
        .animate-scan-line { animation: scan-line 3s linear infinite; }
        @keyframes scanner-fast {
          0% { top: 0; }
          50% { top: 100%; }
          100% { top: 0; }
        }
        .animate-scanner-fast { animation: scanner-fast 1.5s ease-in-out infinite; }
      `}</style>
    </div>
  );
}

export default App;
