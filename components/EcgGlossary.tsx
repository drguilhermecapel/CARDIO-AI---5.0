
import React, { useState } from 'react';
import { explainConceptWithAudio } from '../services/geminiService';

const EcgGlossary: React.FC = () => {
  const [playing, setPlaying] = useState<string | null>(null);

  const concepts = [
    {
      id: 'avnrt',
      title: 'Dupla Via Nodal',
      subtitle: 'Fisiologia da TRN',
      description: 'Presença de duas vias (lenta e rápida) no nó AV. Pode gerar o fenômeno de Double Fire ou sustentar reentrada nodal.',
      color: 'rose'
    },
    {
      id: 'bayes',
      title: 'Síndrome de Bayés',
      subtitle: 'Bloqueio Interatrial Avançado',
      description: 'Bloqueio severo no feixe de Bachmann. Forte preditor de arritmias atriais e risco cardioembólico.',
      color: 'magenta'
    },
    {
      id: 'pwave',
      title: 'Onda P (+/-)',
      subtitle: 'Morfologia Plus-Minus',
      description: 'Padrão bifásico em V1. Quando a fase negativa é profunda, indica sobrecarga atrial esquerda.',
      color: 'cyan'
    },
    {
      id: 'epsilon',
      title: 'Onda Épsilon',
      subtitle: 'Sinal de ARVC/DAVD',
      description: 'Pequena deflexão positiva ao final do QRS. Patognomônico de Displasia Arritmogênica do Ventrículo Direito.',
      color: 'emerald'
    }
  ];

  const handlePlayAudio = async (concept: string) => {
    setPlaying(concept);
    try {
      const audioData = await explainConceptWithAudio(concept);
      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
      
      const dataInt16 = new Int16Array(audioData.buffer);
      const buffer = audioCtx.createBuffer(1, dataInt16.length, 24000);
      const channelData = buffer.getChannelData(0);
      for (let i = 0; i < dataInt16.length; i++) {
        channelData[i] = dataInt16[i] / 32768.0;
      }

      const source = audioCtx.createBufferSource();
      source.buffer = buffer;
      source.connect(audioCtx.destination);
      source.start();
      source.onended = () => setPlaying(null);
    } catch (err) {
      console.error(err);
      setPlaying(null);
    }
  };

  return (
    <div className="w-full max-w-6xl mx-auto py-20 px-4">
      <div className="text-center mb-16">
        <h3 className="text-[10px] font-black text-cyan-500 uppercase tracking-[0.5em] mb-4">Neural Education Module</h3>
        <h2 className="text-4xl font-black text-white italic tracking-tighter uppercase">Atlas de Anatomia Elétrica</h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {concepts.map((item) => (
          <div key={item.id} className="glass-card p-6 rounded-[2.5rem] border border-white/5 relative group overflow-hidden transition-all hover:border-white/20 hover:bg-white/5">
            <div className={`absolute top-0 right-0 w-24 h-24 bg-${item.color}-500/5 blur-3xl rounded-full -mr-12 -mt-12`}></div>
            
            <div className="relative z-10 flex flex-col h-full">
              <div className={`w-10 h-10 mb-4 rounded-xl bg-${item.color}-500/10 border border-${item.color}-500/20 flex items-center justify-center`}>
                 <svg className={`w-5 h-5 text-${item.color}-400`} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                   <path d="M22 12h-4l-3 9L9 3l-3 9H2" strokeLinecap="round" strokeLinejoin="round" />
                 </svg>
              </div>

              <h4 className="text-white text-lg font-black uppercase italic mb-1 leading-tight">{item.title}</h4>
              <p className={`text-[8px] font-bold text-${item.color}-500 uppercase tracking-widest mb-4`}>{item.subtitle}</p>
              
              <p className="text-slate-400 text-[10px] leading-relaxed mb-6 flex-grow">
                {item.description}
              </p>

              <button 
                onClick={() => handlePlayAudio(item.title)}
                disabled={!!playing}
                className={`w-full py-3 rounded-xl text-[8px] font-black uppercase tracking-widest transition-all flex items-center justify-center gap-2 ${
                  playing === item.title 
                    ? 'bg-white text-slate-950 scale-95' 
                    : 'bg-white/5 text-white hover:bg-white hover:text-slate-950'
                }`}
              >
                {playing === item.title ? "Explaining..." : "Listen Briefing"}
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default EcgGlossary;
