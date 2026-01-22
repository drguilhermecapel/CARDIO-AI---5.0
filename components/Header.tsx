import React from 'react';

const Header: React.FC = () => {
  return (
    <header className="glass-card sticky top-0 z-50 border-b border-white/5 backdrop-blur-3xl">
      <div className="max-w-7xl mx-auto px-6">
        <div className="flex justify-between items-center h-20">
          <div className="flex items-center gap-5">
            {/* Bio-Digital Core Icon (SVG) */}
            <div className="relative w-12 h-12 flex items-center justify-center">
              <div className="absolute inset-0 bg-cyan-500/10 rounded-lg rotate-45 border border-cyan-500/20 animate-pulse"></div>
              <svg className="w-8 h-8 text-cyan-400 relative z-10 animate-pulse" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z" />
                <path className="opacity-40" d="M12 11v4M10 13h4" strokeWidth="2" strokeLinecap="round" />
              </svg>
              <div className="absolute -inset-1 border border-cyan-500/10 rounded-lg rotate-12"></div>
            </div>
            
            <div className="flex flex-col">
              <div className="flex items-center gap-2">
                <span className="text-2xl font-black text-white tracking-tighter uppercase italic">Cardio</span>
                <span className="text-2xl font-black text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-magenta-400 neon-text-glow uppercase italic">Neural</span>
              </div>
              <span className="text-[9px] text-cyan-500/50 font-mono font-black uppercase tracking-[0.5em] -mt-1">Bio-Metric Engine v6.0</span>
            </div>
          </div>
          
          <div className="hidden lg:flex items-center gap-12">
             <div className="flex flex-col items-end gap-1">
               <div className="flex gap-1">
                 {[1,2,3,4,5,6,7,8].map(i => (
                   <div key={i} className={`h-1.5 w-6 rounded-sm ${i < 7 ? 'bg-cyan-500/40 shadow-[0_0_8px_rgba(6,182,212,0.3)]' : 'bg-white/5'}`}></div>
                 ))}
               </div>
               <span className="text-[8px] text-slate-500 font-mono uppercase tracking-widest font-black">Memory Integrity 98.4%</span>
             </div>
             
             <div className="h-10 w-px bg-white/10"></div>
             
             <div className="flex items-center gap-4">
                <div className="text-right">
                  <div className="text-[10px] text-white font-black uppercase tracking-widest">Mainframe Status</div>
                  <div className="text-[9px] text-green-400 font-mono uppercase">Link: Stable</div>
                </div>
                <div className="w-10 h-10 rounded-full border border-white/10 flex items-center justify-center bg-slate-900 shadow-inner group cursor-pointer hover:border-cyan-500/50 transition-all">
                   <div className="w-2 h-2 bg-green-500 rounded-full animate-ping"></div>
                </div>
             </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;