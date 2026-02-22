
import React, { useState } from 'react';

interface PaymentModalProps {
  onConfirmPayment: () => void;
  pixKey: string;
}

const PaymentModal: React.FC<PaymentModalProps> = ({ onConfirmPayment, pixKey }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(pixKey);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/80 backdrop-blur-xl animate-fade-in">
      <div className="max-w-lg w-full relative group">
        
        {/* Decorative borders */}
        <div className="absolute -inset-1 bg-gradient-to-r from-red-500 via-orange-500 to-red-500 opacity-50 blur-lg group-hover:opacity-75 transition-opacity duration-1000"></div>
        
        <div className="relative glass-card bg-[#0a0a0a] rounded-[2rem] border border-red-500/30 p-8 md:p-12 overflow-hidden">
          
          {/* Background Scanner Effect */}
          <div className="absolute top-0 left-0 w-full h-1 bg-red-500/20 animate-scanner-fast"></div>

          <div className="flex flex-col items-center text-center">
            
            <div className="w-20 h-20 rounded-2xl bg-red-500/10 border border-red-500/30 flex items-center justify-center mb-6 shadow-[0_0_30px_rgba(239,68,68,0.2)]">
              <svg className="w-10 h-10 text-red-500 animate-pulse" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
              </svg>
            </div>

            <h2 className="text-3xl font-black text-white italic tracking-tighter uppercase mb-2">
              Sistema Bloqueado
            </h2>
            <div className="flex items-center gap-2 mb-6">
              <div className="h-px w-8 bg-red-500/50"></div>
              <span className="text-[10px] font-mono text-red-400 uppercase tracking-[0.3em]">Crédito Gratuito Utilizado</span>
              <div className="h-px w-8 bg-red-500/50"></div>
            </div>

            {/* PREÇO EM DESTAQUE */}
            <div className="mb-8 w-full bg-red-500/5 border border-red-500/20 rounded-2xl p-4 flex flex-col items-center justify-center relative overflow-hidden group-hover:border-red-500/40 transition-colors">
               <div className="absolute inset-0 bg-red-500/5 blur-xl animate-pulse"></div>
               <span className="text-[9px] text-red-300 font-mono uppercase tracking-widest mb-1 relative z-10">Valor da Análise</span>
               <span className="text-5xl font-black text-white tracking-tighter drop-shadow-[0_0_15px_rgba(239,68,68,0.5)] relative z-10">
                 R$ 10,00
               </span>
            </div>

            <p className="text-slate-400 text-sm leading-relaxed mb-8">
              Para processar este eletrocardiograma no <strong>Núcleo de Diagnóstico Neural</strong>, realize o pagamento via PIX.
            </p>

            <div className="w-full bg-white/5 border border-white/10 rounded-xl p-6 mb-8 relative overflow-hidden">
              <div className="absolute top-0 left-0 w-1 h-full bg-gradient-to-b from-transparent via-red-500 to-transparent"></div>
              <p className="text-[10px] text-slate-500 font-mono uppercase tracking-widest mb-3">Chave de Acesso (PIX)</p>
              
              <div className="flex items-center gap-3 bg-black/40 rounded-lg p-3 border border-white/5 cursor-pointer hover:border-white/20 transition-all" onClick={handleCopy}>
                <code className="text-cyan-400 font-mono text-sm flex-1 truncate select-all">
                  {pixKey}
                </code>
                <div className={`text-[10px] font-bold px-2 py-1 rounded ${copied ? 'bg-green-500 text-black' : 'bg-white/10 text-white'}`}>
                  {copied ? 'COPIADO' : 'COPIAR'}
                </div>
              </div>
            </div>

            <button 
              onClick={onConfirmPayment}
              className="w-full py-5 bg-gradient-to-r from-red-600 to-orange-600 rounded-xl text-white font-black text-xs uppercase tracking-[0.3em] hover:brightness-110 active:scale-[0.98] transition-all shadow-xl shadow-red-900/20 group-hover:shadow-red-500/20"
            >
              Confirmar Pagamento e Liberar
            </button>
            
            <p className="mt-6 text-[9px] text-slate-600 font-mono">
              GATEWAY_SEGURO // ID: {Math.random().toString(36).substr(2, 9).toUpperCase()}
            </p>

          </div>
        </div>
      </div>
    </div>
  );
};

export default PaymentModal;
