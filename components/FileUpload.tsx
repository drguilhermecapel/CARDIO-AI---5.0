
import React, { useCallback, useState } from 'react';

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  isLoading: boolean;
  onValidationError: (msg: string) => void;
}

const FileUpload: React.FC<FileUploadProps> = ({ onFileSelect, isLoading, onValidationError }) => {
  const [dragActive, setDragActive] = useState(false);

  const validateFile = (file: File): boolean => {
    const validMimeTypes = [
      'image/png', 
      'image/jpeg', 
      'image/webp', 
      'image/heic', 
      'image/heif',
      'application/pdf'
    ];
    
    // Extensões para verificação secundária
    const validExtensions = ['.png', '.jpg', '.jpeg', '.webp', '.heic', '.heif', '.pdf'];
    const fileName = file.name.toLowerCase();
    const hasValidExtension = validExtensions.some(ext => fileName.endsWith(ext));
    
    const maxSize = 25 * 1024 * 1024; // 25MB para capturas de alta definição
    
    if (!validMimeTypes.includes(file.type) && !hasValidExtension) {
      onValidationError("FORMAT_ERROR: Formato não suportado. Use PNG, JPEG, WEBP, HEIC ou PDF.");
      return false;
    }
    
    if (file.size > maxSize) {
      onValidationError("OVERFLOW_ERROR: O arquivo excede o limite de segurança de 25MB.");
      return false;
    }
    
    return true;
  };

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(e.type === "dragenter" || e.type === "dragover");
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files?.[0]) {
      const file = e.dataTransfer.files[0];
      if (validateFile(file)) onFileSelect(file);
    }
  }, [onFileSelect]);

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      const file = e.target.files[0];
      if (validateFile(file)) onFileSelect(file);
    }
  }, [onFileSelect]);

  return (
    <div className="w-full max-w-4xl mx-auto px-4">
      <div
        className={`relative group rounded-[3rem] border-2 transition-all duration-1000 overflow-hidden
          ${dragActive ? 'border-cyan-400 bg-cyan-400/10 scale-[1.02]' : 'border-white/5 bg-slate-900/60 hover:border-white/20'}
          glass-card
        `}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        {/* Laser Scanner Animation Overlay */}
        <div className="absolute inset-0 pointer-events-none opacity-20 overflow-hidden">
            <div className={`absolute top-0 left-0 w-full h-1.5 bg-cyan-500 blur-sm ${isLoading ? 'animate-scanner-fast' : 'animate-scanner-v'}`}></div>
        </div>

        <div className="p-16 md:p-24 text-center flex flex-col items-center justify-center min-h-[450px] relative z-20">
          <div className="mb-12 relative flex items-center justify-center">
            <div className={`absolute w-24 h-24 border border-cyan-500/20 rounded-full ${isLoading ? 'animate-ping' : ''}`}></div>
            <div className="relative p-10 bg-black/40 rounded-full border border-white/10 text-cyan-400 group-hover:text-white transition-colors duration-500">
               <svg className={`w-16 h-16 ${isLoading ? 'animate-pulse' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                 <path d="M12 3v13m0 0l-4-4m4 4l4-4M5 20h14" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round" />
               </svg>
            </div>
          </div>
          
          <h3 className="text-4xl md:text-5xl font-black text-white mb-4 tracking-tighter uppercase italic">
            Initialize <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-magenta-400">Bio-Scan</span>
          </h3>
          <p className="text-[10px] text-slate-500 mb-12 max-w-sm font-mono uppercase tracking-[0.4em] leading-loose">
            Supported signals: ECG Scan, Mobile Photo (JPG/PNG/HEIC) or PDF Document
          </p>
          
          <label className={`relative cursor-pointer ${isLoading ? 'pointer-events-none' : ''}`}>
            <div className="absolute inset-0 bg-cyan-500/20 blur-2xl rounded-full opacity-0 group-hover:opacity-100 transition-opacity"></div>
            <span className={`relative text-slate-950 font-black py-6 px-16 rounded-2xl shadow-2xl inline-block transition-all uppercase tracking-[0.5em] text-[10px] ${
              isLoading ? 'bg-slate-700 text-slate-400' : 'bg-white hover:bg-cyan-400 hover:scale-105 active:scale-95'
            }`}>
              {isLoading ? 'Processing Signal...' : 'Upload ECG'}
            </span>
            <input 
              type="file" 
              className="hidden" 
              onChange={handleChange} 
              disabled={isLoading}
              accept=".png,.jpg,.jpeg,.webp,.heic,.heif,.pdf,image/png,image/jpeg,image/webp,image/heic,application/pdf" 
            />
          </label>
          
          <div className="mt-16 grid grid-cols-3 gap-8 w-full border-t border-white/5 pt-10">
             {['MULTI-FORMAT', 'SECURE_RSA', 'MAX_25MB'].map(label => (
               <div key={label} className="text-[8px] font-mono text-slate-600 uppercase tracking-widest">{label}</div>
             ))}
          </div>
        </div>
      </div>
      <style>{`
        @keyframes scanner-v {
          0% { top: 0; }
          100% { top: 100%; }
        }
        @keyframes scanner-fast {
          0% { top: 0; }
          50% { top: 100%; }
          100% { top: 0; }
        }
        .animate-scanner-v { animation: scanner-v 6s linear infinite; }
        .animate-scanner-fast { animation: scanner-fast 1.5s ease-in-out infinite; }
      `}</style>
    </div>
  );
};

export default FileUpload;
