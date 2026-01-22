import React, { useState } from 'react';
import { PatientContext } from '../types';

interface PatientFormProps {
  onConfirm: (ctx: PatientContext) => void;
}

const PatientForm: React.FC<PatientFormProps> = ({ onConfirm }) => {
  const [age, setAge] = useState('');
  const [gender, setGender] = useState<'Male' | 'Female' | 'Other'>('Male');
  const [symptoms, setSymptoms] = useState<string[]>([]);
  const [history, setHistory] = useState('');
  const [isSaved, setIsSaved] = useState(false);

  const toggleSymptom = (sym: string) => {
    setSymptoms(prev => 
      prev.includes(sym) ? prev.filter(s => s !== sym) : [...prev, sym]
    );
    setIsSaved(false);
  };

  const handleInputChange = (setter: React.Dispatch<React.SetStateAction<any>>, value: any) => {
    setter(value);
    setIsSaved(false);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onConfirm({
      age,
      gender,
      symptoms: symptoms.length > 0 ? symptoms : ['None'],
      history
    });
    setIsSaved(true);
  };

  const commonSymptoms = ["Chest Pain", "Palpitations", "Syncope", "Shortness of Breath", "Dizziness"];

  return (
    <div className="w-full max-w-3xl mx-auto glass-card rounded-2xl border-white/5 p-8 mb-12 animate-fade-in-up">
      <div className="flex items-center gap-3 mb-8">
        <div className="bg-cyan-500/10 text-cyan-400 p-2.5 rounded-xl border border-cyan-500/20 shadow-inner">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
          </svg>
        </div>
        <div>
          <h3 className="text-sm font-black text-white uppercase tracking-[0.2em]">Subject Profiling</h3>
          <p className="text-[10px] text-slate-500 font-mono uppercase">Neural context input required</p>
        </div>
      </div>
      
      <form onSubmit={handleSubmit} className="space-y-8">
        <div className="grid grid-cols-2 gap-6">
          <div className="group">
            <label className="block text-[10px] font-mono font-bold text-slate-500 uppercase mb-2 tracking-widest group-focus-within:text-cyan-400 transition-colors">Age Parameters</label>
            <input 
              type="number" 
              value={age}
              onChange={(e) => handleInputChange(setAge, e.target.value)}
              placeholder="INT_VAL"
              className="w-full bg-slate-950 border border-white/10 rounded-xl py-3 px-4 text-white font-mono placeholder:text-slate-800 focus:outline-none focus:border-cyan-500 transition-all shadow-inner"
            />
          </div>
          <div className="group">
            <label className="block text-[10px] font-mono font-bold text-slate-500 uppercase mb-2 tracking-widest group-focus-within:text-cyan-400 transition-colors">Biological Gender</label>
            <select 
              value={gender}
              onChange={(e) => handleInputChange(setGender, e.target.value as any)}
              className="w-full bg-slate-950 border border-white/10 rounded-xl py-3 px-4 text-white font-mono focus:outline-none focus:border-cyan-500 transition-all shadow-inner"
            >
              <option value="Male">MALE_REF</option>
              <option value="Female">FEMALE_REF</option>
              <option value="Other">X_NEUTRAL</option>
            </select>
          </div>
        </div>

        <div>
          <label className="block text-[10px] font-mono font-bold text-slate-500 uppercase mb-3 tracking-widest">Active Symptom Array</label>
          <div className="flex flex-wrap gap-2">
            {commonSymptoms.map(sym => (
              <button
                key={sym}
                type="button"
                onClick={() => toggleSymptom(sym)}
                className={`px-4 py-2 rounded-lg text-[10px] font-black tracking-widest uppercase transition-all border ${
                  symptoms.includes(sym) 
                    ? 'bg-cyan-500 text-slate-950 border-cyan-400 shadow-[0_0_15px_rgba(6,182,212,0.3)]' 
                    : 'bg-slate-950 text-slate-500 border-white/10 hover:border-cyan-500/50 hover:text-slate-300'
                }`}
              >
                {sym}
              </button>
            ))}
          </div>
        </div>

        <div>
          <label className="block text-[10px] font-mono font-bold text-slate-500 uppercase mb-3 tracking-widest">Historical Medical Logs</label>
          <textarea
            value={history}
            onChange={(e) => handleInputChange(setHistory, e.target.value)}
            placeholder="Input hypertension, MI, diabetes data points..."
            rows={3}
            className="w-full bg-slate-950 border border-white/10 rounded-xl py-4 px-4 text-white font-mono placeholder:text-slate-800 focus:outline-none focus:border-cyan-500 transition-all shadow-inner resize-none text-xs"
          />
        </div>

        <div className="flex items-center justify-between pt-4 gap-6">
           <div className="flex-1 p-3 rounded-lg border border-white/5 bg-white/5">
             <p className="text-[9px] text-slate-500 italic uppercase leading-relaxed font-mono">
               SYS_NOTE: Contextual data significantly increases the precision of neural ischemia differentiation vs benign repolarization.
             </p>
           </div>
           <button 
             type="submit" 
             className={`px-8 py-3.5 rounded-xl text-[10px] font-black uppercase tracking-[0.3em] transition-all duration-500 ${
               isSaved 
                 ? 'bg-green-500/20 text-green-400 border border-green-500/50 shadow-[0_0_20px_rgba(34,197,94,0.2)]' 
                 : 'bg-white text-slate-950 hover:bg-cyan-400 hover:scale-105 active:scale-95'
             }`}
           >
             {isSaved ? 'LOG_STORED' : 'COMMIT_DATA'}
           </button>
        </div>
      </form>
    </div>
  );
};

export default PatientForm;