
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

  const clinicalFlags = [
    { id: "Chest Pain", label: "DOR TORÁCICA (ANGINA)" },
    { id: "Syncope", label: "SÍNCOPE / DESMAIO" },
    { id: "Palpitations", label: "PALPITAÇÕES" },
    { id: "Dyspnea", label: "DISPNEIA (FALTA AR)" },
    { id: "Dizziness", label: "TONTURA / PRÉ-SÍNCOPE" },
    { id: "Fatigue", label: "FADIGA EXTREMA" },
    { id: "Edema", label: "EDEMA (INCHAÇO)" },
    { id: "Family Hx", label: "HIST. FAM. MORTE SÚBITA" },
    { id: "Prev MI", label: "IAM PRÉVIO / STENT" },
    { id: "Pacemaker", label: "PORTADOR DE CDI/MP" }
  ];

  return (
    <div className="w-full animate-fade-in-up">
      <div className="flex items-center gap-2 mb-6 opacity-70">
        <div className="h-2 w-2 bg-cyan-500 rounded-full animate-pulse"></div>
        <h3 className="text-[10px] font-black text-cyan-400 uppercase tracking-[0.2em]">Clinical Context Parameters</h3>
      </div>
      
      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="grid grid-cols-2 gap-4">
          <div className="group">
            <input 
              type="number" 
              value={age}
              onChange={(e) => handleInputChange(setAge, e.target.value)}
              placeholder="AGE (YRS)"
              className="w-full bg-slate-900/50 border border-white/5 rounded-lg py-2 px-3 text-white text-xs font-mono placeholder:text-slate-600 focus:outline-none focus:border-cyan-500/50 focus:bg-slate-900 transition-all"
            />
          </div>
          <div className="group">
            <select 
              value={gender}
              onChange={(e) => handleInputChange(setGender, e.target.value as any)}
              className="w-full bg-slate-900/50 border border-white/5 rounded-lg py-2 px-3 text-white text-xs font-mono focus:outline-none focus:border-cyan-500/50 focus:bg-slate-900 transition-all appearance-none"
            >
              <option value="Male">MALE_XY</option>
              <option value="Female">FEMALE_XX</option>
              <option value="Other">OTHER</option>
            </select>
          </div>
        </div>

        <div>
          <label className="block text-[8px] font-mono font-bold text-slate-500 uppercase mb-3 tracking-widest flex justify-between">
            <span>Active Flags / Symptoms</span>
            <span className="text-cyan-500/50">{symptoms.length} SELECTED</span>
          </label>
          <div className="grid grid-cols-2 gap-2">
            {clinicalFlags.map(flag => (
              <button
                key={flag.id}
                type="button"
                onClick={() => toggleSymptom(flag.id)}
                className={`px-3 py-2 rounded text-[8px] font-bold tracking-widest uppercase transition-all border text-left flex items-center justify-between ${
                  symptoms.includes(flag.id) 
                    ? 'bg-cyan-500/10 text-cyan-400 border-cyan-500/40 shadow-[0_0_10px_rgba(6,182,212,0.1)]' 
                    : 'bg-white/5 text-slate-500 border-white/5 hover:border-white/20 hover:text-slate-300'
                }`}
              >
                <span>{flag.label}</span>
                {symptoms.includes(flag.id) && <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full shadow-[0_0_5px_rgba(6,182,212,0.8)]"></div>}
              </button>
            ))}
          </div>
        </div>

        <div>
          <textarea
            value={history}
            onChange={(e) => handleInputChange(setHistory, e.target.value)}
            placeholder="Additional Clinical History (Meds, comorbidities)..."
            rows={2}
            className="w-full bg-slate-900/50 border border-white/5 rounded-lg py-2 px-3 text-white font-mono placeholder:text-slate-700 focus:outline-none focus:border-cyan-500/50 focus:bg-slate-900 transition-all resize-none text-[10px]"
          />
        </div>

        <button 
          type="submit" 
          className={`w-full py-3 rounded-lg text-[9px] font-black uppercase tracking-[0.3em] transition-all duration-300 border ${
            isSaved 
              ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30' 
              : 'bg-white/5 text-slate-400 border-white/10 hover:bg-white/10 hover:text-white hover:border-cyan-500/30'
          }`}
        >
          {isSaved ? 'CONTEXT_LOCKED' : 'SYNC CONTEXT'}
        </button>
      </form>
    </div>
  );
};

export default PatientForm;
