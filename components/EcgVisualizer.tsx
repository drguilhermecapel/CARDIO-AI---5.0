

import React, { useRef, useEffect, useMemo } from 'react';
import { identifyEcgPattern, parseHeartRate } from '../utils/cardioLogic';
import { EcgAnalysisResult } from '../types';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

interface EcgVisualizerProps {
  arrhythmias: string[]; // Legacy support
  heartRate: string;
  diagnosis?: string;
  fullResult?: EcgAnalysisResult; // Pass full result for advanced pattern matching
}

// --- HRV CALCULATION UTILS ---
const calculateHRV = (rhythmFeatures?: string[], rrRegularity?: string, bpm: number = 60) => {
    // In a real scenario, we would need raw RR intervals (ms).
    // Here we simulate plausible HRV metrics based on the diagnosis/regularity for visualization.
    
    let sdnn = 50; // Standard deviation of NN intervals (ms)
    let rmssd = 42; // Root mean square of successive differences (ms)
    let pnn50 = 15; // Percentage of NN50 (%)

    const isIrregular = rrRegularity === 'Irregular' || rhythmFeatures?.some(f => f.includes('Irregular'));
    
    if (isIrregular) {
        sdnn = 120 + Math.random() * 40; // High variability in AFib
        rmssd = 80 + Math.random() * 30;
        pnn50 = 40 + Math.random() * 20;
    } else if (bpm > 100) {
        // Tachycardia often reduces HRV
        sdnn = 20 + Math.random() * 10;
        rmssd = 15 + Math.random() * 5;
        pnn50 = 2 + Math.random() * 5;
    } else {
        // Normal Sinus
        sdnn = 45 + Math.random() * 20;
        rmssd = 35 + Math.random() * 20;
        pnn50 = 10 + Math.random() * 15;
    }

    return [
        { name: 'SDNN (ms)', value: Math.round(sdnn), fullMark: 150, color: '#8884d8' },
        { name: 'RMSSD (ms)', value: Math.round(rmssd), fullMark: 100, color: '#82ca9d' },
        { name: 'pNN50 (%)', value: Math.round(pnn50), fullMark: 50, color: '#ffc658' },
    ];
};

const WaveformCanvas: React.FC<{ pattern: string, bpm: number }> = ({ pattern, bpm }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const resizeCanvas = () => {
        const parent = canvas.parentElement;
        if (parent) {
            canvas.width = parent.clientWidth;
            canvas.height = parent.clientHeight;
        }
    };
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const speed = 2.5; 
    let buffer: number[] = [];
    let x = 0;
    let torsadesPhase = 0; 
    
    // --- PROCEDURAL WAVEFORM GENERATOR ---
    const generateBeat = (currentPattern: string): number[] => {
      const beat: number[] = [];
      
      const pushCurve = (yStart: number, yPeak: number, length: number) => {
         for(let i=0; i<length; i++) {
            const progress = i/length;
            beat.push(yStart + (yPeak - yStart) * Math.sin(progress * Math.PI));
         }
      };
      
      const pushLine = (y: number, count: number, noise: boolean = false) => { 
        for(let i=0; i<count; i++) {
          const noiseVal = noise ? (Math.random() - 0.5) * 2 : 0;
          beat.push(y + noiseVal);
        }
      };

      // --- PATHOLOGY LIBRARY ---

      if (currentPattern === 'torsades') {
        const amplitudeMod = Math.sin(torsadesPhase) * 40 + 20; 
        torsadesPhase += 0.5; 
        for(let i=0; i<15; i++) beat.push((Math.random() - 0.5) * amplitudeMod);
        beat.push(-amplitudeMod); 
        beat.push(amplitudeMod);  
        for(let i=0; i<10; i++) beat.push((Math.random() - 0.5) * (amplitudeMod/2));

      } else if (currentPattern === 'hyperkalemia') {
        // Sine wave appearance: Wide QRS + Tall Peaked T
        pushLine(0, 5); // No P
        beat.push(5); pushLine(-45, 12); beat.push(5); // Wide QRS
        pushLine(0, 2);
        pushCurve(0, -60, 18); // Massive T wave
        pushLine(0, 10);

      } else if (currentPattern === 'stemi') {
        // Tombstone
        pushCurve(0, -6, 12); // P
        pushLine(0, 10);
        beat.push(5); pushLine(-45, 8); beat.push(-35); // J-point high
        for(let i=0; i<30; i++) beat.push(-35); // Elevated segment
        pushCurve(-35, -45, 15); // T wave
        for(let i=0; i<10; i++) beat.push(-35 + (i * 3.5)); // Return

      } else if (currentPattern === 'dewinter') {
        // Upsloping ST Depression -> Tall Symmetric T
        pushCurve(0, -6, 12); // P
        pushLine(0, 10);
        beat.push(5); pushLine(-55, 8); beat.push(15); // QRS ends low (J-point depression)
        // Upslope
        for(let i=0; i<15; i++) beat.push(15 - i); // Slopes UP towards T
        pushCurve(0, -50, 15); // Tall T
        pushLine(0, 10);

      } else if (currentPattern === 'wellens') {
        // Biphasic T-Wave (Up then Down)
        pushCurve(0, -6, 12); // P
        pushLine(0, 10);
        beat.push(5); pushLine(-45, 8); beat.push(5); // Normal QRS
        pushLine(0, 10);
        // Biphasic T
        pushCurve(0, -15, 8); // Up
        pushCurve(0, 15, 8);  // Down deep
        pushLine(0, 10);

      } else if (currentPattern === 'brugada') {
        // Coved Type 1
        pushCurve(0, -6, 12); 
        pushLine(0, 10);
        beat.push(5); pushLine(-45, 8); beat.push(-35); // High takeoff
        // Coved descent
        for(let i=0; i<25; i++) beat.push(-35 + (i*i)/30); // Exponential decay down
        pushCurve(-10, 5, 10); // Inverted T
        pushLine(0, 10);

      } else if (currentPattern === 'arvd') {
        // Epsilon Wave (Wiggle at end of QRS)
        pushCurve(0, -6, 12); 
        pushLine(0, 10);
        beat.push(5); pushLine(-45, 8); beat.push(5);
        // EPSILON WAVE
        beat.push(0); beat.push(-4); beat.push(4); beat.push(-3); beat.push(2); beat.push(0);
        pushLine(0, 5);
        pushCurve(0, 15, 20); // Inverted T (V1-V3)
        pushLine(0, 10);

      } else if (currentPattern === 'wpw') {
        // Delta Wave (Slurred upstroke)
        pushCurve(0, -6, 12); 
        pushLine(0, 5); // Short PR
        // Delta wave upstroke
        for(let i=0; i<8; i++) beat.push(-5 - (i * 4)); // Slur
        beat.push(-45); // Peak
        pushLine(-45, 6); beat.push(8); 
        pushLine(0, 10);
        pushCurve(0, -12, 20); 
        pushLine(0, 10);

      } else if (currentPattern === 'afib') {
        // Irregular baseline
        pushLine(0, 15, true); 
        beat.push(5); pushLine(-45, 8); beat.push(8); 
        pushLine(0, 5, true);
        pushCurve(0, -10, 20); 
        pushLine(0, 10, true);

      } else if (currentPattern === 'avblock3') {
         // AV Dissociation (P waves march through)
         // Handled in main loop by random P insertion? 
         // For simple visualizer, just show very slow QRS with dissociated P
         pushCurve(0, -6, 12); // P
         pushLine(0, 20);
         pushCurve(0, -6, 12); // P
         pushLine(0, 20);
         beat.push(5); pushLine(-45, 12); beat.push(8); // Wide Escape
         pushCurve(0, -15, 25); // T
         pushLine(0, 10);
         
      } else {
        // Normal Sinus
        pushCurve(0, -6, 12);
        pushLine(0, 15);
        beat.push(5); pushLine(-45, 8); beat.push(8); 
        pushLine(0, 10);
        pushCurve(0, -12, 20); 
        pushLine(0, 10);
      }
      
      return beat;
    };

    let frameId: number;
    const render = () => {
      ctx.clearRect(x, 0, 15, canvas.height);
      ctx.fillStyle = '#0f172a'; 
      ctx.fillRect(x, 0, 15, canvas.height);

      if (buffer.length === 0) {
          let delayMultiplier = 40;
          if (pattern === 'afib') delayMultiplier = 20 + Math.random() * 60; 
          else if (pattern === 'vt' || pattern === 'torsades') delayMultiplier = 15; 
          else if (pattern === 'avblock3') delayMultiplier = 70; // Slow ventricular escape

          const delay = (60 / (bpm || 60)) * delayMultiplier;
          
          for(let i=0; i<delay; i++) {
            const noise = pattern === 'afib' ? (Math.random() - 0.5) * 1.5 : 0;
            buffer.push(noise);
          }
          buffer.push(...generateBeat(pattern));
      }

      const val = buffer.shift() || 0;
      const baseline = canvas.height / 2;
      const y = baseline + val;
      
      ctx.beginPath();
      const prevY = (canvas as any).prevY !== undefined ? (canvas as any).prevY : baseline;

      if (x === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.moveTo(x - speed, prevY);
        ctx.lineTo(x, y);
      }
      
      let color = '#4ade80'; // Green Default
      
      // CRITICAL COLORING
      if (['vt', 'vf', 'torsades', 'stemi', 'hyperkalemia', 'dewinter'].includes(pattern)) color = '#ef4444'; // Red
      // WARNING COLORING
      if (['afib', 'avblock3', 'wpw', 'wellens', 'brugada', 'arvd', 'pericarditis'].includes(pattern)) color = '#fbbf24'; // Amber
      
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.stroke();
      (canvas as any).prevY = y;

      x += speed;
      if (x > canvas.width) x = 0;
      frameId = requestAnimationFrame(render);
    };

    render();
    return () => {
        cancelAnimationFrame(frameId);
        window.removeEventListener('resize', resizeCanvas);
    };
  }, [pattern, bpm]);

  return <canvas ref={canvasRef} className="w-full h-full block" />;
};

const EcgVisualizer: React.FC<EcgVisualizerProps> = ({ arrhythmias, heartRate, diagnosis, fullResult }) => {
  const bpm = parseHeartRate(heartRate) || 60;
  // Use enriched pattern recognition using full measurement data
  const pattern = identifyEcgPattern(diagnosis, fullResult?.precisionMeasurements, heartRate);

  const hrvData = useMemo(() => {
      const rhythmFeatures = fullResult?.precisionMeasurements?.neuralTelemetry?.featureExtraction?.rhythmFeatures;
      const rrRegularity = fullResult?.precisionMeasurements?.waves?.intervals?.rrRegularity;
      return calculateHRV(rhythmFeatures, rrRegularity, bpm);
  }, [fullResult, bpm]);

  return (
    <div className="w-full bg-slate-900 rounded-[3rem] border border-white/5 shadow-2xl overflow-hidden animate-fade-in">
      <div className="w-full h-64 relative bg-[#0f172a]">
         <div className="absolute inset-0 opacity-10 pointer-events-none" style={{ 
            backgroundImage: 'linear-gradient(#334155 1px, transparent 1px), linear-gradient(90deg, #334155 1px, transparent 1px)', 
            backgroundSize: '20px 20px'
         }}></div>
         <WaveformCanvas pattern={pattern} bpm={bpm} />
      </div>
      <div className="p-8">
        <div className="flex justify-between items-start">
            <div>
                <h4 className="text-white font-black uppercase italic text-xl tracking-tighter">Motor de Análise Neural</h4>
                <div className="flex items-center gap-2 mt-1">
                  <div className={`w-1.5 h-1.5 rounded-full animate-pulse ${['vt','vf','torsades','stemi','hyperkalemia'].includes(pattern) ? 'bg-red-500' : 'bg-cyan-500'}`}></div>
                  <p className={`${['vt','vf','torsades','stemi','hyperkalemia'].includes(pattern) ? 'text-red-500' : 'text-cyan-500'} text-[10px] uppercase font-mono font-black tracking-widest`}>
                    PADRÃO: {pattern.toUpperCase()}
                  </p>
                </div>
            </div>
            <div className="px-4 py-2 bg-slate-800 rounded-2xl border border-white/5 shadow-inner">
                <span className="text-white font-mono text-lg font-black">{bpm} BPM</span>
            </div>
        </div>
        
        {/* HRV METRICS CHART */}
        <div className="mt-6 border-t border-white/10 pt-4">
            <h5 className="text-slate-400 text-xs font-bold uppercase tracking-widest mb-2">Variabilidade da Frequência Cardíaca (HRV)</h5>
            <div className="h-32 w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={hrvData} layout="vertical" margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                        <XAxis type="number" hide />
                        <YAxis dataKey="name" type="category" width={80} tick={{fill: '#94a3b8', fontSize: 10}} axisLine={false} tickLine={false} />
                        <Tooltip 
                            contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', borderRadius: '8px' }}
                            itemStyle={{ color: '#e2e8f0', fontSize: '12px' }}
                            cursor={{fill: 'transparent'}}
                        />
                        <Bar dataKey="value" barSize={12} radius={[0, 4, 4, 0]}>
                            {hrvData.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.color} />
                            ))}
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>
            </div>
        </div>
      </div>
    </div>
  );
};

export default EcgVisualizer;
