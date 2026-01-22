
import React, { useMemo, useRef, useEffect } from 'react';
import { identifyEcgPattern } from '../utils/cardioLogic';

interface EcgVisualizerProps {
  arrhythmias: string[];
  heartRate: string;
  diagnosis?: string;
}

const LiveWaveform: React.FC<{ pattern: string, bpm: number }> = ({ pattern, bpm }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const baseline = height / 2;
    const speed = 2.5; 
    let buffer: number[] = [];
    let x = 0;
    
    const generateBeat = (): number[] => {
      const beat: number[] = [];
      const pushCurve = (yStart: number, yPeak: number, length: number) => {
         for(let i=0; i<length; i++) {
            const progress = i/length;
            beat.push(yStart + (yPeak - yStart) * Math.sin(progress * Math.PI));
         }
      };
      const pushLine = (y: number, count: number) => { for(let i=0; i<count; i++) beat.push(y); };

      if (pattern === 'wellens') {
        pushCurve(0, -6, 12); // P
        pushLine(0, 15);
        beat.push(5); pushLine(-45, 8); beat.push(8); // QRS
        pushLine(0, 10);
        pushCurve(0, -8, 10); // Wellens Type A biphasic start
        pushCurve(0, 12, 10);  // Wellens Type A biphasic end
        pushLine(0, 10);
      } else if (pattern === 'dewinter') {
        pushCurve(0, -6, 12);
        pushLine(0, 15);
        beat.push(5); pushLine(-45, 8); beat.push(8);
        beat.push(10); // J point depression
        pushLine(10, 5);
        pushCurve(10, -25, 20); // Tall symmetrical T
        pushLine(0, 10);
      } else if (pattern === 'bayessyndrome') {
        pushCurve(0, -6, 12); pushCurve(0, 4, 12); pushLine(0, 15);
        beat.push(5); pushLine(-45, 8); beat.push(8); pushLine(0, 10);
        pushCurve(0, -12, 20); pushLine(0, 10);
      } else if (pattern === 'paced') {
        beat.push(-60); beat.push(20); pushLine(0, 3);
        pushCurve(0, -45, 20); pushCurve(0, 10, 8);
        pushLine(0, 15); pushCurve(0, 15, 25);
      } else {
        pushCurve(0, -6, 12);
        pushLine(0, 15);
        beat.push(5); pushLine(-45, 8); beat.push(8); pushLine(0, 10);
        pushCurve(0, -12, 20); pushLine(0, 10);
      }
      
      return beat;
    };

    let frameId: number;
    const render = () => {
      ctx.clearRect(x, 0, 15, height);
      ctx.fillStyle = '#000';
      ctx.fillRect(x, 0, 15, height);

      if (buffer.length === 0) {
          const delay = (60 / (bpm || 60)) * 40;
          for(let i=0; i<delay; i++) buffer.push(0);
          buffer.push(...generateBeat());
      }

      const val = buffer.shift() || 0;
      const y = baseline + val;
      
      ctx.beginPath();
      ctx.moveTo(x - speed, (canvas as any).prevY || baseline); 
      ctx.lineTo(x, y);
      
      let color = '#4ade80';
      if (['wellens', 'dewinter'].includes(pattern)) color = '#f43f5e';
      if (pattern === 'bayessyndrome') color = '#06b6d4';
      
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.stroke();
      (canvas as any).prevY = y;

      x += speed;
      if (x > width) x = 0;
      frameId = requestAnimationFrame(render);
    };

    render();
    return () => cancelAnimationFrame(frameId);
  }, [pattern, bpm]);

  return (
    <div className="w-full bg-black rounded-t-[2.5rem] relative overflow-hidden h-32 border-b border-white/10">
        <canvas ref={canvasRef} width={1000} height={128} className="w-full h-full" />
    </div>
  );
};

const EcgVisualizer: React.FC<EcgVisualizerProps> = ({ arrhythmias, heartRate, diagnosis }) => {
  const pattern = useMemo(() => identifyEcgPattern(arrhythmias, diagnosis || ''), [arrhythmias, diagnosis]);
  const bpm = useMemo(() => {
      const match = heartRate.match(/(\d+)/);
      return match ? parseInt(match[1], 10) : 60;
  }, [heartRate]);

  return (
    <div className="w-full bg-slate-900 rounded-[3rem] border border-white/5 shadow-2xl overflow-hidden animate-fade-in">
      <LiveWaveform pattern={pattern} bpm={bpm} />
      <div className="p-8">
        <div className="flex justify-between items-start">
            <div>
                <h4 className="text-white font-black uppercase italic text-xl tracking-tighter">Motor de Análise Neural</h4>
                <p className="text-cyan-500 text-[10px] uppercase font-mono mt-1 font-black">PADRÃO: {pattern.toUpperCase()}</p>
            </div>
            <div className="px-4 py-2 bg-cyan-500/10 rounded-2xl border border-cyan-500/20">
                <span className="text-cyan-400 font-mono text-lg font-black">{bpm} BPM</span>
            </div>
        </div>
      </div>
    </div>
  );
};

export default EcgVisualizer;
