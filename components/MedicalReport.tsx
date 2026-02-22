
import React from 'react';
import { EcgAnalysisResult, PatientContext } from '../types';

interface MedicalReportProps {
  result: EcgAnalysisResult;
  imagePreview: string;
  patientContext?: PatientContext;
}

const MedicalReport = React.forwardRef<HTMLDivElement, MedicalReportProps>(({ result, imagePreview, patientContext }, ref) => {
  const m = result.precisionMeasurements;
  const today = new Date().toLocaleDateString('pt-BR');
  const time = new Date().toLocaleTimeString('pt-BR');

  // Helper for axis quadrant
  const getAxisQuadrant = (axis?: number) => {
    if (axis === undefined) return "N/A";
    if (axis >= -30 && axis <= 90) return "Normal";
    if (axis < -30 && axis >= -90) return "LAD (Left)";
    if (axis > 90 && axis <= 180) return "RAD (Right)";
    return "Extreme (NW)";
  };

  return (
    <div ref={ref} className="bg-white text-black font-sans box-border" style={{ 
      width: '210mm', 
      minHeight: '297mm', 
      padding: '10mm 15mm',
      margin: '0 auto',
      fontSize: '11px',
      lineHeight: '1.4'
    }}>
      
      {/* HEADER: HOSPITAL STANDARD */}
      <div className="border-b-2 border-black pb-4 mb-4 flex justify-between">
        <div className="flex gap-4">
           <div className="w-12 h-12 bg-slate-900 text-white flex items-center justify-center">
              <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                 <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
              </svg>
           </div>
           <div>
              <h1 className="text-xl font-bold uppercase tracking-tight text-slate-900">CardioAI Diagnostic Center</h1>
              <p className="text-[9px] uppercase tracking-widest text-slate-500">Universal Electrocardiography Compendium Report</p>
           </div>
        </div>
        <div className="text-right">
           <div className={`font-mono font-bold text-lg ${result.urgency === 'Emergency' ? 'text-red-600' : 'text-black'}`}>{result.urgency.toUpperCase()}</div>
           <div className="text-[10px]">Report ID: {result.id?.substring(0,8).toUpperCase()}</div>
           <div className="text-[10px]">{today} {time}</div>
        </div>
      </div>

      {/* PATIENT DEMOGRAPHICS */}
      <div className="grid grid-cols-4 gap-4 mb-4 border-b border-gray-300 pb-2 bg-gray-50 p-2 rounded">
         <div>
            <span className="block text-[9px] uppercase text-gray-500 font-bold">Patient Name / ID</span>
            <span className="font-semibold">{result.id || "Unknown"}</span>
         </div>
         <div>
            <span className="block text-[9px] uppercase text-gray-500 font-bold">Age / Gender</span>
            <span className="font-semibold">{patientContext?.age || "--"} / {patientContext?.gender || "--"}</span>
         </div>
         <div>
            <span className="block text-[9px] uppercase text-gray-500 font-bold">Clinical Indication</span>
            <span className="font-semibold truncate block">{patientContext?.symptoms?.join(", ") || "Routine Analysis"}</span>
         </div>
         <div>
            <span className="block text-[9px] uppercase text-gray-500 font-bold">Tech Quality</span>
            <span className="font-semibold">{result.technicalQuality.overallScore}/10 ({result.technicalQuality.leadPlacementValidation})</span>
         </div>
      </div>

      {/* PRECISION MATRIX */}
      <div className="flex gap-6 mb-6">
         {/* Left Column: Intervals */}
         <div className="w-1/3">
            <h3 className="text-[9px] font-bold uppercase mb-1 border-b border-black">Precision Metrics</h3>
            <div className="grid grid-cols-2 text-[10px] gap-y-1">
               <div className="text-gray-600">Heart Rate</div><div className="font-mono font-bold text-right">{result.heartRate} BPM</div>
               <div className="text-gray-600">PR Interval</div><div className="font-mono font-bold text-right">{m.prIntervalMs} ms</div>
               <div className="text-gray-600">QRS Duration</div><div className="font-mono font-bold text-right">{m.qrsComplex.durationMs} ms</div>
               <div className="text-gray-600">QT / QTc (Fridericia)</div><div className="font-mono font-bold text-right">{m.qtAnalysis?.qtInterval || m.qtIntervalMs}/{m.qtAnalysis?.qtcInterval || m.qtcIntervalMs} ms</div>
               <div className="text-gray-600">P-Wave Axis</div><div className="font-mono font-bold text-right">{m.axis?.pAxis ?? '--'}°</div>
               <div className="text-gray-600">QRS Axis</div><div className="font-mono font-bold text-right">{m.axis?.qrsAxis ?? m.axis?.qrsAxisDegrees ?? '--'}°</div>
            </div>
         </div>
         
         {/* Middle Column: OMI Analysis */}
         <div className="w-1/3">
            <h3 className="text-[9px] font-bold uppercase mb-1 border-b border-black">Ischemia / OMI Protocol</h3>
            <div className="grid grid-cols-2 text-[10px] gap-y-1">
               <div className="text-gray-600">ST Trend</div><div className={`font-bold text-right ${m.ischemiaAnalysis?.stSegmentTrend === 'Elevation' ? 'text-red-600' : 'text-black'}`}>{m.ischemiaAnalysis?.stSegmentTrend || 'Neutral'}</div>
               <div className="text-gray-600">Affected Territory</div><div className="font-bold text-right">{m.ischemiaAnalysis?.affectedWall || 'None'}</div>
               <div className="text-gray-600">Culprit Artery</div><div className="font-bold text-right">{m.ischemiaAnalysis?.culpritArtery || 'N/A'}</div>
               <div className="text-gray-600">Reciprocal Changes</div><div className="font-bold text-right">{m.ischemiaAnalysis?.reciprocalChangesFound ? 'Yes' : 'No'}</div>
               <div className="text-gray-600">Sgarbossa Score</div><div className="font-mono font-bold text-right">{m.ischemiaAnalysis?.sgarbossaScore || 0}</div>
            </div>
         </div>

         {/* Right Column: Morphology & Structure */}
         <div className="w-1/3">
            <h3 className="text-[9px] font-bold uppercase mb-1 border-b border-black">Structure & Conduction</h3>
            <div className="grid grid-cols-2 text-[10px] gap-y-1">
               <div className="text-gray-600">LVH / RVH</div><div className="font-bold text-right">{m.structuralAnalysis?.lvhDetected ? 'LVH' : (m.structuralAnalysis?.rvhDetected ? 'RVH' : 'None')}</div>
               <div className="text-gray-600">Bundle Block</div><div className="font-bold text-right">{m.conductionAnalysis?.ivcdType === 'None' ? 'No' : m.conductionAnalysis?.ivcdType}</div>
               <div className="text-gray-600">Fascicular</div><div className="font-bold text-right">{m.conductionAnalysis?.fascicularBlock === 'None' ? 'No' : m.conductionAnalysis?.fascicularBlock}</div>
               <div className="text-gray-600">Morphology V1</div><div className="font-mono font-bold text-right">{m.qrsComplex.morphologyV1}</div>
               <div className="text-gray-600">Morphology V6</div><div className="font-mono font-bold text-right">{m.qrsComplex.morphologyV6}</div>
            </div>
         </div>
      </div>

      {/* UNIVERSAL COMPENDIUM FINDINGS */}
      <div className="mb-6 p-4 border border-gray-200 bg-gray-50 rounded">
         <h3 className="text-[10px] font-bold uppercase text-slate-700 mb-2 border-b border-gray-300">Detailed Compendium Findings</h3>
         <div className="grid grid-cols-2 gap-4 text-[10px]">
            <div>
               <ul className="list-disc list-inside space-y-1">
                  {m.ischemiaAnalysis?.wellensSyndrome !== 'None' && <li className="text-red-700 font-bold">Wellens Syndrome Detected: {m.ischemiaAnalysis?.wellensSyndrome}</li>}
                  {m.ischemiaAnalysis?.deWinterPattern && <li className="text-red-700 font-bold">De Winter T-Waves (LAD Occlusion Equivalent)</li>}
                  {m.conductionAnalysis?.wpwPattern && <li className="text-amber-700 font-bold">Delta Waves / WPW Pattern Detected</li>}
                  {m.neuralTelemetry?.differentialDiagnoses?.slice(0,2).map((d, i) => (
                      <li key={i}>{d.diagnosis} ({d.probability}%)</li>
                  ))}
               </ul>
            </div>
            <div>
               <ul className="list-disc list-inside space-y-1">
                  {m.structuralAnalysis?.atrialEnlargement !== 'None' && <li>Atrial Enlargement: {m.structuralAnalysis?.atrialEnlargement}</li>}
                  {m.waves?.tWave?.morphology !== 'Normal' && <li>T-Wave Morphology: {m.waves?.tWave?.morphology}</li>}
                  {m.waves?.pWave?.morphology !== 'Sinus' && <li>P-Wave: {m.waves?.pWave?.morphology}</li>}
               </ul>
            </div>
         </div>
      </div>

      {/* DIAGNOSTIC INTERPRETATION */}
      <div className="mb-6 border-2 border-black p-4 bg-white">
         <h3 className="text-[10px] font-bold uppercase text-gray-500 mb-2">Primary Diagnosis & Reasoning</h3>
         <div className="text-xl font-black uppercase leading-tight mb-2 tracking-tight">
            {result.diagnosis}
         </div>
         <div className="text-xs text-gray-800 font-medium font-mono border-t border-gray-100 pt-2">
            "{result.clinicalReasoning}"
         </div>
      </div>

      {/* ECG STRIP SNAPSHOT */}
      <div className="mb-4">
         <h3 className="text-[9px] font-bold uppercase mb-1">Visual Evidence (Lead II Rhythm Strip)</h3>
         <div className="h-32 w-full border border-gray-300 overflow-hidden relative">
            <img src={imagePreview} className="w-full h-full object-cover object-center grayscale contrast-125" alt="Trace" />
            <div className="absolute bottom-1 right-1 text-[8px] bg-white px-1 border border-gray-200 font-mono">25mm/s 10mm/mV</div>
         </div>
      </div>

      {/* CLINICAL IMPLICATIONS */}
      <div className="mb-6">
         <h4 className="text-[9px] font-bold uppercase mb-1 border-b border-gray-200">Clinical Management Implications</h4>
         <div className="grid grid-cols-2 gap-4">
             <ul className="text-[10px] list-none space-y-1">
                {result.clinicalImplications.map((imp, i) => (
                   <li key={i} className="flex gap-2">
                     <span className="text-slate-400">•</span>
                     <span>{imp}</span>
                   </li>
                ))}
             </ul>
             <div className="text-[9px] text-gray-500 bg-gray-50 p-2 rounded">
                <strong>Guideline References:</strong> {result.guidelineReferences?.join(', ') || 'AHA/ACC 2022, ESC 2023'}
             </div>
         </div>
      </div>

      {/* FOOTER */}
      <div className="mt-auto pt-4 border-t-2 border-black">
         <div className="flex justify-between items-end">
            <div className="text-[8px] text-gray-500 w-2/3 text-justify">
               <p className="font-bold text-black mb-1">COMPUTER ASSISTED INTERPRETATION // PHYSICIAN REVIEW REQUIRED</p>
               <p>Analysis performed by CardioAI Nexus v10.0 (Universal Compendium). This report is a decision support tool generated by AI and must be verified by a qualified cardiologist. Sensitivity/Specificity varies by signal quality.</p>
            </div>
            <div className="text-right">
               <div className="h-8 mb-1 flex justify-end">
                  <div className="font-script text-xl opacity-80">CardioAI_Nexus</div>
               </div>
               <div className="text-[9px] font-bold border-t border-gray-400 pt-1 uppercase">
                  Electronically Signed: {result.id}
               </div>
            </div>
         </div>
      </div>

    </div>
  );
});

export default MedicalReport;
