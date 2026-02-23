
import React, { useRef, useEffect, useMemo } from 'react';
import * as THREE from 'three';
import { identifyEcgPattern } from '../utils/cardioLogic';
import { StructuralAnalysis, IschemiaAnalysis, ConductionAnalysis } from '../types';

interface Heart3DProps {
  heartRate: number;
  urgency?: 'Emergency' | 'Urgent' | 'Routine' | 'Low';
  diagnosis?: string;
  rhythm?: string;
  numericAxis?: number;
  prIntervalMs?: number;
  qrsDurationMs?: number;
  structural?: StructuralAnalysis;
  ischemia?: IschemiaAnalysis;
  conduction?: ConductionAnalysis;
}

const Heart3D: React.FC<Heart3DProps> = ({ 
  heartRate, 
  urgency = 'Routine', 
  diagnosis, 
  rhythm,
  numericAxis = 60,
  prIntervalMs,
  qrsDurationMs,
  structural,
  ischemia,
  conduction
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const heartGroupRef = useRef<THREE.Group | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const materialsRef = useRef<{ [key: string]: THREE.MeshPhysicalMaterial }>({});

  const pattern = useMemo(() => {
    const partialMeasurements: any = {
      ischemiaAnalysis: ischemia,
      structuralAnalysis: structural,
      conductionAnalysis: conduction,
      waves: { intervals: { prMs: prIntervalMs }, qrsComplex: { durationMs: qrsDurationMs } },
      qrsComplex: { durationMs: qrsDurationMs }
    };
    return identifyEcgPattern(diagnosis || rhythm || '', partialMeasurements, heartRate.toString());
  }, [rhythm, diagnosis, prIntervalMs, qrsDurationMs, heartRate, ischemia, structural, conduction]);

  // --- ANATOMICAL COLORS ---
  const COLORS = {
    MUSCLE: 0xcc4444,     // Healthy Myocardium
    VEIN: 0x3b82f6,       // Vena Cava / Pulmonary Art
    ARTERY: 0xef4444,     // Aorta
    ISCHEMIA: 0x4a0404,   // Necrotic/Ischemic Tissue (Dark Red/Black)
    INJURY: 0xa855f7,     // Acute Injury (Purple tint)
    FAT: 0xffeebb         // Epicardial fat hint
  };

  useEffect(() => {
    if (!containerRef.current) return;

    // --- INIT THREEJS ---
    const width = containerRef.current.clientWidth;
    const height = containerRef.current.clientHeight;
    
    const scene = new THREE.Scene();
    sceneRef.current = scene;
    
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(0, 0, 12);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(width, height);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // --- MATERIALS ---
    const muscleMat = new THREE.MeshPhysicalMaterial({
      color: COLORS.MUSCLE,
      roughness: 0.6,
      metalness: 0.1,
      clearcoat: 0.3,
      clearcoatRoughness: 0.5,
      sheen: 0.5,
      sheenColor: 0xffaaaa,
    });

    const vesselMat = new THREE.MeshPhysicalMaterial({
      color: COLORS.ARTERY,
      roughness: 0.4,
      metalness: 0.2,
      clearcoat: 0.8
    });

    const venousMat = new THREE.MeshPhysicalMaterial({
      color: COLORS.VEIN,
      roughness: 0.5,
      metalness: 0.1
    });

    materialsRef.current = { muscle: muscleMat, vessel: vesselMat, venous: venousMat };

    // --- GEOMETRY CONSTRUCTION (COMPOSITE HEART) ---
    const heartGroup = new THREE.Group();
    heartGroupRef.current = heartGroup;
    scene.add(heartGroup);

    // 1. LEFT VENTRICLE (The Powerhouse)
    // Elongated, conical shape
    const lvScale = structural?.lvhDetected ? 1.4 : 1.0; // Hypertrophy Logic
    const lvGeom = new THREE.SphereGeometry(1.4 * lvScale, 64, 64);
    lvGeom.scale(0.8, 1.4, 0.8); // More conical
    const lvMesh = new THREE.Mesh(lvGeom, muscleMat.clone());
    lvMesh.position.set(0.2, -0.6, -0.2); // Apex points down, left, anterior
    lvMesh.rotation.z = -0.5;
    lvMesh.rotation.x = 0.2;
    lvMesh.castShadow = true;
    lvMesh.receiveShadow = true;
    lvMesh.userData = { name: 'LV' }; 
    heartGroup.add(lvMesh);

    // 2. RIGHT VENTRICLE (The Crescent)
    // Wraps around the anterior/right aspect of LV
    const rvScale = structural?.rvhDetected ? 1.3 : 1.0;
    const rvGeom = new THREE.SphereGeometry(1.2 * rvScale, 54, 54);
    rvGeom.scale(0.9, 1.2, 0.6);
    const rvMesh = new THREE.Mesh(rvGeom, muscleMat.clone());
    rvMesh.position.set(-0.7, -0.3, 0.5); 
    rvMesh.rotation.z = 0.3;
    rvMesh.rotation.y = 0.4;
    rvMesh.castShadow = true;
    rvMesh.userData = { name: 'RV' };
    heartGroup.add(rvMesh);

    // 3. LEFT ATRIUM (Posterior)
    const laGeom = new THREE.SphereGeometry(0.75, 32, 32);
    laGeom.scale(1.1, 0.9, 0.9);
    const laMesh = new THREE.Mesh(laGeom, muscleMat.clone());
    laMesh.position.set(0.6, 1.1, -0.8);
    laMesh.userData = { name: 'LA' };
    heartGroup.add(laMesh);

    // 4. RIGHT ATRIUM (Right Superior)
    const raGeom = new THREE.SphereGeometry(0.8, 32, 32);
    raGeom.scale(0.9, 1.1, 0.9);
    const raMesh = new THREE.Mesh(raGeom, muscleMat.clone());
    raMesh.position.set(-0.9, 1.0, 0.1);
    raMesh.userData = { name: 'RA' };
    heartGroup.add(raMesh);

    // 5. AORTA (The Arch)
    const aortaCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0.1, 0.8, -0.1), // Aortic root (from LV)
      new THREE.Vector3(0.2, 2.0, 0.1),  // Ascending
      new THREE.Vector3(-0.4, 2.6, -0.3), // Arch
      new THREE.Vector3(-0.7, 1.5, -0.8)  // Descending
    ]);
    const aortaGeom = new THREE.TubeGeometry(aortaCurve, 32, 0.45, 16, false);
    const aortaMesh = new THREE.Mesh(aortaGeom, vesselMat);
    heartGroup.add(aortaMesh);

    // 6. PULMONARY ARTERY (Crossing Aorta)
    const pulmCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(-0.4, 0.6, 0.6), // Pulmonary root (from RV)
      new THREE.Vector3(-0.1, 1.6, 0.4), // Main PA
      new THREE.Vector3(0.7, 1.9, -0.1)  // Branching to lungs
    ]);
    const pulmGeom = new THREE.TubeGeometry(pulmCurve, 32, 0.4, 16, false);
    const pulmMesh = new THREE.Mesh(pulmGeom, venousMat);
    heartGroup.add(pulmMesh);

    // 7. SUPERIOR VENA CAVA (SVC)
    const svcCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(-0.9, 1.5, 0.1), // Entering RA
      new THREE.Vector3(-1.0, 2.5, 0.2)
    ]);
    const svcGeom = new THREE.TubeGeometry(svcCurve, 16, 0.35, 16, false);
    const svcMesh = new THREE.Mesh(svcGeom, venousMat);
    heartGroup.add(svcMesh);

    // 8. INFERIOR VENA CAVA (IVC)
    const ivcCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(-0.9, 0.2, 0.0), // Entering RA from below
      new THREE.Vector3(-1.0, -1.0, -0.1)
    ]);
    const ivcGeom = new THREE.TubeGeometry(ivcCurve, 16, 0.35, 16, false);
    const ivcMesh = new THREE.Mesh(ivcGeom, venousMat);
    heartGroup.add(ivcMesh);

    // 9. CORONARY ARTERIES (LAD)
    const ladCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0.1, 0.8, 0.3), // Origin near aorta
      new THREE.Vector3(-0.1, 0.2, 0.9), // Down the interventricular sulcus
      new THREE.Vector3(0.0, -0.8, 0.9),
      new THREE.Vector3(0.2, -1.6, 0.5)  // Towards apex
    ]);
    const ladGeom = new THREE.TubeGeometry(ladCurve, 32, 0.06, 8, false);
    const ladMesh = new THREE.Mesh(ladGeom, vesselMat);
    heartGroup.add(ladMesh);
    
    // 10. CORONARY ARTERIES (RCA)
    const rcaCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(-0.1, 0.8, 0.2), // Origin
      new THREE.Vector3(-0.8, 0.4, 0.6), // Right AV groove
      new THREE.Vector3(-1.1, -0.2, 0.2),
      new THREE.Vector3(-0.6, -0.8, -0.4) // Posterior
    ]);
    const rcaGeom = new THREE.TubeGeometry(rcaCurve, 32, 0.06, 8, false);
    const rcaMesh = new THREE.Mesh(rcaGeom, vesselMat);
    heartGroup.add(rcaMesh);

    // --- PATHOLOGY VISUALIZATION (ISCHEMIA MAP) ---
    // Instead of lights, we tint the specific mesh material
    if (ischemia?.affectedWall) {
      const wall = ischemia.affectedWall.toLowerCase();
      const severityColor = ischemia.stSegmentTrend === 'Elevation' ? COLORS.ISCHEMIA : COLORS.INJURY;
      const targetColor = new THREE.Color(severityColor);

      if (wall.includes('anterior')) {
        // Anterior wall is mostly LV front + RV
        (lvMesh.material as THREE.MeshPhysicalMaterial).color.lerp(targetColor, 0.6);
        (lvMesh.material as THREE.MeshPhysicalMaterial).emissive.set(COLORS.INJURY);
        (lvMesh.material as THREE.MeshPhysicalMaterial).emissiveIntensity = 0.3;
      } 
      else if (wall.includes('inferior')) {
        // Inferior is LV bottom. We can't paint vertices easily here, so we tint LV
        // and add a marker at the bottom
        (lvMesh.material as THREE.MeshPhysicalMaterial).color.lerp(targetColor, 0.4);
        const marker = new THREE.Mesh(new THREE.SphereGeometry(0.5), new THREE.MeshBasicMaterial({ color: COLORS.ISCHEMIA }));
        marker.position.set(0.3, -1.8, 0); // Apex/Bottom
        heartGroup.add(marker);
      }
      else if (wall.includes('lateral')) {
         // High Lateral (LCx)
         (laMesh.material as THREE.MeshPhysicalMaterial).color.set(targetColor);
         (lvMesh.material as THREE.MeshPhysicalMaterial).color.lerp(targetColor, 0.3);
      }
      else if (wall.includes('septal')) {
         // Septal is hidden, tint RV/LV overlap area
         (rvMesh.material as THREE.MeshPhysicalMaterial).color.lerp(targetColor, 0.5);
      }
    }

    // --- ELECTRICAL AXIS VECTOR ---
    // Visualize the mean electrical vector
    const rad = (numericAxis - 90) * (Math.PI / 180) * -1; // Convert ECG degrees to Rads adjusted for screen Y-up
    const axisDir = new THREE.Vector3(Math.cos(rad), Math.sin(rad), 0).normalize();
    const arrow = new THREE.ArrowHelper(axisDir, new THREE.Vector3(0,0,2), 3.5, COLORS.INJURY, 0.5, 0.4);
    scene.add(arrow);


    // --- LIGHTING ---
    const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 0.6);
    scene.add(hemiLight);
    
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.5);
    dirLight.position.set(5, 5, 5);
    dirLight.castShadow = true;
    scene.add(dirLight);

    const redLight = new THREE.PointLight(0xff0000, 0.8, 10);
    redLight.position.set(-5, -5, 2);
    scene.add(redLight);

    // --- ANIMATION LOOP ---
    const clock = new THREE.Clock();
    let frameId = 0;

    const animate = () => {
      frameId = requestAnimationFrame(animate);
      const time = clock.getElapsedTime();
      
      // 1. RHYTHM LOGIC
      // Default: Sinus Rhythm
      let bpm = heartRate || 60;
      let atriaRate = bpm;
      let ventRate = bpm;
      let irregularity = 0; // 0 to 1
      let atrialChaos = false; // For AFib

      if (pattern === 'afib') {
         atrialChaos = true;
         ventRate = bpm; // Ventricles irregular
         irregularity = 0.4;
      } 
      else if (pattern === 'flutter') {
         atriaRate = 300; // Fast atrial
         ventRate = bpm; 
      }
      else if (pattern === 'avblock3') {
         atriaRate = 80;
         ventRate = 35; // Dissociation
      }
      else if (pattern === 'vt' || pattern === 'vf') {
         ventRate = bpm;
         atriaRate = 0; // Hidden
      }
      else if (pattern === 'svt') {
         ventRate = bpm;
         atriaRate = bpm; // Fast 1:1 conduction
      }

      // 2. TIMING
      // Simulate cardiac cycle: 60/BPM = beat duration
      // Systole usually ~0.3s, Diastole remainder
      
      const calcPhase = (rate: number, offsetMs: number, isIrregular: boolean) => {
         const cycleDuration = 60 / rate;
         let t = time;
         if (isIrregular) t += Math.sin(time * 3) * 0.1; // Jitter
         const localTime = (t - (offsetMs / 1000)) % cycleDuration;
         return localTime / cycleDuration; // 0 to 1
      };

      // Delays (Conduction Blocks)
      let lvDelay = 0;
      let rvDelay = 0;
      if (conduction?.ivcdType === 'LBBB') lvDelay = 120; // 120ms delay
      if (conduction?.ivcdType === 'RBBB') rvDelay = 120;
      if (conduction?.wpwPattern) lvDelay = -40; // Pre-excitation

      // Phases
      const aPhase = calcPhase(atriaRate, 0, false);
      const lvPhase = calcPhase(ventRate, 150 + lvDelay, pattern === 'afib'); // AV delay ~150ms normally
      const rvPhase = calcPhase(ventRate, 150 + rvDelay, pattern === 'afib');

      // 3. CONTRACTION FUNCTIONS
      // Returns scale factor (1.0 = relaxed, <1.0 = contracted)
      const contraction = (phase: number, strength: number, sharp: boolean) => {
          // Sharp contraction (Systole) occupies ~30% of cycle
          if (phase < 0.3) {
             // Systole: Rapid squeeze
             const p = phase / 0.3; 
             return 1 - (Math.sin(p * Math.PI) * strength);
          } else {
             // Diastole: Relaxed
             return 1.0;
          }
      };

      // Apply to Meshes
      // LV
      let lvSqueeze = contraction(lvPhase, 0.15, true); 
      if (pattern === 'stemi' || urgency === 'Emergency') lvSqueeze = contraction(lvPhase, 0.05, true); // Hypokinesis
      lvMesh.scale.setScalar(lvSqueeze);

      // RV
      const rvSqueeze = contraction(rvPhase, 0.15, true);
      rvMesh.scale.setScalar(rvSqueeze);

      // Atria
      if (atrialChaos) {
         // AFib: Quivering (high freq low amp noise)
         const quiver = 1 + (Math.sin(time * 50) * 0.02);
         laMesh.scale.setScalar(quiver);
         raMesh.scale.setScalar(quiver);
      } else {
         // Normal kick
         const aSqueeze = contraction(aPhase, 0.12, true);
         laMesh.scale.setScalar(aSqueeze);
         raMesh.scale.setScalar(aSqueeze);
      }

      // 4. MOTION & FEEL
      // Gentle floating
      heartGroup.rotation.y = Math.sin(time * 0.2) * 0.1;
      heartGroup.position.y = Math.sin(time * 1) * 0.05;

      renderer.render(scene, camera);
    };

    animate();

    const handleResize = () => {
      if (!containerRef.current) return;
      const w = containerRef.current.clientWidth;
      const h = containerRef.current.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(frameId);
      if(rendererRef.current) rendererRef.current.dispose();
      heartGroup.clear();
    };
  }, [heartRate, pattern, structural, ischemia, conduction, urgency, numericAxis]);

  return (
    <div className={`relative w-full h-[500px] rounded-[3rem] overflow-hidden border transition-all duration-500
      ${urgency === 'Emergency' ? 'bg-rose-950/30 border-rose-500/30 shadow-[0_0_50px_rgba(225,29,72,0.1)]' : 'bg-slate-900/50 border-white/5 shadow-2xl'}
    `}>
      <div ref={containerRef} className="w-full h-full cursor-move" title="Anatomical Digital Twin" />
      
      {/* 3D HUD OVERLAY */}
      <div className="absolute top-6 left-8 pointer-events-none">
         <div className="flex flex-col gap-1">
            <h3 className="text-white font-black uppercase tracking-tighter text-2xl italic">Anatomical Twin</h3>
            <div className="flex items-center gap-2">
               <div className={`w-2 h-2 rounded-full ${urgency === 'Emergency' ? 'bg-rose-500 animate-ping' : 'bg-emerald-500'}`}></div>
               <span className="text-[10px] font-mono text-cyan-400 uppercase tracking-widest">
                  SYNC: {conduction?.ivcdType || 'NORMAL'}
               </span>
            </div>
            {ischemia?.affectedWall && (
                <div className="mt-2 px-3 py-1 bg-rose-500/20 border border-rose-500/40 rounded text-[9px] text-rose-300 font-bold uppercase tracking-widest w-max">
                   ⚠️ ISCHEMIA: {ischemia.affectedWall}
                </div>
            )}
            {structural?.lvhDetected && (
                <div className="mt-1 px-3 py-1 bg-amber-500/20 border border-amber-500/40 rounded text-[9px] text-amber-300 font-bold uppercase tracking-widest w-max">
                   LV HYPERTROPHY
                </div>
            )}
         </div>
      </div>

      <div className="absolute bottom-6 right-8 text-right pointer-events-none">
         <div className="text-[9px] text-slate-500 font-mono uppercase tracking-[0.3em] mb-1">Electrical Axis</div>
         <div className="text-4xl font-black text-white font-mono tracking-tighter">{numericAxis}°</div>
         <div className="w-32 h-1 bg-white/10 mt-2 ml-auto rounded-full overflow-hidden">
             <div className="h-full bg-cyan-500 animate-pulse" style={{ width: '60%' }}></div>
         </div>
      </div>
    </div>
  );
};

export default Heart3D;
