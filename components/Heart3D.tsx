
import React, { useRef, useEffect, useMemo } from 'react';
import * as THREE from 'three';
import { identifyEcgPattern } from '../utils/cardioLogic';

interface Heart3DProps {
  heartRate: number;
  urgency?: 'Emergency' | 'Urgent' | 'Routine' | 'Low';
  diagnosis?: string;
  rhythm?: string;
  numericAxis?: number;
  prIntervalMs?: number;
  qrsDurationMs?: number;
}

const Heart3D: React.FC<Heart3DProps> = ({ 
  heartRate, 
  urgency = 'Routine', 
  diagnosis, 
  rhythm,
  numericAxis = 60,
  prIntervalMs,
  qrsDurationMs
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const heartGroupRef = useRef<THREE.Group | null>(null);

  const pattern = useMemo(() => 
    identifyEcgPattern([rhythm || ''], diagnosis || '', prIntervalMs, qrsDurationMs), 
  [rhythm, diagnosis, prIntervalMs, qrsDurationMs]);

  const COLORS = {
    NORMAL: '#06b6d4',
    CRITICAL: '#ef4444',
    WARNING: '#f59e0b',
    ELECTRIC: '#ffffff',
    ISCHEMIC_GLOW: '#ff0000',
    AXIS_VECTOR: '#d946ef'
  };

  useEffect(() => {
    if (!containerRef.current) return;

    const scene = new THREE.Scene();
    const width = containerRef.current.clientWidth;
    const height = containerRef.current.clientHeight;
    const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(width, height);
    containerRef.current.appendChild(renderer.domElement);

    const group = new THREE.Group();
    heartGroupRef.current = group;
    scene.add(group);

    const axisRad = (numericAxis - 60) * (Math.PI / 180);
    group.rotation.z = axisRad * 0.6;
    group.rotation.y = axisRad * 0.2;
    group.rotation.x = 0.2;

    const baseColor = ['Emergency', 'vf', 'vfl', 'vt'].includes(pattern) || urgency === 'Emergency' 
      ? COLORS.CRITICAL 
      : (urgency === 'Urgent' ? COLORS.WARNING : COLORS.NORMAL);
    
    const chamberMat = (opacity = 0.7) => new THREE.MeshPhongMaterial({
      color: new THREE.Color(baseColor),
      transparent: true,
      opacity: opacity,
      emissive: new THREE.Color(baseColor),
      emissiveIntensity: 0.15,
      shininess: 80,
      side: THREE.DoubleSide
    });

    // Left Ventricle (LV)
    const lvGeom = new THREE.SphereGeometry(1.1, 48, 48);
    lvGeom.scale(0.85, 1.4, 0.8);
    lvGeom.translate(-0.3, -0.4, 0);
    const lvMesh = new THREE.Mesh(lvGeom, chamberMat(0.8));
    group.add(lvMesh);

    // Right Ventricle (RV)
    const rvGeom = new THREE.SphereGeometry(1.0, 48, 48);
    rvGeom.scale(0.9, 1.1, 0.6);
    rvGeom.translate(0.4, -0.2, 0.2);
    const rvMesh = new THREE.Mesh(rvGeom, chamberMat(0.6));
    group.add(rvMesh);

    // Left Atrium (LA)
    const laGeom = new THREE.SphereGeometry(0.7, 32, 32);
    laGeom.scale(1, 0.8, 1);
    laGeom.translate(-0.4, 0.9, -0.2);
    const laMesh = new THREE.Mesh(laGeom, chamberMat(0.5));
    group.add(laMesh);

    // Right Atrium (RA)
    const raGeom = new THREE.SphereGeometry(0.7, 32, 32);
    raGeom.scale(1, 0.8, 1);
    raGeom.translate(0.5, 0.8, 0.1);
    const raMesh = new THREE.Mesh(raGeom, chamberMat(0.5));
    group.add(raMesh);

    const ambLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambLight);
    const spotLight = new THREE.SpotLight(0xffffff, 1.5);
    spotLight.position.set(5, 5, 10);
    scene.add(spotLight);

    camera.position.z = 6;

    let clock = new THREE.Clock();
    const animate = () => {
      const frameId = requestAnimationFrame(animate);
      const t = clock.getElapsedTime();
      
      // Default Base Frequency (beats per second)
      let atrialBps = heartRate / 60;
      let ventricularBps = heartRate / 60;
      
      let aIntensity = 0.08;
      let vIntensity = 0.12;
      let aJitter = 0;
      let vJitter = 0;
      let rvJitter = 0;

      // Neural Modulation based on Rhythm Pattern
      switch(pattern) {
        case 'vf':
          ventricularBps = 0; // No meaningful pumping
          atrialBps = 0;
          vIntensity = 0.05;
          vJitter = Math.sin(t * 120) * 0.04; // High frequency fibrillation
          break;
        case 'vfl':
          ventricularBps = 4; // Fast regular flutter
          vIntensity = 0.15;
          break;
        case 'vt':
          ventricularBps = 3.5; // Fast tachycardia
          atrialBps = heartRate / 120; // Dissociated, usually slower
          vIntensity = 0.18;
          aIntensity = 0.04;
          break;
        case 'afib':
          atrialBps = 12 + Math.random(); // Fast and chaotic
          aIntensity = 0.02;
          aJitter = Math.sin(t * 50) * 0.01;
          // Ventricles are irregular in AFib
          vJitter = Math.sin(t * (atrialBps / 4)) * 0.02;
          break;
        case 'aflutter':
          atrialBps = 5; // Fixed fast flutter waves
          aIntensity = 0.1;
          break;
        case 'arvd':
          // Right ventricle shows localized dyskinesia
          rvJitter = Math.sin(t * 30) * 0.03;
          vIntensity = 0.1;
          break;
        case 'brady':
          vIntensity = 0.15; // Deeper contraction for slow rates
          break;
        case 'paced':
          // Sharper, more mechanical feeling
          vIntensity = 0.2;
          break;
      }

      // Calculate Pulse Dynamics
      // Atrial contraction (P wave simulation)
      const atrialPhase = (t * atrialBps) % 1;
      const atrialScale = 1 + (Math.sin(atrialPhase * Math.PI) * aIntensity) + aJitter;
      
      // Ventricular contraction (QRS simulation)
      // We add a slight phase offset (PR interval) for normal rhythms
      const vOffset = pattern === 'vt' || pattern === 'vf' ? 0 : 0.15;
      const ventricularPhase = (t * ventricularBps + vOffset) % 1;
      const ventricularScale = 1 + (Math.pow(Math.sin(ventricularPhase * Math.PI), 2) * vIntensity) + vJitter;
      
      // Apply transforms to meshes
      laMesh.scale.setScalar(atrialScale);
      raMesh.scale.setScalar(atrialScale);
      
      lvMesh.scale.setScalar(ventricularScale);
      // RV might have specific dyskinesia (ARVD)
      rvMesh.scale.setScalar(ventricularScale + rvJitter);

      group.rotation.y += 0.005;
      renderer.render(scene, camera);
    };

    animate();

    const handleResize = () => {
      if (!containerRef.current) return;
      camera.aspect = containerRef.current.clientWidth / containerRef.current.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      renderer.dispose();
      if (containerRef.current) containerRef.current.removeChild(renderer.domElement);
    };
  }, [heartRate, pattern, urgency, numericAxis]);

  return (
    <div className={`relative w-full h-[400px] ${['vf', 'vfl', 'vt'].includes(pattern) ? 'bg-rose-950/20' : 'bg-slate-950/20'} rounded-[3rem] overflow-hidden border ${['vf', 'vfl', 'vt'].includes(pattern) ? 'border-rose-500/30' : 'border-white/5'} group transition-colors duration-1000`}>
      <div ref={containerRef} className="w-full h-full" />
      
      {/* HUD Overlays */}
      <div className="absolute top-8 left-8">
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${['vf', 'vfl', 'vt'].includes(pattern) ? 'bg-red-500 animate-ping' : 'bg-cyan-500 animate-pulse'}`}></div>
          <span className="text-[10px] font-mono text-slate-500 uppercase tracking-widest font-black">
            Signal: {pattern === 'normal' ? 'Sinus Sync' : 'Pathological Rhythm'}
          </span>
        </div>
      </div>

      <div className="absolute bottom-8 right-8 z-10 text-right">
         <span className={`text-5xl font-black ${['vf', 'vfl', 'vt'].includes(pattern) ? 'text-rose-500' : 'text-white'} font-mono tracking-tighter italic drop-shadow-[0_0_15px_rgba(0,0,0,0.5)]`}>
           {pattern === 'vf' ? 'CHAOS' : heartRate} <span className="text-xs text-cyan-500">{pattern === 'vf' ? '' : 'BPM'}</span>
         </span>
         <div className="text-[9px] text-slate-400 uppercase font-mono tracking-[0.2em] mt-2 font-black">
           {rhythm || diagnosis || 'Analyzing Morphology...'}
         </div>
      </div>
    </div>
  );
};

export default Heart3D;
