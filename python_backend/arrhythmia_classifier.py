import numpy as np
from typing import List, Dict, Any, Optional, Tuple

class ArrhythmiaClassifier:
    """
    Multi-stage Arrhythmia Classification System.
    Integrates Heuristic Logic, Statistical Analysis, and (simulated) ML/DL Ensemble.
    """
    def __init__(self, fs: int = 500):
        self.fs = fs
        # In a real implementation, load ML models here
        # self.xgb_model = joblib.load('xgb_arrhythmia.pkl')
        # self.lstm_model = load_model('lstm_arrhythmia.h5')

    def _extract_rr_features(self, rr_intervals_ms: np.ndarray) -> Dict[str, float]:
        """
        Extracts time-domain and non-linear features from RR intervals.
        """
        if len(rr_intervals_ms) < 2:
            return {"rmssd": 0, "sdnn": 0, "cv": 0, "entropy": 0}

        diff_rr = np.diff(rr_intervals_ms)
        
        # Time Domain
        mean_rr = np.mean(rr_intervals_ms)
        sdnn = np.std(rr_intervals_ms)
        rmssd = np.sqrt(np.mean(diff_rr ** 2))
        cv = sdnn / mean_rr if mean_rr > 0 else 0
        pnn50 = np.sum(np.abs(diff_rr) > 50) / len(diff_rr) * 100
        
        # Non-linear / Statistical (Entropy)
        # Simple histogram entropy
        hist, _ = np.histogram(rr_intervals_ms, bins='auto', density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log(hist))
        
        return {
            "mean_rr": float(mean_rr),
            "sdnn": float(sdnn),
            "rmssd": float(rmssd),
            "cv": float(cv),
            "pnn50": float(pnn50),
            "entropy": float(entropy)
        }

    def detect_afib(self, rr_features: Dict[str, float], p_wave_confidence: float) -> Dict[str, Any]:
        """
        Detects Atrial Fibrillation using ensemble logic.
        Target Sensitivity > 96%.
        """
        score = 0.0
        reasons = []
        
        # 1. Heuristic Layer
        # AFib is "Irregularly Irregular" -> High CV, High Entropy, High RMSSD
        if rr_features['cv'] > 0.15: # High variability
            score += 0.3
            reasons.append(f"High RR variability (CV={rr_features['cv']:.2f})")
        
        if rr_features['rmssd'] > 40: # High beat-to-beat variation
            score += 0.2
            reasons.append(f"High RMSSD ({rr_features['rmssd']:.1f}ms)")
            
        if rr_features['entropy'] > 3.5: # Chaotic rhythm
            score += 0.2
            reasons.append("Chaotic RR distribution")
            
        # 2. P-Wave Logic
        # Absence of P-waves is critical
        if p_wave_confidence < 0.3:
            score += 0.3
            reasons.append("Absent/Low-confidence P-waves")
        elif p_wave_confidence > 0.8:
            score -= 0.5 # Strong P-waves argue against AFib
            
        # 3. ML/DL Layer (Simulated)
        # Assume ML model sees the irregularity and confirms
        ml_probability = 0.85 if score > 0.4 else 0.1
        
        # Ensemble Vote
        final_probability = (score + ml_probability) / 2
        is_afib = final_probability > 0.6
        
        return {
            "is_afib": bool(is_afib),
            "probability": float(final_probability),
            "reasons": reasons
        }

    def detect_ectopy(self, rr_intervals_ms: np.ndarray, qrs_durations_ms: List[float]) -> Dict[str, Any]:
        """
        Detects PVCs (Premature Ventricular Contractions) and PACs (Premature Atrial Contractions).
        Target Sensitivity: PVC > 90%.
        """
        pvcs = []
        pacs = []
        
        if len(rr_intervals_ms) < 3:
            return {"pvcs": [], "pacs": []}
            
        # Calculate running mean for local context
        window = 5
        
        for i in range(1, len(rr_intervals_ms) - 1):
            prev_rr = rr_intervals_ms[i-1]
            curr_rr = rr_intervals_ms[i]
            next_rr = rr_intervals_ms[i+1]
            
            # Prematurity Criteria: Current RR is < 80% of previous
            is_premature = curr_rr < 0.8 * prev_rr
            
            if is_premature:
                # Compensatory Pause: Next RR is lengthened
                # Full compensatory: curr + next ~ 2 * prev
                is_compensatory = (curr_rr + next_rr) >= 1.8 * prev_rr
                
                # Morphology Check (if available)
                # PVCs are wide (>120ms), PACs are narrow (<120ms)
                # We need to map RR index to QRS index. Assuming 1:1 mapping for simplicity here.
                # i-th RR interval ends at i-th beat (roughly)
                qrs_dur = qrs_durations_ms[i] if i < len(qrs_durations_ms) else 80
                
                if qrs_dur > 120:
                    pvcs.append({
                        "index": i,
                        "prematurity": float(curr_rr/prev_rr),
                        "qrs_dur": qrs_dur,
                        "type": "PVC"
                    })
                else:
                    pacs.append({
                        "index": i,
                        "prematurity": float(curr_rr/prev_rr),
                        "type": "PAC"
                    })
                    
        return {
            "pvcs_count": len(pvcs),
            "pacs_count": len(pacs),
            "details": pvcs + pacs
        }

    def detect_blocks(self, pr_intervals_ms: List[float], dropped_beats: bool) -> Dict[str, Any]:
        """
        Detects AV Blocks (1st, 2nd, 3rd degree).
        Target Sensitivity > 92%.
        """
        valid_prs = [pr for pr in pr_intervals_ms if pr is not None and not np.isnan(pr)]
        
        if not valid_prs:
            if dropped_beats:
                return {"type": "3rd Degree AV Block (Complete)", "confidence": 0.85, "reason": "AV Dissociation / No conducted P-waves"}
            return {"type": "None", "confidence": 0.0}
            
        mean_pr = np.mean(valid_prs)
        std_pr = np.std(valid_prs)
        
        # 1st Degree
        if mean_pr > 200 and not dropped_beats:
            return {
                "type": "1st Degree AV Block",
                "confidence": 0.95,
                "reason": f"Prolonged PR interval ({mean_pr:.0f}ms) > 200ms"
            }
            
        # 2nd Degree Type I (Wenckebach)
        # Progressive prolongation?
        # Check trend in PR intervals
        if dropped_beats and len(valid_prs) > 3:
            # Simple check: is variance high and is there a pattern?
            # In Wenckebach, PR lengthens.
            is_lengthening = valid_prs[-1] > valid_prs[0] + 40 # Rough check
            if is_lengthening:
                 return {
                    "type": "2nd Degree AV Block Type I (Wenckebach)",
                    "confidence": 0.80,
                    "reason": "Progressive PR prolongation with dropped beats"
                }
        
        # 2nd Degree Type II
        if dropped_beats and std_pr < 20: # Constant PR
             return {
                "type": "2nd Degree AV Block Type II",
                "confidence": 0.85,
                "reason": "Constant PR intervals with dropped beats"
            }
            
        return {"type": "None", "confidence": 0.0}

    def classify_rhythm(self, 
                        rr_intervals_ms: List[float], 
                        p_wave_confidence: float, 
                        qrs_durations_ms: List[float],
                        pr_intervals_ms: List[float]) -> Dict[str, Any]:
        """
        Main classification pipeline.
        """
        rr_arr = np.array(rr_intervals_ms)
        
        # 1. Feature Extraction
        features = self._extract_rr_features(rr_arr)
        
        # 2. Detect AFib
        afib_res = self.detect_afib(features, p_wave_confidence)
        
        # 3. Detect Ectopy
        ectopy_res = self.detect_ectopy(rr_arr, qrs_durations_ms)
        
        # 4. Detect Blocks
        # Infer dropped beats from RR pauses without QRS? 
        # For now, we assume dropped_beats flag comes from P-QRS matching logic (not implemented here)
        # We'll assume False for this snippet unless RR is double mean
        has_pause = np.max(rr_arr) > 1.8 * np.mean(rr_arr)
        blocks_res = self.detect_blocks(pr_intervals_ms, dropped_beats=has_pause)
        
        # 5. Final Diagnosis Logic
        diagnosis = "Sinus Rhythm"
        confidence = 0.9
        
        if afib_res['is_afib']:
            diagnosis = "Atrial Fibrillation"
            confidence = afib_res['probability']
        elif blocks_res['type'] != "None":
            diagnosis = blocks_res['type']
            confidence = blocks_res['confidence']
        elif ectopy_res['pvcs_count'] > 0:
            if ectopy_res['pvcs_count'] > 5: # Arbitrary threshold for "Frequent"
                diagnosis = "Sinus Rhythm with Frequent PVCs"
            else:
                diagnosis = "Sinus Rhythm with Occasional PVCs"
        elif features['mean_rr'] < 600: # > 100 bpm
            diagnosis = "Sinus Tachycardia"
        elif features['mean_rr'] > 1000: # < 60 bpm
            diagnosis = "Sinus Bradycardia"
            
        return {
            "diagnosis": diagnosis,
            "confidence": confidence,
            "features": features,
            "modules": {
                "afib": afib_res,
                "ectopy": ectopy_res,
                "blocks": blocks_res
            }
        }

# Example Usage
if __name__ == "__main__":
    classifier = ArrhythmiaClassifier()
    
    # Mock Data: Normal Sinus
    rr_normal = np.random.normal(800, 20, 20) # 75 bpm, regular
    res_normal = classifier.classify_rhythm(rr_normal, 0.95, [90]*20, [160]*20)
    print(f"Normal: {res_normal['diagnosis']}")
    
    # Mock Data: AFib
    rr_afib = np.random.exponential(800, 20) # Irregular
    res_afib = classifier.classify_rhythm(rr_afib, 0.1, [90]*20, [])
    print(f"AFib: {res_afib['diagnosis']} (Confidence: {res_afib['confidence']:.2f})")
    
    # Mock Data: PVCs
    rr_pvc = np.array([800, 800, 800, 500, 1100, 800, 800]) # Premature then pause
    qrs_pvc = [90, 90, 90, 140, 90, 90, 90] # Wide QRS at index 3
    res_pvc = classifier.classify_rhythm(rr_pvc, 0.9, qrs_pvc, [160]*7)
    print(f"PVC: {res_pvc['diagnosis']}")
