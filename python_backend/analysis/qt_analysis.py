import numpy as np
from typing import Dict, List, Any, Optional

class QTIntervalAnalyzer:
    """
    Advanced QT Interval Analysis for Arrhythmia Risk Stratification.
    Calculates QTc (Bazett, Fridericia, Framingham), QT Dispersion, and TdP Risk.
    """
    
    def __init__(self):
        pass

    def calculate_qtc(self, qt_ms: float, rr_sec: float, method: str = 'bazett') -> float:
        """
        Calculates Corrected QT interval.
        
        Args:
            qt_ms: QT interval in milliseconds
            rr_sec: RR interval in seconds (60/HR)
            method: 'bazett', 'fridericia', 'framingham', 'hodges'
        """
        if rr_sec <= 0: return qt_ms
        
        if method == 'bazett':
            # QTc = QT / sqrt(RR)
            # Overcorrects at high HR
            return qt_ms / np.sqrt(rr_sec)
            
        elif method == 'fridericia':
            # QTc = QT / cbrt(RR)
            # Better for high HR
            return qt_ms / np.cbrt(rr_sec)
            
        elif method == 'framingham':
            # QTc = QT + 154 * (1 - RR)
            return qt_ms + 154 * (1 - rr_sec)
            
        elif method == 'hodges':
            # QTc = QT + 1.75 * (HR - 60)
            hr = 60 / rr_sec
            return qt_ms + 1.75 * (hr - 60)
            
        return qt_ms

    def calculate_dispersion(self, qt_measurements: Dict[str, float]) -> Dict[str, float]:
        """
        Calculates QT Dispersion across available leads.
        QTd = QT_max - QT_min
        """
        values = [v for v in qt_measurements.values() if v > 0]
        if not values:
            return {"qt_dispersion": 0.0, "qt_max": 0.0, "qt_min": 0.0}
            
        qt_max = max(values)
        qt_min = min(values)
        qt_disp = qt_max - qt_min
        
        return {
            "qt_dispersion": round(qt_disp, 1),
            "qt_max": round(qt_max, 1),
            "qt_min": round(qt_min, 1)
        }

    def calculate_heterogeneity(self, qt_ms: float, qrs_ms: float, tp_te_ms: float) -> Dict[str, float]:
        """
        Calculates indices of Ventricular Heterogeneity.
        
        1. Tp-Te / QT Ratio: Index of transmural dispersion of repolarization.
        2. iCEB (Index of Cardio-Electrophysiological Balance): QT / QRS.
        """
        results = {}
        
        if qt_ms > 0:
            results['tp_te_qt_ratio'] = round(tp_te_ms / qt_ms, 3)
            
        if qrs_ms > 0:
            results['iCEB'] = round(qt_ms / qrs_ms, 2)
            
        return results

    def assess_tdp_risk(self, 
                        qtc_ms: float, 
                        qt_dispersion: float, 
                        sex: str = 'Male',
                        tp_te_ms: float = 0) -> Dict[str, Any]:
        """
        Assesses risk of Torsades de Pointes (TdP).
        """
        risk_score = 0
        factors = []
        
        # 1. QTc Prolongation
        # Male > 450, Female > 460 (Borderline)
        # > 500 (High Risk)
        limit = 450 if sex == 'Male' else 460
        
        if qtc_ms > 500:
            risk_score += 3
            factors.append(f"Severe QTc prolongation ({int(qtc_ms)}ms > 500ms)")
        elif qtc_ms > limit:
            risk_score += 1
            factors.append(f"Prolonged QTc ({int(qtc_ms)}ms > {limit}ms)")
            
        # 2. QT Dispersion
        # Normal < 50ms. > 60-80ms indicates heterogeneity.
        if qt_dispersion > 80:
            risk_score += 2
            factors.append(f"High QT Dispersion ({int(qt_dispersion)}ms)")
        elif qt_dispersion > 60:
            risk_score += 1
            factors.append(f"Elevated QT Dispersion ({int(qt_dispersion)}ms)")
            
        # 3. Tp-Te Interval (Transmural Dispersion)
        # > 100ms is associated with TdP/SCD
        if tp_te_ms > 100:
            risk_score += 2
            factors.append(f"Prolonged Tp-Te ({int(tp_te_ms)}ms)")
            
        risk_level = "Low"
        if risk_score >= 4: risk_level = "Very High"
        elif risk_score >= 2: risk_level = "Moderate"
        elif risk_score >= 1: risk_level = "Low-Moderate"
        
        return {
            "risk_level": risk_level,
            "score": risk_score,
            "risk_factors": factors,
            "recommendation": "Monitor electrolytes (K+, Mg++). Avoid QT-prolonging drugs." if risk_score >= 2 else "Routine monitoring."
        }

    def analyze(self, 
                qt_measurements: Dict[str, float], 
                hr: float, 
                qrs_ms: float, 
                sex: str = 'Male',
                tp_te_ms: float = 0) -> Dict[str, Any]:
        """
        Full QT Analysis Pipeline.
        """
        # 1. Representative QT (usually median or max of II/V5)
        # For safety, we often take the max QT found to avoid missing risk
        qt_vals = [v for v in qt_measurements.values() if v > 0]
        if not qt_vals:
            return {"error": "No valid QT measurements"}
            
        global_qt = max(qt_vals) # Conservative approach
        rr_sec = 60.0 / hr if hr > 0 else 1.0
        
        # 2. QTc Calculations
        qtc_vals = {
            "Bazett": self.calculate_qtc(global_qt, rr_sec, 'bazett'),
            "Fridericia": self.calculate_qtc(global_qt, rr_sec, 'fridericia'),
            "Framingham": self.calculate_qtc(global_qt, rr_sec, 'framingham'),
            "Hodges": self.calculate_qtc(global_qt, rr_sec, 'hodges')
        }
        
        # Select primary QTc (Fridericia preferred for high HR, Bazett standard)
        primary_qtc = qtc_vals['Fridericia'] if hr > 80 else qtc_vals['Bazett']
        
        # 3. Dispersion
        dispersion = self.calculate_dispersion(qt_measurements)
        
        # 4. Heterogeneity
        heterogeneity = self.calculate_heterogeneity(global_qt, qrs_ms, tp_te_ms)
        
        # 5. Risk
        risk = self.assess_tdp_risk(primary_qtc, dispersion['qt_dispersion'], sex, tp_te_ms)
        
        return {
            "qt_interval_ms": global_qt,
            "qtc_primary_ms": round(primary_qtc, 1),
            "qtc_methods": {k: round(v, 1) for k, v in qtc_vals.items()},
            "dispersion": dispersion,
            "heterogeneity_indices": heterogeneity,
            "tdp_risk": risk
        }

# Example Usage
if __name__ == "__main__":
    analyzer = QTIntervalAnalyzer()
    
    # Mock Measurements (ms)
    qt_leads = {
        'I': 400, 'II': 410, 'III': 405,
        'V1': 390, 'V2': 400, 'V3': 415,
        'V4': 420, 'V5': 418, 'V6': 410
    }
    
    # High Risk Case
    # HR 60 -> RR 1.0s
    # QT 420ms -> QTc 420ms (Normal)
    # Dispersion: 420 - 390 = 30ms (Normal)
    
    res = analyzer.analyze(qt_leads, hr=60, qrs_ms=90, sex='Male', tp_te_ms=80)
    import json
    print(json.dumps(res, indent=2))
