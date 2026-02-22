import numpy as np
from typing import Dict, List, Any, Optional

class TemporalComparator:
    """
    Performs Serial ECG Analysis (Temporal Comparison).
    Compares current ECG with previous records to detect dynamic changes,
    evolution of ischemia, and new conduction abnormalities.
    """
    
    def __init__(self):
        pass

    def compare_records(self, current_features: Dict[str, Any], previous_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compares two sets of ECG features.
        
        Args:
            current_features: Extracted features from the new ECG.
            previous_features: Extracted features from the baseline/previous ECG.
            
        Returns:
            Dict containing detected changes and clinical significance.
        """
        changes = []
        significance = "Stable"
        
        # 1. New Conduction Abnormalities (High Risk)
        # New LBBB
        curr_lbbb = current_features.get('is_lbbb', False)
        prev_lbbb = previous_features.get('is_lbbb', False)
        
        if curr_lbbb and not prev_lbbb:
            changes.append("New Left Bundle Branch Block (LBBB)")
            significance = "Critical" # New LBBB can be STEMI equivalent
            
        # New RBBB
        curr_rbbb = current_features.get('is_rbbb', False)
        prev_rbbb = previous_features.get('is_rbbb', False)
        
        if curr_rbbb and not prev_rbbb:
            changes.append("New Right Bundle Branch Block (RBBB)")
            if significance != "Critical": significance = "Significant"

        # 2. ST-T Evolution (Ischemia/Infarction)
        # ST Elevation changes
        curr_ste = current_features.get('max_st_elevation', 0.0)
        prev_ste = previous_features.get('max_st_elevation', 0.0)
        
        if curr_ste > prev_ste + 0.1: # > 1mm increase
            changes.append(f"Dynamic ST elevation increase (+{curr_ste - prev_ste:.2f}mV)")
            significance = "Critical"
        elif curr_ste < prev_ste - 0.1:
            changes.append("Resolution of ST elevation")
            
        # T-wave Inversion (New)
        curr_t_inv = current_features.get('leads_with_t_inv', [])
        prev_t_inv = previous_features.get('leads_with_t_inv', [])
        new_t_inv = list(set(curr_t_inv) - set(prev_t_inv))
        
        if new_t_inv:
            changes.append(f"New T-wave inversion in leads: {', '.join(new_t_inv)}")
            if significance != "Critical": significance = "Significant"

        # 3. Q-Wave Development (Infarction)
        curr_q = current_features.get('leads_with_q_waves', [])
        prev_q = previous_features.get('leads_with_q_waves', [])
        new_q = list(set(curr_q) - set(prev_q))
        
        if new_q:
            changes.append(f"New pathological Q waves in: {', '.join(new_q)}")
            if significance == "Stable": significance = "Significant"

        # 4. QT Interval Prolongation (Drug toxicity / Electrolytes)
        curr_qtc = current_features.get('qtc', 400)
        prev_qtc = previous_features.get('qtc', 400)
        delta_qtc = curr_qtc - prev_qtc
        
        if delta_qtc > 60:
            changes.append(f"Marked QTc prolongation (+{int(delta_qtc)}ms)")
            if significance != "Critical": significance = "Significant"
        elif delta_qtc > 30:
            changes.append(f"QTc prolongation (+{int(delta_qtc)}ms)")

        # 5. Rate/Rhythm Changes
        curr_rhythm = current_features.get('rhythm', 'Sinus')
        prev_rhythm = previous_features.get('rhythm', 'Sinus')
        
        if curr_rhythm != prev_rhythm:
            changes.append(f"Rhythm change: {prev_rhythm} -> {curr_rhythm}")
            if curr_rhythm == "AFib" and prev_rhythm == "Sinus":
                significance = "Significant"

        return {
            "status": significance,
            "changes_detected": changes,
            "comparison_summary": f"ECG is {significance}. {'; '.join(changes)}." if changes else "No significant changes compared to previous ECG."
        }

# Example Usage
if __name__ == "__main__":
    comparator = TemporalComparator()
    
    # Baseline
    prev = {
        'is_lbbb': False,
        'max_st_elevation': 0.05,
        'qtc': 420,
        'rhythm': 'Sinus',
        'leads_with_q_waves': []
    }
    
    # Current (New LBBB + STE)
    curr = {
        'is_lbbb': True,
        'max_st_elevation': 0.2,
        'qtc': 430,
        'rhythm': 'Sinus',
        'leads_with_q_waves': []
    }
    
    print(comparator.compare_records(curr, prev))
