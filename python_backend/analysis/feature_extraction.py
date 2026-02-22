import numpy as np
from scipy import signal
from scipy.ndimage import median_filter
import logging
from typing import Dict, List, Any, Tuple, Optional

# Configure Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FeatureExtraction")

class ECGFeatureExtractor:
    """
    Comprehensive ECG Feature Extraction Module.
    Extracts fiducial points, intervals, amplitudes, and morphological features
    from 12-lead ECG signals.
    """
    
    def __init__(self, fs: int = 500):
        self.fs = fs
        self.leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    def _bandpass_filter(self, data: np.ndarray, lowcut: float = 0.5, highcut: float = 50.0, order: int = 4) -> np.ndarray:
        """Butterworth bandpass filter."""
        nyquist = 0.5 * self.fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(order, [low, high], btype='band')
        return signal.filtfilt(b, a, data)

    def _detect_r_peaks(self, lead_data: np.ndarray) -> np.ndarray:
        """
        Pan-Tompkins inspired R-peak detection on a single lead (usually II or V5).
        """
        # 1. Differentiate
        diff = np.diff(lead_data)
        
        # 2. Square
        sq = diff ** 2
        
        # 3. Integrate (Moving Window)
        window_width = int(0.15 * self.fs) # 150ms
        integrated = np.convolve(sq, np.ones(window_width)/window_width, mode='same')
        
        # 4. Find Peaks
        # Min distance 200ms (refractory period)
        peaks, _ = signal.find_peaks(integrated, distance=int(0.2 * self.fs), height=np.mean(integrated))
        
        # 5. Refine peaks (find max in original signal around detected integrated peaks)
        r_peaks = []
        search_window = int(0.05 * self.fs)
        for p in peaks:
            start = max(0, p - search_window)
            end = min(len(lead_data), p + search_window)
            if end > start:
                r_max = np.argmax(lead_data[start:end]) + start
                r_peaks.append(r_max)
                
        return np.array(r_peaks)

    def _delineate_beat(self, beat_window: np.ndarray, r_idx_local: int) -> Dict[str, int]:
        """
        Delineates P, QRS, T waves within a single beat window.
        Uses simple slope and threshold logic relative to R-peak.
        """
        points = {}
        
        # QRS Onset (Q_on) / Offset (S_off)
        # Search backwards from R for Q-onset (slope change)
        # Search forwards from R for S-offset
        
        # Simplified: Threshold based on R-peak height
        # Real implementation would use wavelet transform (WT)
        
        # QRS Offset (J-point)
        # Look for return to baseline or slope flattening after R
        # Approx 40-100ms after R
        search_j = int(0.1 * self.fs)
        s_region = beat_window[r_idx_local : r_idx_local + search_j]
        if len(s_region) > 0:
            s_min_idx = np.argmin(s_region)
            points['S_peak'] = r_idx_local + s_min_idx
            points['J_point'] = points['S_peak'] + int(0.02 * self.fs) # +20ms approx
        else:
            points['J_point'] = r_idx_local + int(0.04 * self.fs)

        # QRS Onset
        search_q = int(0.1 * self.fs)
        q_region = beat_window[max(0, r_idx_local - search_q) : r_idx_local]
        if len(q_region) > 0:
            q_min_idx = np.argmin(q_region)
            points['Q_peak'] = (r_idx_local - search_q) + q_min_idx
            points['QRS_onset'] = points['Q_peak'] - int(0.02 * self.fs)
        else:
            points['QRS_onset'] = r_idx_local - int(0.04 * self.fs)

        # T-wave Peak and Offset
        # Look in 150ms to 500ms after R
        t_start = r_idx_local + int(0.15 * self.fs)
        t_end = r_idx_local + int(0.5 * self.fs)
        
        if t_end < len(beat_window):
            t_region = beat_window[t_start:t_end]
            if len(t_region) > 0:
                # T peak is max abs deviation (could be inverted)
                t_peak_idx = np.argmax(np.abs(t_region))
                points['T_peak'] = t_start + t_peak_idx
                
                # T offset: return to baseline
                # Simple heuristic: +100ms after peak
                points['T_offset'] = points['T_peak'] + int(0.1 * self.fs)
        
        # P-wave Peak and Onset
        # Look in 200ms before QRS onset
        p_end = points.get('QRS_onset', r_idx_local - 50)
        p_start = max(0, p_end - int(0.25 * self.fs))
        
        if p_end > p_start:
            p_region = beat_window[p_start:p_end]
            if len(p_region) > 0:
                p_peak_idx = np.argmax(np.abs(p_region))
                points['P_peak'] = p_start + p_peak_idx
                points['P_onset'] = points['P_peak'] - int(0.04 * self.fs)

        return points

    def _calculate_median_beat(self, lead_data: np.ndarray, r_peaks: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Constructs a median beat from all detected beats to reduce noise.
        Returns (median_beat_signal, r_peak_index_in_median_beat).
        """
        # Window: 250ms before R, 500ms after R
        pre_r = int(0.25 * self.fs)
        post_r = int(0.5 * self.fs)
        
        beats = []
        for r in r_peaks:
            start = r - pre_r
            end = r + post_r
            if start >= 0 and end < len(lead_data):
                beats.append(lead_data[start:end])
                
        if not beats:
            return np.zeros(pre_r + post_r), pre_r
            
        stacked_beats = np.vstack(beats)
        median_beat = np.median(stacked_beats, axis=0)
        return median_beat, pre_r

    def extract(self, ecg_signal: np.ndarray) -> Dict[str, Any]:
        """
        Main extraction method.
        Input: (12, N) numpy array.
        Output: Dictionary of features.
        """
        features = {
            'global': {},
            'leads': {}
        }
        
        # 1. Preprocessing (Filter)
        filtered_ecg = np.array([self._bandpass_filter(lead) for lead in ecg_signal])
        
        # 2. R-peak Detection (Use Lead II or V5)
        # Try Lead II (index 1)
        r_peaks = self._detect_r_peaks(filtered_ecg[1])
        if len(r_peaks) < 2:
            # Try V5 (index 10)
            r_peaks = self._detect_r_peaks(filtered_ecg[10])
            
        if len(r_peaks) < 2:
            logger.warning("Insufficient R-peaks detected.")
            return features

        # Calculate Heart Rate
        rr_intervals = np.diff(r_peaks) / self.fs # seconds
        hr = 60.0 / np.mean(rr_intervals)
        features['global']['hr'] = round(hr, 1)
        features['global']['rr_intervals'] = rr_intervals.tolist()
        
        # 3. Per-Lead Analysis (Median Beat)
        global_intervals = {'pr': [], 'qrs': [], 'qt': []}
        
        for i, lead_name in enumerate(self.leads):
            lead_data = filtered_ecg[i]
            median_beat, r_idx = self._calculate_median_beat(lead_data, r_peaks)
            
            # Delineate Median Beat
            points = self._delineate_beat(median_beat, r_idx)
            
            # Extract Amplitudes (mV) - Assuming signal is in mV
            # If signal is raw ADC, calibration needed. Assuming pre-calibrated float mV.
            amps = {}
            if 'P_peak' in points: amps['p_amp'] = median_beat[points['P_peak']]
            if 'Q_peak' in points: amps['q_amp'] = median_beat[points['Q_peak']]
            if 'R_peak' not in points: points['R_peak'] = r_idx
            amps['r_amp'] = median_beat[points['R_peak']]
            if 'S_peak' in points: amps['s_amp'] = median_beat[points['S_peak']]
            if 'T_peak' in points: amps['t_amp'] = median_beat[points['T_peak']]
            
            # ST Elevation/Depression
            # J-point and J+60ms
            if 'J_point' in points and points['J_point'] < len(median_beat):
                amps['j_point_elev'] = median_beat[points['J_point']]
                
                j_plus_60 = points['J_point'] + int(0.06 * self.fs)
                if j_plus_60 < len(median_beat):
                    amps['st_60_elev'] = median_beat[j_plus_60]
            
            # Store Lead Features
            features['leads'][lead_name] = {
                'amplitudes': amps,
                'points': {k: int(v) for k, v in points.items()}
            }
            
            # Collect Intervals (ms)
            if 'P_onset' in points and 'QRS_onset' in points:
                global_intervals['pr'].append((points['QRS_onset'] - points['P_onset']) / self.fs * 1000)
            
            if 'QRS_onset' in points and 'J_point' in points:
                global_intervals['qrs'].append((points['J_point'] - points['QRS_onset']) / self.fs * 1000)
                
            if 'QRS_onset' in points and 'T_offset' in points:
                global_intervals['qt'].append((points['T_offset'] - points['QRS_onset']) / self.fs * 1000)

        # 4. Global Intervals (Median of leads)
        features['global']['pr_interval'] = np.nanmedian(global_intervals['pr']) if global_intervals['pr'] else 0
        features['global']['qrs_dur'] = np.nanmedian(global_intervals['qrs']) if global_intervals['qrs'] else 0
        features['global']['qt_interval'] = np.nanmedian(global_intervals['qt']) if global_intervals['qt'] else 0
        
        # QTc (Bazett)
        if features['global']['qt_interval'] > 0 and hr > 0:
            rr_sec = 60.0 / hr
            features['global']['qtc'] = features['global']['qt_interval'] / np.sqrt(rr_sec)
            
        # 5. Derived Features for Logic
        # Max ST Elevation
        st_elevs = [f['amplitudes'].get('j_point_elev', 0) for f in features['leads'].values()]
        features['global']['max_st_elevation'] = max(st_elevs) if st_elevs else 0.0
        
        # Leads with STE > 0.1 mV
        features['global']['leads_with_ste'] = [
            lead for lead, f in features['leads'].items() 
            if f['amplitudes'].get('j_point_elev', 0) > 0.1
        ]
        
        # Leads with T-inversion (T amp < -0.1 mV)
        features['global']['leads_with_t_inv'] = [
            lead for lead, f in features['leads'].items() 
            if f['amplitudes'].get('t_amp', 0) < -0.1
        ]
        
        return features

# Example Usage
if __name__ == "__main__":
    extractor = ECGFeatureExtractor()
    
    # Mock Signal (Sine waves)
    t = np.linspace(0, 10, 5000)
    sig = np.zeros((12, 5000))
    # Lead II: P, QRS, T
    # Simple simulation
    beat = np.exp(-((t[:500]-0.2)**2)/0.002) * 0.1 + \
           np.exp(-((t[:500]-0.4)**2)/0.001) * 1.0 + \
           np.exp(-((t[:500]-0.7)**2)/0.005) * 0.2
           
    # Repeat beat
    full_lead = np.tile(beat, 10)
    sig[1] = full_lead[:5000] # Lead II
    sig[10] = full_lead[:5000] # V5
    
    feats = extractor.extract(sig)
    import json
    print(json.dumps(feats['global'], indent=2))
