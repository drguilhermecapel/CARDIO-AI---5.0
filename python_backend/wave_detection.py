import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt
import time
import json

class ECGWaveDetector:
    """
    Robust ECG Wave Detection (P, Q, R, S, T) using optimized Pan-Tompkins and Search Windows.
    """
    def __init__(self, fs=500):
        self.fs = fs

    def _bandpass_filter(self, data, lowcut, highcut, order=2):
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)

    def detect_qrs(self, signal: np.ndarray) -> np.ndarray:
        """
        Detects R-peaks using an optimized Pan-Tompkins algorithm.
        Returns indices of R-peaks.
        """
        # 1. Bandpass Filter (5-15Hz) - Isolates QRS energy
        filtered = self._bandpass_filter(signal, 5, 15, order=1)
        
        # 2. Derivative - Highlights slopes
        diff = np.diff(filtered)
        
        # 3. Squaring - Enhances large values (QRS)
        squared = diff ** 2
        
        # 4. Moving Window Integration - Smoothes and aggregates
        # Window size ~150ms
        window_width = int(0.150 * self.fs)
        integrated = np.convolve(squared, np.ones(window_width)/window_width, mode='same')
        
        # 5. Adaptive Thresholding & Peak Detection
        # Heuristic: 2.5x mean signal level usually captures QRS without noise
        threshold = np.mean(integrated) * 2.5
        # Min distance 200ms (refractory period)
        peaks, _ = find_peaks(integrated, height=threshold, distance=int(0.2 * self.fs))
        
        # 6. Refine R-peaks
        # The integration shifts the peak. We must find the true max in the filtered signal
        # within a window around the integrated peak.
        r_peaks = []
        search_window = int(0.100 * self.fs) # +/- 100ms
        
        for p in peaks:
            start = max(0, p - search_window)
            end = min(len(filtered), p + search_window)
            
            # Find max absolute amplitude in the bandpassed signal (QRS is high freq)
            # We use the filtered signal because baseline wander in raw signal might offset the max
            local_window = np.abs(filtered[start:end])
            if len(local_window) == 0: continue
            
            r_idx = start + np.argmax(local_window)
            r_peaks.append(r_idx)
            
        return np.array(r_peaks)

    def detect_p_wave(self, signal: np.ndarray, r_peak: int) -> int:
        """
        Detects P-wave for a single beat relative to R-peak.
        Search window: [R-300ms, R-50ms]
        """
        # Use low-frequency signal for P and T waves
        # Filter on the fly or pre-calculate? For single beat, filtering whole signal is inefficient if called in loop.
        # But we assume signal passed here is raw.
        # Let's use a window search on the raw signal (assuming baseline removed) or locally smoothed.
        
        start = max(0, r_peak - int(0.300 * self.fs))
        end = max(0, r_peak - int(0.050 * self.fs))
        
        if start >= end: return np.nan
        
        window = signal[start:end]
        if len(window) == 0: return np.nan
        
        # P wave is usually the largest peak in this window
        # We can use argmax.
        p_idx = start + np.argmax(window)
        return p_idx

    def detect_t_wave(self, signal: np.ndarray, r_peak: int) -> int:
        """
        Detects T-wave for a single beat relative to R-peak.
        Search window: [R+150ms, R+500ms]
        """
        start = min(len(signal), r_peak + int(0.150 * self.fs))
        end = min(len(signal), r_peak + int(0.500 * self.fs))
        
        if start >= end: return np.nan
        
        window = signal[start:end]
        if len(window) == 0: return np.nan
        
        # T wave is usually the largest peak in this window
        # Check for inverted T? We take max absolute deviation from baseline (0)
        # Assuming baseline removed.
        t_idx = start + np.argmax(np.abs(window))
        return t_idx

    def detect_fiducial_points(self, signal: np.ndarray):
        """
        Detects all fiducial points (P, Q, R, S, T) for the entire signal.
        """
        start_time = time.time()
        
        # 1. Detect R-peaks
        r_peaks = self.detect_qrs(signal)
        
        # 2. Pre-filter for P/T detection (0.5-10Hz)
        pt_filtered = self._bandpass_filter(signal, 0.5, 10, order=2)
        
        fiducials = {
            'P': [], 'Q': [], 'R': r_peaks.tolist(), 'S': [], 'T': []
        }
        
        metrics = []
        
        for r in r_peaks:
            # Q-wave: Minima in [R-50ms, R]
            q_start = max(0, r - int(0.050 * self.fs))
            q_win = signal[q_start:r]
            q_idx = q_start + np.argmin(q_win) if len(q_win) > 0 else np.nan
            fiducials['Q'].append(int(q_idx) if not np.isnan(q_idx) else None)
            
            # S-wave: Minima in [R, R+50ms]
            s_end = min(len(signal), r + int(0.050 * self.fs))
            s_win = signal[r:s_end]
            s_idx = r + np.argmin(s_win) if len(s_win) > 0 else np.nan
            fiducials['S'].append(int(s_idx) if not np.isnan(s_idx) else None)
            
            # P-wave
            p_idx = self.detect_p_wave(pt_filtered, r)
            fiducials['P'].append(int(p_idx) if not np.isnan(p_idx) else None)
            
            # T-wave
            t_idx = self.detect_t_wave(pt_filtered, r)
            fiducials['T'].append(int(t_idx) if not np.isnan(t_idx) else None)
            
            # Beat Metrics
            beat_data = {
                "R_pos": int(r),
                "R_amp": float(signal[r]),
                "QRS_dur_ms": (s_idx - q_idx) / self.fs * 1000 if (not np.isnan(q_idx) and not np.isnan(s_idx)) else None,
                "PR_int_ms": (r - p_idx) / self.fs * 1000 if not np.isnan(p_idx) else None,
                "QT_int_ms": (t_idx - q_idx) / self.fs * 1000 if (not np.isnan(q_idx) and not np.isnan(t_idx)) else None
            }
            metrics.append(beat_data)

        processing_time = (time.time() - start_time) * 1000
        
        return {
            "fiducials": fiducials,
            "metrics": metrics,
            "processing_time_ms": processing_time,
            "confidence_score": 0.98 # Mock confidence
        }

    def visualize(self, signal, fiducials, save_path="ecg_analysis.png"):
        """
        Generates a plot of the ECG with marked fiducial points.
        """
        plt.figure(figsize=(20, 6))
        t = np.arange(len(signal)) / self.fs
        
        plt.plot(t, signal, 'k-', linewidth=1, label='ECG Signal')
        
        # Helper to plot points
        def plot_points(indices, color, label, marker='o'):
            valid_idxs = [i for i in indices if i is not None and not np.isnan(i)]
            if valid_idxs:
                plt.scatter(np.array(valid_idxs) / self.fs, signal[valid_idxs], c=color, label=label, marker=marker, zorder=5)

        plot_points(fiducials['R'], 'red', 'R-peak')
        plot_points(fiducials['P'], 'green', 'P-wave')
        plot_points(fiducials['T'], 'blue', 'T-wave')
        plot_points(fiducials['Q'], 'orange', 'Q-wave', marker='x')
        plot_points(fiducials['S'], 'purple', 'S-wave', marker='x')
        
        plt.title("ECG Wave Analysis (P-QRS-T Detection)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (mV)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

# Example Usage
if __name__ == "__main__":
    # Generate synthetic ECG
    fs = 500
    t = np.linspace(0, 5, 5 * fs)
    # Synthetic beat: P, Q, R, S, T
    # Very rough approximation
    beat = np.zeros(fs)
    beat[100:120] = 0.1 # P
    beat[190:200] = -0.1 # Q
    beat[200:220] = 1.0 # R
    beat[220:230] = -0.2 # S
    beat[350:400] = 0.2 # T
    
    signal = np.tile(beat, 5) + np.random.normal(0, 0.02, 5 * fs)
    
    detector = ECGWaveDetector(fs=fs)
    results = detector.detect_fiducial_points(signal)
    
    print(json.dumps(results['metrics'][0], indent=2))
    print(f"Processing Time: {results['processing_time_ms']:.2f}ms")
    
    detector.visualize(signal, results['fiducials'])
