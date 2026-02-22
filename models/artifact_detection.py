import numpy as np
import scipy.signal as signal

class LightweightArtifactDetector:
    def __init__(self, sampling_rate=500):
        self.fs = sampling_rate

    def process_record(self, lead_signal):
        """
        Detects beats and evaluates quality.
        Returns a list of dicts: [{'index': 123, 'is_valid': True}, ...]
        """
        # 1. Bandpass Filter (5-15Hz for QRS energy)
        sos = signal.butter(3, [5, 15], 'bandpass', fs=self.fs, output='sos')
        filtered = signal.sosfiltfilt(sos, lead_signal)
        
        # 2. Energy
        energy = filtered ** 2
        
        # 3. Peak Detection
        # Height threshold: 2x mean energy
        # Distance: 200ms (refractory period)
        peaks, _ = signal.find_peaks(energy, height=np.mean(energy)*2, distance=int(0.2*self.fs))
        
        results = []
        for p in peaks:
            # Check signal quality around peak (window +/- 100ms)
            start = max(0, p - int(0.1*self.fs))
            end = min(len(lead_signal), p + int(0.1*self.fs))
            segment = lead_signal[start:end]
            
            # Simple validity check:
            # 1. Not flatline (std > threshold)
            # 2. Not saturating (max < threshold)
            is_valid = True
            if np.std(segment) < 0.01: is_valid = False
            if np.max(np.abs(segment)) > 5.0: is_valid = False # mV
            
            results.append({'index': int(p), 'is_valid': is_valid})
            
        return results
