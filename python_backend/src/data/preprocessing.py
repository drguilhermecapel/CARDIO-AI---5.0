# Código crítico de pré-processamento
import scipy.signal as sig
import numpy as np

class ECGPreprocessor:
    def __init__(self, sampling_rate=500):
        self.fs = sampling_rate
    
    def apply_butterworth_filter(self, signal, lowcut=0.5, highcut=150):
        """Filtragem Butterworth de 4ª ordem"""
        sos = sig.butter(4, [lowcut, highcut], btype='band', fs=self.fs, output='sos')
        return sig.sosfilt(sos, signal)
    
    def remove_baseline_wander(self, signal, poly_order=5):
        """Remove baseline wandering via ajuste polinomial"""
        window = min(len(signal)//2 * 2 - 1, 2001)
        coefs = np.polyfit(np.arange(len(signal)), signal, poly_order)
        baseline = np.polyval(coefs, np.arange(len(signal)))
        return signal - baseline
    
    def remove_powerline_noise(self, signal, freq=50):
        """Remove ruído de 50Hz (Brasil)"""
        b, a = sig.butter(4, [freq-1, freq+1], btype='bandstop', fs=self.fs)
        return sig.filtfilt(b, a, signal)
    
    def preprocess_complete(self, signal_raw):
        """Pipeline completo: filtragem → baseline → normalização"""
        sig_filtered = self.apply_butterworth_filter(signal_raw)
        sig_no_baseline = self.remove_baseline_wander(sig_filtered)
        sig_no_powerline = self.remove_powerline_noise(sig_no_baseline)
        sig_normalized = (sig_no_powerline - np.mean(sig_no_powerline)) / (np.std(sig_no_powerline) + 1e-8)
        return sig_normalized
