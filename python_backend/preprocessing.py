import numpy as np
import cv2
import pywt
from scipy.signal import butter, filtfilt

class ECGPreprocessor:
    def __init__(self, target_size=(1024, 512)):
        self.target_size = target_size

    def normalize_image(self, image_array):
        """
        Normalize image intensity to 0-1 range.
        """
        return image_array.astype(np.float32) / 255.0

    def remove_grid(self, image):
        """
        Remove ECG grid lines using morphological operations (HSV color masking).
        Assumes standard red/pink grid or black grid.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Mask for red/pink grid
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        grid_mask = mask1 + mask2
        
        # Inpaint to remove grid
        clean_image = cv2.inpaint(image, grid_mask, 3, cv2.INPAINT_TELEA)
        return clean_image

    def apply_wavelet_denoising(self, signal, wavelet='db4', level=1):
        """
        Apply Discrete Wavelet Transform (DWT) for denoising.
        """
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        # Thresholding detail coefficients
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(signal)))
        
        new_coeffs = list(map(lambda x: pywt.threshold(x, threshold, mode='soft'), coeffs))
        return pywt.waverec(new_coeffs, wavelet)

    def detect_qrs_pan_tompkins(self, ecg_signal, fs=500):
        """
        Simplified Pan-Tompkins algorithm for QRS detection.
        """
        # 1. Bandpass Filter (5-15Hz)
        nyquist = 0.5 * fs
        low = 5 / nyquist
        high = 15 / nyquist
        b, a = butter(1, [low, high], btype='band')
        filtered = filtfilt(b, a, ecg_signal)
        
        # 2. Derivative
        diff = np.diff(filtered)
        
        # 3. Squaring
        squared = diff ** 2
        
        # 4. Moving Integration
        window_size = int(0.150 * fs)
        integrated = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
        
        # 5. Thresholding
        threshold = np.mean(integrated) * 2
        peaks = np.where(integrated > threshold)[0]
        
        # Debouncing (Refractory period 200ms)
        qrs_peaks = []
        last_peak = -999
        refractory = int(0.200 * fs)
        
        for p in peaks:
            if p - last_peak > refractory:
                qrs_peaks.append(p)
                last_peak = p
                
        return qrs_peaks

    def preprocess_pipeline(self, image_bytes):
        """
        Full pipeline: Decode -> Remove Grid -> Resize -> Normalize.
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Invalid image data")
            
        # 1. Grid Removal (Optional, depends on model training)
        # img = self.remove_grid(img)
        
        # 2. Resize
        img = cv2.resize(img, self.target_size)
        
        # 3. Normalize
        img = self.normalize_image(img)
        
        # 4. Transpose to CHW (Channel, Height, Width) for PyTorch/ONNX
        img = np.transpose(img, (2, 0, 1))
        
        # 5. Add Batch Dimension
        img = np.expand_dims(img, axis=0)
        
        return img
