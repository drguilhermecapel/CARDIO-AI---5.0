import onnxruntime as ort
import numpy as np
import os
import time

class InferenceEngine:
    def __init__(self, model_path="ecg_mobilenet_optimized.onnx", use_gpu=True):
        self.model_path = model_path
        self.providers = []
        
        if use_gpu and 'CUDAExecutionProvider' in ort.get_available_providers():
            print("Inference Engine: GPU Detected (CUDA).")
            self.providers.append('CUDAExecutionProvider')
        
        if 'TensorrtExecutionProvider' in ort.get_available_providers():
             print("Inference Engine: TensorRT Detected.")
             self.providers.append('TensorrtExecutionProvider')
             
        self.providers.append('CPUExecutionProvider') # Fallback
        
        print(f"Inference Engine: Loading model with providers: {self.providers}")
        
        try:
            self.session = ort.InferenceSession(self.model_path, providers=self.providers)
        except Exception as e:
            print(f"Error loading model: {e}. Falling back to CPU.")
            self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])

        self.input_name = self.session.get_inputs()[0].name

    def predict(self, preprocessed_image):
        """
        Run inference on preprocessed image (numpy array).
        """
        start_time = time.time()
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: preprocessed_image})
        
        latency = (time.time() - start_time) * 1000 # ms
        
        # Assuming output[0] is logits
        logits = outputs[0]
        probabilities = self._softmax(logits)
        
        return {
            "probabilities": probabilities.tolist(),
            "latency_ms": latency,
            "device": self.session.get_providers()[0]
        }

    def _softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1, keepdims=True)

# Singleton Instance
# engine = InferenceEngine()
