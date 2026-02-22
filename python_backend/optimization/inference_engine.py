import tensorflow as tf
import numpy as np
import time
import logging
import os
import hashlib
from typing import List, Dict, Any, Union, Optional
from functools import lru_cache

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OptimizedInference")

class OptimizedECGInference:
    """
    High-Performance Inference Engine for ECG Analysis.
    Supports:
    - Mixed Precision (Float16)
    - TFLite Quantization (INT8)
    - Batch Processing
    - Caching
    """
    
    def __init__(self, model_path: str = None, quantize: bool = False, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self.quantize = quantize
        self.model = None
        self.interpreter = None # For TFLite
        self.input_details = None
        self.output_details = None
        self.latencies = []
        
        # 1. GPU / Mixed Precision Setup
        if self.use_gpu:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    # Enable Mixed Precision
                    policy = tf.keras.mixed_precision.Policy('mixed_float16')
                    tf.keras.mixed_precision.set_global_policy(policy)
                    logger.info(f"Mixed Precision enabled. GPUs: {len(gpus)}")
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except Exception as e:
                    logger.error(f"GPU Setup Error: {e}")
            else:
                logger.warning("No GPU found. Falling back to CPU.")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            logger.info("GPU disabled by user.")

        # 2. Model Loading / Creation
        if model_path and os.path.exists(model_path):
            if model_path.endswith('.tflite'):
                self._load_tflite(model_path)
            else:
                self._load_keras(model_path)
                if self.quantize:
                    self._convert_and_load_tflite()
        else:
            logger.warning("Model path not found. Initializing Dummy Model.")
            self._init_dummy_model()
            if self.quantize:
                self._convert_and_load_tflite()

        # 3. Cache Setup
        self.cache = {}
        self.cache_capacity = 1000

    def _init_dummy_model(self):
        """Creates a simple model for testing."""
        inputs = tf.keras.Input(shape=(5000, 12))
        x = tf.keras.layers.Conv1D(32, 5, activation='relu')(inputs)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        outputs = tf.keras.layers.Dense(5, activation='softmax')(x)
        self.model = tf.keras.Model(inputs, outputs)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')

    def _load_keras(self, path: str):
        logger.info(f"Loading Keras model from {path}")
        self.model = tf.keras.models.load_model(path)

    def _load_tflite(self, path: str):
        logger.info(f"Loading TFLite model from {path}")
        self.interpreter = tf.lite.Interpreter(model_path=path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def _convert_and_load_tflite(self):
        """Converts current Keras model to TFLite INT8 and loads it."""
        logger.info("Quantizing model to INT8 TFLite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Representative dataset for full INT8 quantization
        def representative_dataset():
            for _ in range(10):
                data = np.random.rand(1, 5000, 12).astype(np.float32)
                yield [data]

        converter.representative_dataset = representative_dataset
        # Ensure full integer quantization
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8  # or tf.float32 depending on input requirement
        converter.inference_output_type = tf.int8

        try:
            tflite_model = converter.convert()
            # Save temporary
            with open("temp_quant.tflite", "wb") as f:
                f.write(tflite_model)
            self._load_tflite("temp_quant.tflite")
            self.model = None # Free up memory
        except Exception as e:
            logger.error(f"Quantization failed: {e}. Reverting to Keras model.")

    def _get_cache_key(self, ecg: np.ndarray) -> str:
        # Hash the array content for caching
        return hashlib.md5(ecg.tobytes()).hexdigest()

    def predict_single(self, ecg: np.ndarray) -> Dict[str, Any]:
        """
        Predicts for a single ECG (12, 5000) or (5000, 12).
        Returns dict with prediction and latency.
        """
        start_time = time.time()
        
        # Input Validation & Reshaping
        if ecg.shape == (12, 5000):
            ecg = ecg.T # (5000, 12)
        
        if ecg.shape != (5000, 12):
             return {"error": f"Invalid shape {ecg.shape}"}

        # Check Cache
        cache_key = self._get_cache_key(ecg)
        if cache_key in self.cache:
            latency = (time.time() - start_time) * 1000
            self.latencies.append(latency)
            return {**self.cache[cache_key], "latency_ms": latency, "cached": True}

        # Inference
        input_data = np.expand_dims(ecg, axis=0).astype(np.float32)
        
        if self.interpreter:
            # TFLite Inference
            # Check input type expected by interpreter
            input_type = self.input_details[0]['dtype']
            if input_type == np.int8:
                # Simple quantization for input if needed (scale/zero_point)
                scale, zero_point = self.input_details[0]['quantization']
                input_data = (input_data / scale + zero_point).astype(np.int8)

            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Dequantize output if needed
            if self.output_details[0]['dtype'] == np.int8:
                scale, zero_point = self.output_details[0]['quantization']
                output_data = (output_data.astype(np.float32) - zero_point) * scale
                
            prediction = output_data[0]
            
        else:
            # Keras Inference
            prediction = self.model.predict(input_data, verbose=0)[0]

        latency = (time.time() - start_time) * 1000
        self.latencies.append(latency)
        
        result = {
            "prediction": prediction.tolist(),
            "confidence": float(np.max(prediction)),
            "latency_ms": latency,
            "cached": False
        }
        
        # Update Cache
        if len(self.cache) >= self.cache_capacity:
            self.cache.pop(next(iter(self.cache))) # Simple FIFO for demo (LRU better)
        self.cache[cache_key] = result
        
        return result

    def predict_batch(self, ecg_list: List[np.ndarray]) -> Dict[str, Any]:
        """
        Batch inference optimization.
        """
        start_time = time.time()
        
        # Preprocess batch
        processed_batch = []
        for ecg in ecg_list:
            if ecg.shape == (12, 5000):
                ecg = ecg.T
            processed_batch.append(ecg)
            
        batch_input = np.array(processed_batch).astype(np.float32)
        
        if self.interpreter:
            # TFLite doesn't support dynamic batching easily without resizing
            # Loop for TFLite (slower than GPU batching)
            results = [self.predict_single(ecg) for ecg in ecg_list]
            predictions = [r['prediction'] for r in results]
        else:
            # Keras GPU Batching
            predictions = self.model.predict(batch_input, verbose=0)
            
        total_time = (time.time() - start_time) * 1000
        avg_latency = total_time / len(ecg_list)
        
        return {
            "predictions": predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
            "total_time_ms": total_time,
            "avg_latency_ms": avg_latency,
            "batch_size": len(ecg_list)
        }

    def benchmark(self, num_iterations: int = 100) -> Dict[str, Any]:
        """
        Runs a benchmark suite.
        """
        logger.info(f"Starting benchmark with {num_iterations} iterations...")
        self.latencies = [] # Reset
        
        # Warmup
        warmup_data = np.random.randn(5000, 12).astype(np.float32)
        self.predict_single(warmup_data)
        
        # Run
        for _ in range(num_iterations):
            data = np.random.randn(5000, 12).astype(np.float32)
            self.predict_single(data)
            
        latencies = np.array(self.latencies)
        
        report = {
            "p50_ms": np.percentile(latencies, 50),
            "p95_ms": np.percentile(latencies, 95),
            "p99_ms": np.percentile(latencies, 99),
            "mean_ms": np.mean(latencies),
            "min_ms": np.min(latencies),
            "max_ms": np.max(latencies),
            "throughput_req_sec": 1000 / np.mean(latencies),
            "quantization": "INT8" if self.interpreter else "None (Float32/16)",
            "gpu_enabled": self.use_gpu
        }
        
        return report

# Example Usage Script
if __name__ == "__main__":
    # 1. Initialize Engine (Simulate Quantization)
    engine = OptimizedECGInference(quantize=True, use_gpu=False) # CPU for TFLite demo
    
    # 2. Run Benchmark
    report = engine.benchmark(100)
    
    # 3. Output Report
    import json
    print(json.dumps(report, indent=2))
    
    # 4. Batch Test
    batch_data = [np.random.randn(5000, 12).astype(np.float32) for _ in range(10)]
    batch_res = engine.predict_batch(batch_data)
    print(f"\nBatch (10) Avg Latency: {batch_res['avg_latency_ms']:.2f} ms")
