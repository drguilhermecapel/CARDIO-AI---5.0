import tensorflow as tf
import numpy as np
from sklearn.isotonic import IsotonicRegression
import logging
from typing import List, Dict, Union, Tuple, Optional
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EnsembleECG")

class EnsembleECGAnalyzer:
    """
    Advanced ECG Analysis using an Ensemble of Deep Learning Models.
    
    Integrates CNN, ResNet, and Vision Transformer architectures with 
    confidence calibration and tiered alerting.
    """
    
    def __init__(self, model_paths: Dict[str, str] = None, num_classes: int = 5):
        self.num_classes = num_classes
        self.models = {}
        self.calibrators = {} # Per model, per class
        self.class_names = ["Normal", "AFib", "Other", "Noise", "STEMI"] # Example
        
        # Check GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"GPU Detected: {len(gpus)} device(s).")
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                logger.error(e)
        else:
            logger.info("No GPU detected. Running on CPU.")

        if model_paths:
            self.load_pretrained_models(model_paths)
        else:
            logger.warning("No model paths provided. Initializing with dummy models for demonstration.")
            self._init_dummy_models()

    def _init_dummy_models(self):
        """Initialize un-trained architectures for demo purposes."""
        input_shape = (5000, 12) # 10s ECG @ 500Hz
        self.models['cnn'] = self._build_cnn(input_shape)
        self.models['resnet'] = self._build_resnet(input_shape)
        self.models['vit'] = self._build_vit(input_shape)

    def _build_cnn(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Conv1D(32, 5, activation='relu')(inputs)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
        return tf.keras.Model(inputs, outputs, name="SimpleCNN")

    def _build_resnet(self, input_shape):
        # Simplified ResNet1D
        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Conv1D(64, 7, padding='same')(inputs)
        # ... (ResBlocks would go here)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
        return tf.keras.Model(inputs, outputs, name="ResNet18")

    def _build_vit(self, input_shape):
        # Simplified Transformer
        inputs = tf.keras.Input(shape=input_shape)
        # Patch creation (Conv1D)
        x = tf.keras.layers.Conv1D(64, 10, strides=10)(inputs) 
        # Transformer Encoder
        # MultiHeadAttention requires query, value, key. Here we use self-attention (query=value=key=x)
        # We need to ensure dimensions match.
        # Conv1D output: (Batch, 500, 64)
        x = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=64)(x, x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
        return tf.keras.Model(inputs, outputs, name="ViT")

    def load_pretrained_models(self, paths: Dict[str, str]):
        """
        Loads Keras models from disk.
        
        Args:
            paths: Dict with keys 'cnn', 'resnet', 'vit' and file paths.
        """
        for name, path in paths.items():
            if os.path.exists(path):
                logger.info(f"Loading {name} from {path}...")
                self.models[name] = tf.keras.models.load_model(path)
            else:
                logger.error(f"Model path {path} not found.")
                # Fallback to dummy
                self.models[name] = self._build_cnn((5000, 12))

    def calibrate(self, X_val: np.ndarray, y_val: np.ndarray):
        """
        Fits Isotonic Regression calibrators for each model and class.
        
        Args:
            X_val: Validation signals (N, 5000, 12)
            y_val: One-hot encoded labels (N, num_classes)
        """
        logger.info("Calibrating models...")
        for name, model in self.models.items():
            probs = model.predict(X_val, verbose=0)
            self.calibrators[name] = []
            
            for i in range(self.num_classes):
                # Isotonic Regression for each class (One-vs-Rest)
                iso_reg = IsotonicRegression(out_of_bounds='clip')
                # Fit on predicted prob vs actual binary label
                iso_reg.fit(probs[:, i], y_val[:, i])
                self.calibrators[name].append(iso_reg)
        logger.info("Calibration complete.")

    def _apply_calibration(self, probs: np.ndarray, model_name: str) -> np.ndarray:
        """Applies learned calibration to raw probabilities."""
        if model_name not in self.calibrators:
            return probs
            
        calibrated_probs = np.zeros_like(probs)
        for i in range(self.num_classes):
            # Isotonic Regression predict expects 1D array
            calibrated_probs[:, i] = self.calibrators[model_name][i].predict(probs[:, i])
            
        # Re-normalize to sum to 1
        row_sums = calibrated_probs.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1.0
        return calibrated_probs / row_sums

    def predict(self, ecg_batch: np.ndarray) -> List[Dict]:
        """
        Performs ensemble prediction on a batch of ECGs.
        
        Args:
            ecg_batch: Numpy array of shape (Batch, Samples, Leads)
            
        Returns:
            List of result dictionaries.
        """
        # 1. Get Predictions from all models
        model_preds = {}
        for name, model in self.models.items():
            raw_probs = model.predict(ecg_batch, verbose=0)
            cal_probs = self._apply_calibration(raw_probs, name)
            model_preds[name] = cal_probs

        # 2. Ensemble Aggregation (Soft Voting)
        # We can weigh models here if needed. Currently equal weight.
        # Stack predictions: (Num_Models, Batch, Num_Classes)
        stacked_probs = np.stack(list(model_preds.values()))
        avg_probs = np.mean(stacked_probs, axis=0)
        
        results = []
        for i in range(len(ecg_batch)):
            # Get top class
            class_idx = np.argmax(avg_probs[i])
            confidence = avg_probs[i][class_idx]
            diagnosis = self.class_names[class_idx] if class_idx < len(self.class_names) else "Unknown"
            
            # 3. Tier Logic
            tier, action = self._determine_tier(confidence)
            
            # 4. Uncertainty / Disagreement Analysis
            disagreement = self._calculate_disagreement(model_preds, i, class_idx)
            
            result = {
                "diagnosis": diagnosis,
                "confidence": float(confidence),
                "tier": tier,
                "action_required": action,
                "evidence": {
                    "model_votes": {k: float(v[i][class_idx]) for k, v in model_preds.items()},
                    "disagreement_score": float(disagreement),
                    "is_consensus": disagreement < 0.1
                }
            }
            results.append(result)
            
        return results

    def _determine_tier(self, confidence: float) -> Tuple[int, str]:
        if confidence >= 0.95:
            return 3, "IMMEDIATE ALERT: Response required < 5 min"
        elif confidence >= 0.80:
            return 2, "MANDATORY REVIEW: Physician sign-off required"
        else:
            return 1, "INFORMATIONAL: Clinical correlation suggested"

    def _calculate_disagreement(self, model_preds: Dict[str, np.ndarray], idx: int, class_idx: int) -> float:
        """
        Calculates standard deviation of probabilities across models for the top class.
        """
        probs = [preds[idx][class_idx] for preds in model_preds.values()]
        # Std dev of the probability vectors
        return np.std(probs)

# Example Usage
if __name__ == "__main__":
    # Create Analyzer
    analyzer = EnsembleECGAnalyzer()
    
    # Mock Data (Batch of 5 ECGs)
    X_test = np.random.randn(5, 5000, 12).astype(np.float32)
    
    # Mock Calibration (usually done once with validation set)
    # Create dummy one-hot labels
    y_val = np.eye(5)[np.random.choice(5, 5)] 
    analyzer.calibrate(X_test, y_val)
    
    # Predict
    results = analyzer.predict(X_test)
    
    import json
    print(json.dumps(results, indent=2))
