import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import welch
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss, precision_recall_curve, auc, classification_report
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib
import shap
import mlflow
import logging
import time
from typing import Dict, List, Any, Tuple, Union

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CardioAI_ML")

# --- 1. Feature Extractor (100+ Features) ---
class FeatureExtractor:
    """
    Extracts comprehensive features from 12-lead ECG signals.
    Target: >100 features (Statistical, Morphological, Frequency, Wavelet).
    """
    def __init__(self, fs=500):
        self.fs = fs

    def _get_statistical_features(self, signal: np.ndarray, prefix: str) -> Dict[str, float]:
        """Extracts mean, std, skew, kurtosis, min, max, energy."""
        return {
            f"{prefix}_mean": np.mean(signal),
            f"{prefix}_std": np.std(signal),
            f"{prefix}_skew": skew(signal),
            f"{prefix}_kurt": kurtosis(signal),
            f"{prefix}_min": np.min(signal),
            f"{prefix}_max": np.max(signal),
            f"{prefix}_energy": np.sum(signal**2)
        }

    def _get_frequency_features(self, signal: np.ndarray, prefix: str) -> Dict[str, float]:
        """Extracts PSD band powers."""
        f, Pxx = welch(signal, fs=self.fs, nperseg=min(len(signal), 256))
        
        # Bands: VLF (0-0.04), LF (0.04-0.15), HF (0.15-0.4) - usually for HRV (RR intervals)
        # For raw ECG signal, we look at QRS power (5-15Hz), T-wave (0.5-5Hz), Noise (>50Hz)
        
        total_power = np.sum(Pxx)
        if total_power == 0: total_power = 1e-9
        
        p_low = np.sum(Pxx[(f >= 0.5) & (f < 5)])
        p_qrs = np.sum(Pxx[(f >= 5) & (f < 15)])
        p_high = np.sum(Pxx[(f >= 15) & (f < 40)])
        
        return {
            f"{prefix}_p_low": p_low,
            f"{prefix}_p_qrs": p_qrs,
            f"{prefix}_p_high": p_high,
            f"{prefix}_spec_entropy": entropy(Pxx)
        }

    def extract(self, ecg_signal: np.ndarray) -> pd.DataFrame:
        """
        Input: (12, N_samples) numpy array.
        Output: DataFrame with 1 row and ~100+ columns.
        """
        features = {}
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
        # 1. Per-Lead Features (12 * 11 = 132 features)
        for i, lead in enumerate(lead_names):
            sig = ecg_signal[i]
            features.update(self._get_statistical_features(sig, lead))
            features.update(self._get_frequency_features(sig, lead))
            
        # 2. Global Features (Cross-lead)
        # e.g., QT dispersion (max QT - min QT), Axis (requires morphological analysis)
        # For simplicity in this ML extractor, we stick to signal stats.
        
        return pd.DataFrame([features])

# --- 2. Classifiers ---

class RandomForestECGClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, 
            random_state=random_state, 
            n_jobs=-1,
            class_weight='balanced'
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

class SVMECGClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, kernel='rbf', random_state=42):
        self.C = C
        self.kernel = kernel
        self.random_state = random_state
        # Platt Scaling via CalibratedClassifierCV or probability=True
        self.model = SVC(C=C, kernel=kernel, probability=True, random_state=random_state)

    def fit(self, X, y):
        scaler = StandardScaler()
        self.pipeline = Pipeline([('scaler', scaler), ('svm', self.model)])
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

class LSTMCNNECGClassifier(nn.Module):
    """
    Hybrid Deep Learning Model: CNN for spatial/morphological features, LSTM for temporal.
    """
    def __init__(self, input_channels=12, num_classes=2):
        super(LSTMCNNECGClassifier, self).__init__()
        
        # CNN Block
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)
        
        # LSTM Block
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        
        # Dense Block
        self.fc1 = nn.Linear(64 * 2, 64) # Bidirectional
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: (Batch, 12, Samples)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Permute for LSTM: (Batch, Features, Time) -> (Batch, Time, Features)
        x = x.permute(0, 2, 1)
        
        # LSTM
        # output, (hn, cn) = self.lstm(x)
        # Take last time step? Or Global Average Pooling?
        # Let's use Global Average Pooling over time
        x, _ = self.lstm(x)
        x = torch.mean(x, dim=1) # Global Average Pooling
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x # Logits

    def predict_proba(self, x_tensor):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x_tensor)
            return F.softmax(logits, dim=1).numpy()

# --- 3. Ensemble Classifier ---

class EnsembleECGClassifier:
    def __init__(self, models: List[Any], weights: List[float] = None):
        self.models = models
        self.weights = weights if weights else [1/len(models)] * len(models)

    def predict_proba(self, X_features, X_raw_tensor):
        """
        Combines predictions from ML models (using features) and DL models (using raw tensor).
        """
        probs = []
        for model in self.models:
            if isinstance(model, (RandomForestECGClassifier, SVMECGClassifier, Pipeline)):
                p = model.predict_proba(X_features)
            elif isinstance(model, LSTMCNNECGClassifier):
                p = model.predict_proba(X_raw_tensor)
            else:
                p = np.zeros((len(X_features), 2)) # Fallback
            probs.append(p)
            
        # Weighted Average
        avg_prob = np.average(probs, axis=0, weights=self.weights)
        return avg_prob

    def predict(self, X_features, X_raw_tensor, threshold=0.5):
        probs = self.predict_proba(X_features, X_raw_tensor)
        return (probs[:, 1] >= threshold).astype(int)

# --- 4. Model Validator ---

class ModelValidator:
    @staticmethod
    def evaluate(y_true, y_prob, threshold=0.5):
        y_pred = (y_prob >= threshold).astype(int)
        
        roc_auc = roc_auc_score(y_true, y_prob)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        f1 = f1_score(y_true, y_pred)
        brier = brier_score_loss(y_true, y_prob)
        
        report = classification_report(y_true, y_pred, output_dict=True)
        
        metrics = {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "f1_score": f1,
            "brier_score": brier,
            "accuracy": report['accuracy'],
            "sensitivity": report['1']['recall'],
            "specificity": report['0']['recall'] # Assuming binary 0/1
        }
        
        # Log to MLflow
        # mlflow.log_metrics(metrics)
        
        return metrics

# --- 5. Explainability Analyzer ---

class ExplainabilityAnalyzer:
    def __init__(self, model, X_background):
        self.model = model
        self.X_background = X_background
        # Initialize SHAP explainer
        # For Tree models: TreeExplainer
        # For Kernel (SVM): KernelExplainer
        if isinstance(model, RandomForestECGClassifier):
            self.explainer = shap.TreeExplainer(model.model)
        else:
            self.explainer = shap.KernelExplainer(model.predict_proba, X_background)

    def explain_local(self, X_instance):
        """Returns SHAP values for a single instance."""
        shap_values = self.explainer.shap_values(X_instance)
        return shap_values

    def get_top_features(self, shap_values, feature_names, top_n=10):
        # Calculate mean absolute SHAP value per feature
        if isinstance(shap_values, list): # Multiclass
            vals = np.abs(shap_values[1]).mean(0)
        else:
            vals = np.abs(shap_values).mean(0)
            
        feature_importance = pd.DataFrame(list(zip(feature_names, vals)), columns=['col_name', 'feature_importance_vals'])
        feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
        return feature_importance.head(top_n)

# --- Example Usage ---
if __name__ == "__main__":
    # 1. Generate Mock Data
    N_SAMPLES = 100
    X_raw = np.random.randn(N_SAMPLES, 12, 5000).astype(np.float32) # 10s ECG
    y = np.random.randint(0, 2, N_SAMPLES)
    
    # 2. Feature Extraction
    extractor = FeatureExtractor()
    print("Extracting features...")
    start = time.time()
    # Process in loop (inefficient for demo, vectorize in prod)
    features_list = [extractor.extract(X_raw[i]).iloc[0] for i in range(N_SAMPLES)]
    X_features = pd.DataFrame(features_list)
    print(f"Extraction Time: {time.time() - start:.2f}s")
    
    # 3. Train Models
    print("Training RF...")
    rf = RandomForestECGClassifier(n_estimators=10)
    rf.fit(X_features, y)
    
    print("Training SVM...")
    svm = SVMECGClassifier()
    svm.fit(X_features, y)
    
    print("Initializing DL...")
    dl = LSTMCNNECGClassifier()
    # Mock training for DL (skipping loop)
    
    # 4. Ensemble Prediction
    print("Ensemble Prediction...")
    ensemble = EnsembleECGClassifier(models=[rf, svm, dl], weights=[0.4, 0.3, 0.3])
    
    X_raw_tensor = torch.tensor(X_raw)
    probs = ensemble.predict_proba(X_features, X_raw_tensor)
    
    # 5. Validation
    metrics = ModelValidator.evaluate(y, probs[:, 1])
    print("Metrics:", metrics)
    
    # 6. Explainability
    print("Generating Explanations...")
    explainer = ExplainabilityAnalyzer(rf, X_features.iloc[:10]) # Background
    shap_vals = explainer.explain_local(X_features.iloc[0:1])
    top_feats = explainer.get_top_features(shap_vals, X_features.columns)
    print("Top Features:\n", top_feats)
