import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy.special import softmax
from scipy.optimize import minimize_scalar
import logging
from typing import Dict, List, Any, Tuple, Optional, Union

logger = logging.getLogger("ProbabilityCalibrator")

class ProbabilityCalibrator:
    """
    Advanced Probability Calibration for Clinical Models.
    Implements Isotonic Regression, Platt Scaling, and Temperature Scaling.
    Provides reliability analysis with confidence intervals.
    """
    
    def __init__(self, method: str = 'isotonic'):
        self.method = method
        self.calibrators = {} # Per-class calibrators for multi-class
        self.temperature = 1.0 # For Temperature Scaling
        self.is_fitted = False

    def fit(self, 
            probs: np.ndarray, 
            labels: np.ndarray, 
            logits: Optional[np.ndarray] = None) -> 'ProbabilityCalibrator':
        """
        Fits the calibrator on validation data.
        
        Args:
            probs: (N_samples, N_classes) or (N_samples,)
            labels: (N_samples,) or (N_samples, N_classes) one-hot
            logits: (N_samples, N_classes) - Required for Temperature Scaling
        """
        # Handle multi-class via One-vs-Rest for Isotonic/Platt
        if len(probs.shape) > 1 and probs.shape[1] > 1:
            self.n_classes = probs.shape[1]
            if len(labels.shape) == 1:
                labels_onehot = np.eye(self.n_classes)[labels]
            else:
                labels_onehot = labels
                
            if self.method == 'temperature_scaling':
                if logits is None:
                    # Estimate logits from probs (inverse softmax approximation)
                    # Clip to avoid log(0)
                    eps = 1e-15
                    probs = np.clip(probs, eps, 1 - eps)
                    logits = np.log(probs)
                self._fit_temperature_scaling(logits, labels)
            else:
                for i in range(self.n_classes):
                    self._fit_binary(probs[:, i], labels_onehot[:, i], class_idx=i)
        else:
            # Binary case
            self.n_classes = 2
            self._fit_binary(probs, labels, class_idx=0)
            
        self.is_fitted = True
        return self

    def _fit_binary(self, probs: np.ndarray, labels: np.ndarray, class_idx: int):
        """Fits a binary calibrator for a specific class."""
        if self.method == 'isotonic':
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(probs, labels)
        elif self.method == 'platt':
            # Platt scaling is essentially Logistic Regression on the outputs
            calibrator = LogisticRegression(C=1e9, solver='lbfgs')
            # Reshape for sklearn
            calibrator.fit(probs.reshape(-1, 1), labels)
        else:
            raise ValueError(f"Unknown method: {self.method}")
            
        self.calibrators[class_idx] = calibrator

    def _fit_temperature_scaling(self, logits: np.ndarray, labels: np.ndarray):
        """
        Finds optimal temperature T to minimize NLL.
        """
        # Labels should be class indices for NLL calculation usually, 
        # but let's stick to cross-entropy with one-hot if provided
        if len(labels.shape) > 1:
            y_true = np.argmax(labels, axis=1)
        else:
            y_true = labels

        def nll(t):
            # Apply temperature
            scaled_logits = logits / t
            # Softmax
            # exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            # probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            probs = softmax(scaled_logits, axis=1)
            
            # NLL
            # Select prob of true class
            # p_true = probs[np.arange(len(y_true)), y_true]
            # loss = -np.mean(np.log(p_true + 1e-15))
            
            # Using sklearn log_loss for stability
            from sklearn.metrics import log_loss
            loss = log_loss(y_true, probs)
            return loss

        # Optimize T > 0
        res = minimize_scalar(nll, bounds=(0.01, 10.0), method='bounded')
        self.temperature = res.x
        logger.info(f"Optimal Temperature found: {self.temperature:.4f}")

    def calibrate(self, 
                  probs: np.ndarray, 
                  logits: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Returns calibrated probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted.")

        if self.method == 'temperature_scaling':
            if logits is None:
                eps = 1e-15
                probs = np.clip(probs, eps, 1 - eps)
                logits = np.log(probs)
            
            scaled_logits = logits / self.temperature
            return softmax(scaled_logits, axis=1)
            
        else:
            calibrated_probs = np.zeros_like(probs)
            if len(probs.shape) > 1:
                for i in range(self.n_classes):
                    if self.method == 'isotonic':
                        calibrated_probs[:, i] = self.calibrators[i].predict(probs[:, i])
                    elif self.method == 'platt':
                        calibrated_probs[:, i] = self.calibrators[i].predict_proba(probs[:, i].reshape(-1, 1))[:, 1]
                
                # Re-normalize to sum to 1
                row_sums = calibrated_probs.sum(axis=1, keepdims=True)
                calibrated_probs = calibrated_probs / row_sums
                return calibrated_probs
            else:
                # Binary
                if self.method == 'isotonic':
                    return self.calibrators[0].predict(probs)
                elif self.method == 'platt':
                    return self.calibrators[0].predict_proba(probs.reshape(-1, 1))[:, 1]

    def assess_reliability(self, 
                           y_true: np.ndarray, 
                           y_prob: np.ndarray, 
                           n_bins: int = 10,
                           n_bootstrap: int = 1000) -> Dict[str, Any]:
        """
        Calculates ECE, MCE, and Reliability Curve with Confidence Intervals.
        """
        # For multi-class, we can assess per-class or overall (top-label)
        # Here we implement Top-Label Calibration (common in Deep Learning)
        # or Per-Class. Let's do Per-Class average for clinical detail.
        
        if len(y_prob.shape) > 1:
            # Flatten for "marginal" calibration or iterate
            # Let's do Top-Class Calibration (Confidence vs Accuracy)
            preds = np.argmax(y_prob, axis=1)
            confs = np.max(y_prob, axis=1)
            
            if len(y_true.shape) > 1:
                y_true_cls = np.argmax(y_true, axis=1)
            else:
                y_true_cls = y_true
                
            # Accuracy matches
            acc_match = (preds == y_true_cls).astype(int)
            
            return self._calculate_metrics(acc_match, confs, n_bins, n_bootstrap)
        else:
            return self._calculate_metrics(y_true, y_prob, n_bins, n_bootstrap)

    def _calculate_metrics(self, 
                           y_true: np.ndarray, 
                           y_prob: np.ndarray, 
                           n_bins: int,
                           n_bootstrap: int) -> Dict[str, Any]:
        
        # Binning
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        binids = np.digitize(y_prob, bins) - 1
        
        bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
        bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
        bin_total = np.bincount(binids, minlength=len(bins))
        
        nonzero = bin_total > 0
        prob_pred = bin_sums[nonzero] / bin_total[nonzero]
        prob_true = bin_true[nonzero] / bin_total[nonzero]
        
        # ECE & MCE
        ece = np.sum(np.abs(prob_pred - prob_true) * (bin_total[nonzero] / len(y_true)))
        mce = np.max(np.abs(prob_pred - prob_true))
        
        # Brier Score
        brier = np.mean((y_prob - y_true) ** 2)
        
        # Bootstrap for CI
        ece_scores = []
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(y_true), len(y_true), replace=True)
            # Re-calculate ECE on sample
            # (Simplified for speed: just re-bin)
            b_prob = y_prob[indices]
            b_true = y_true[indices]
            b_binids = np.digitize(b_prob, bins) - 1
            b_bin_sums = np.bincount(b_binids, weights=b_prob, minlength=len(bins))
            b_bin_true = np.bincount(b_binids, weights=b_true, minlength=len(bins))
            b_bin_total = np.bincount(b_binids, minlength=len(bins))
            
            b_nonzero = b_bin_total > 0
            b_pred = b_bin_sums[b_nonzero] / b_bin_total[b_nonzero]
            b_actual = b_bin_true[b_nonzero] / b_bin_total[b_nonzero]
            
            b_ece = np.sum(np.abs(b_pred - b_actual) * (b_bin_total[b_nonzero] / len(b_true)))
            ece_scores.append(b_ece)
            
        ece_ci = np.percentile(ece_scores, [2.5, 97.5])
        
        # Over/Under Confidence
        # Avg Confidence vs Avg Accuracy
        avg_conf = np.mean(y_prob)
        avg_acc = np.mean(y_true)
        bias = avg_conf - avg_acc # >0 Overconfident, <0 Underconfident
        
        return {
            "ECE": round(ece, 4),
            "ECE_95_CI": [round(ece_ci[0], 4), round(ece_ci[1], 4)],
            "MCE": round(mce, 4),
            "Brier": round(brier, 4),
            "Bias": round(bias, 4),
            "Status": "Overconfident" if bias > 0.02 else ("Underconfident" if bias < -0.02 else "Calibrated"),
            "Curve": {
                "prob_pred": prob_pred.tolist(),
                "prob_true": prob_true.tolist(),
                "bin_counts": bin_total[nonzero].tolist()
            }
        }

# Example Usage
if __name__ == "__main__":
    calibrator = ProbabilityCalibrator(method='isotonic')
    
    # Mock Data (Uncalibrated)
    # Model is overconfident: predicts 0.9 but accuracy is 0.7
    probs = np.random.uniform(0.6, 1.0, 1000)
    # True labels generated with lower probability
    labels = np.random.binomial(1, probs * 0.8) 
    
    calibrator.fit(probs, labels)
    cal_probs = calibrator.calibrate(probs)
    
    metrics = calibrator.assess_reliability(labels, cal_probs)
    import json
    print(json.dumps(metrics, indent=2))
