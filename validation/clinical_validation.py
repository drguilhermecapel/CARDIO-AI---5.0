import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, cohen_kappa_score, f1_score

class ClinicalValidator:
    def __init__(self, model_name="CardioAI_Nexus_v1"):
        self.model_name = model_name
        self.results = {}

    def calculate_metrics(self, y_true, y_pred, y_prob):
        """
        Calculate comprehensive clinical metrics.
        y_true: Ground truth labels (0/1)
        y_pred: Predicted labels (0/1)
        y_prob: Predicted probabilities (0-1)
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0 # Precision
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        f1 = f1_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_prob)
        
        return {
            "Sensitivity (Recall)": round(sensitivity, 4),
            "Specificity": round(specificity, 4),
            "PPV (Precision)": round(ppv, 4),
            "NPV": round(npv, 4),
            "F1-Score": round(f1, 4),
            "Cohen's Kappa": round(kappa, 4),
            "ROC-AUC": round(roc_auc, 4),
            "Confusion Matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)}
        }

    def run_multicenter_validation(self, datasets):
        """
        Simulate validation across multiple datasets (e.g., PTB-XL, CPSC, CODE, Internal).
        datasets: Dict of {dataset_name: (y_true, y_pred, y_prob)}
        """
        report = f"# Clinical Validation Report - {self.model_name}\n\n"
        
        for name, data in datasets.items():
            y_true, y_pred, y_prob = data
            metrics = self.calculate_metrics(y_true, y_pred, y_prob)
            
            report += f"## Dataset: {name}\n"
            report += "| Metric | Value | Reference (AHA/ACC) |\n"
            report += "|---|---|---|\n"
            report += f"| Sensitivity | {metrics['Sensitivity (Recall)']} | > 0.90 |\n"
            report += f"| Specificity | {metrics['Specificity']} | > 0.85 |\n"
            report += f"| ROC-AUC | {metrics['ROC-AUC']} | > 0.90 |\n"
            report += f"| Cohen's Kappa | {metrics['Cohen's Kappa']} | > 0.80 |\n\n"
            
            # Failure Analysis
            report += "### Failure Analysis\n"
            report += f"- False Negatives: {metrics['Confusion Matrix']['FN']} (Critical Misses)\n"
            report += f"- False Positives: {metrics['Confusion Matrix']['FP']} (Alarm Fatigue Risk)\n\n"

        return report

# Example Usage (Mock Data)
if __name__ == "__main__":
    # Simulate data for STEMI detection
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 1000)
    y_prob = np.random.rand(1000)
    y_pred = (y_prob > 0.5).astype(int)
    
    validator = ClinicalValidator()
    datasets = {
        "PTB-XL (Germany)": (y_true, y_pred, y_prob),
        "CODE (Brazil)": (y_true, y_pred, y_prob), # Mocking same data for example
        "CPSC (China)": (y_true, y_pred, y_prob)
    }
    
    print(validator.run_multicenter_validation(datasets))
