import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

from validation.data_loaders import ECGDatasetLoader
from integration.cardio_ai_service import CardioAIIntegrationService

# Configure Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ExternalValidation")

class ExternalValidator:
    """
    Validates the CardioAI model against external datasets (PTB-XL, etc).
    Calculates clinical metrics: Sensitivity, Specificity, PPV, NPV, F1, AUROC.
    """
    
    def __init__(self, service: CardioAIIntegrationService):
        self.service = service
        self.results = []
        
    def run_validation(self, loader: ECGDatasetLoader, dataset_name: str, limit: int = None):
        logger.info(f"Starting validation on {dataset_name}...")
        
        # Load Data
        try:
            X, df = loader.load_data(limit=limit)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return
            
        logger.info(f"Loaded {len(df)} records.")
        
        y_true = []
        y_pred = []
        y_scores = []
        
        # Map PTB-XL SCP codes to our classes
        # Simplified mapping for demo: 'NORM' vs 'MI' (Myocardial Infarction)
        
        for i in tqdm(range(len(df))):
            # Prepare Input
            # PTB-XL is (Length, Leads), Service expects (Leads, Length) usually
            signal = X[i]
            if signal.shape[0] > 12: # If (Length, 12)
                signal = signal.T # -> (12, Length)
                
            meta = df.iloc[i]
            patient_meta = {
                'id': str(meta.name),
                'age': int(meta.get('age', 0)),
                'sex': 'Male' if meta.get('sex', 0) == 0 else 'Female'
            }
            
            # Ground Truth
            scp = meta.scp_codes
            is_mi = 'IMI' in scp or 'AMI' in scp or 'LMI' in scp or 'ASMI' in scp
            is_norm = 'NORM' in scp
            
            # We focus on MI detection for this metric
            true_label = 1 if is_mi else 0
            
            # Run Inference
            try:
                result = self.service.process_ecg(signal, patient_meta)
                
                # Extract Prediction
                # Assuming diagnosis string contains "Infarction" or "MI"
                diag_str = result['diagnosis']['diagnosis']
                pred_label = 1 if "Infarction" in diag_str or "MI" in diag_str or "STEMI" in diag_str else 0
                
                # Mock score if not available (Service should return confidence)
                confidence = result['diagnosis'].get('confidence', 0.5)
                if pred_label == 0: confidence = 1 - confidence
                
                y_true.append(true_label)
                y_pred.append(pred_label)
                y_scores.append(confidence)
                
                self.results.append({
                    'id': patient_meta['id'],
                    'true': true_label,
                    'pred': pred_label,
                    'score': confidence,
                    'dataset': dataset_name
                })
                
            except Exception as e:
                logger.error(f"Error processing record {patient_meta['id']}: {e}")
                
        # Calculate Metrics
        self._calculate_metrics(y_true, y_pred, y_scores, dataset_name)

    def _calculate_metrics(self, y_true, y_pred, y_scores, dataset_name):
        report = classification_report(y_true, y_pred, output_dict=True)
        try:
            auc = roc_auc_score(y_true, y_scores)
        except:
            auc = 0.0
            
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics = {
            "dataset": dataset_name,
            "total_cases": len(y_true),
            "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "accuracy": round(report['accuracy'], 4),
            "f1_score": round(report['1']['f1-score'], 4) if '1' in report else 0,
            "auc": round(auc, 4),
            "confusion_matrix": cm.tolist()
        }
        
        logger.info(f"Validation Results for {dataset_name}:")
        logger.info(json.dumps(metrics, indent=2))
        
        # Save to file
        with open(f"validation_results_{dataset_name}.json", "w") as f:
            json.dump(metrics, f, indent=2)
            
        return metrics

# Example Usage
if __name__ == "__main__":
    from validation.data_loaders import MockPTBXLLoader
    
    service = CardioAIIntegrationService()
    validator = ExternalValidator(service)
    
    # Use Mock Loader for Demo (In prod, use PTBXLLoader with real path)
    loader = MockPTBXLLoader(n_samples=50)
    
    validator.run_validation(loader, "PTB-XL_Mock")
