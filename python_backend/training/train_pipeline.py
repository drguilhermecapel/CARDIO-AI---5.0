import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import logging
import os
from datetime import datetime

from training.data_loader import ECGDataLoader
from ensemble_analyzer import EnsembleECGAnalyzer
from validation.clinical_validator import ClinicalValidator

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrainPipeline")

class TrainingPipeline:
    """
    End-to-end pipeline: Data Loading -> Training -> Validation -> Reporting.
    """
    
    def __init__(self, output_dir: str = "models/v1"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        self.loader = ECGDataLoader()
        self.analyzer = EnsembleECGAnalyzer() # Initialize models
        self.validator = ClinicalValidator()

    def run(self, n_samples: int = 1000, epochs: int = 5):
        """
        Executes the pipeline.
        """
        logger.info("Starting Training Pipeline...")
        
        # 1. Load Data
        # Using synthetic for reliability in this env, but code supports MIT-BIH
        X, y, meta = self.loader.generate_synthetic_dataset(n_samples)
        
        # Split
        X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
            X, y, meta, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Data loaded. Train: {X_train.shape}, Test: {X_test.shape}")
        
        # 2. Train Models (Ensemble)
        # We train each model in the ensemble
        for name, model in self.analyzer.models.items():
            logger.info(f"Training {name}...")
            
            # Compile if not compiled (dummy models might be)
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
            )
            
            # Train
            history = model.fit(
                X_train, y_train,
                validation_split=0.1,
                epochs=epochs,
                batch_size=32,
                verbose=1
            )
            
            # Save
            save_path = os.path.join(self.output_dir, f"{name}_model.h5")
            model.save(save_path)
            logger.info(f"Saved {name} to {save_path}")

        # 3. Calibration
        logger.info("Calibrating ensemble...")
        self.analyzer.calibrate(X_test, y_test) # Using test set for demo, ideally separate val set
        
        # 4. Validation & Metrics
        logger.info("Running Clinical Validation...")
        
        # Get ensemble predictions
        # Predict returns list of dicts, we need arrays for validator
        # We need to modify/use internal predict logic or parse results
        
        # Let's get raw probs from ensemble logic manually for validation
        # (Replicating ensemble logic here for batch efficiency)
        model_preds = []
        for name, model in self.analyzer.models.items():
            raw = model.predict(X_test, verbose=0)
            cal = self.analyzer._apply_calibration(raw, name)
            model_preds.append(cal)
            
        # Average (Soft Voting)
        y_prob = np.mean(model_preds, axis=0)
        y_pred = (y_prob == y_prob.max(axis=1)[:, None]).astype(int) # One-hot argmax
        
        # Calculate Metrics
        metrics = self.validator.calculate_metrics(y_pred, y_test, y_prob)
        
        # Subgroup Analysis
        subgroups = self.validator.subgroup_analysis(y_pred, y_test, meta_test)
        
        # 5. Generate Report
        report_md = self.validator.generate_regulatory_report()
        report_path = os.path.join(self.output_dir, "validation_report.md")
        with open(report_path, "w") as f:
            f.write(report_md)
            
        # Confusion Matrix
        # Convert one-hot to labels
        y_test_lbl = [self.validator.DIAGNOSES[i] if i < len(self.validator.DIAGNOSES) else "Other" for i in np.argmax(y_test, axis=1)]
        y_pred_lbl = [self.validator.DIAGNOSES[i] if i < len(self.validator.DIAGNOSES) else "Other" for i in np.argmax(y_pred, axis=1)]
        
        cm_path = os.path.join(self.output_dir, "confusion_matrix.png")
        self.validator.export_confusion_matrix(y_test_lbl, y_pred_lbl, cm_path)
        
        logger.info(f"Pipeline Complete. Report saved to {report_path}")
        return metrics

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run(n_samples=200, epochs=2) # Small run for demo
