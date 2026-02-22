import argparse
import json
import os
import numpy as np
import tensorflow as tf
import pickle
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from google.cloud import storage
import matplotlib.pyplot as plt

def load_data(gcs_path):
    """Loads validation data from GCS."""
    # Mock implementation for demo
    print(f"Loading validation data from {gcs_path}...")
    # In reality: download blob, np.load()
    X = np.random.randn(1000, 5000, 12).astype(np.float32)
    y = np.random.randint(0, 2, size=(1000, 5)) # Multi-label
    return X, y

def calibrate_model(project_id, model_path, data_path, output_bucket, run_id, method='isotonic', mc_samples=1):
    print(f"Starting Calibration ({method}) for Run {run_id}...")
    print(f"MC Dropout Samples: {mc_samples}")
    
    # 1. Load Model
    try:
        model = tf.keras.models.load_model(model_path)
    except:
        print("Model not found locally, using mock model.")
        model = None

    # 2. Load Validation Data
    X_val, y_val = load_data(data_path)
    
    # 3. Get Predictions (Standard or MC Dropout)
    if model:
        if mc_samples > 1:
            print(f"Running MC Dropout Inference ({mc_samples} passes)...")
            mc_preds = []
            # Batch processing for MC Dropout to avoid OOM on large val sets
            batch_size = 32
            dataset = tf.data.Dataset.from_tensor_slices(X_val).batch(batch_size)
            
            for i in range(mc_samples):
                print(f"  Pass {i+1}/{mc_samples}")
                pass_preds = []
                for batch in dataset:
                    # Force training=True to enable Dropout
                    p = model(batch, training=True)
                    if isinstance(p, dict):
                        p = p['pathology']
                    pass_preds.append(p.numpy())
                mc_preds.append(np.concatenate(pass_preds, axis=0))
            
            # Stack: (Samples, N, Classes)
            mc_preds = np.array(mc_preds)
            
            # Calculate Statistics
            preds = np.mean(mc_preds, axis=0) # Use Mean for calibration
            std_dev = np.std(mc_preds, axis=0)
            
            # 95% CI
            ci_lower = np.percentile(mc_preds, 2.5, axis=0)
            ci_upper = np.percentile(mc_preds, 97.5, axis=0)
            avg_ci_width = np.mean(ci_upper - ci_lower, axis=0)
            
            print(f"MC Dropout Complete. Average CI Width per class: {avg_ci_width}")
            
        else:
            preds = model.predict(X_val)
            if isinstance(preds, dict):
                preds = preds['pathology']
            avg_ci_width = np.zeros(preds.shape[1])
    else:
        preds = np.random.rand(len(y_val), 5).astype(np.float32)
        avg_ci_width = np.zeros(5)

    classes = ["Normal", "AFIB", "MI", "PVC", "Noise"]
    calibrators = {}
    brier_scores = {}
    
    # 4. Calibrate Per Class
    plt.figure(figsize=(10, 10))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    for i, cls in enumerate(classes):
        y_true = y_val[:, i]
        y_prob = preds[:, i]
        
        # Brier Score Before
        bs_before = brier_score_loss(y_true, y_prob)
        
        # Train Calibrator
        if method == 'isotonic':
            calibrator = IsotonicRegression(out_of_bounds='clip')
        else: # Platt Scaling (Logistic Regression)
            calibrator = LogisticRegression(C=1.0, solver='lbfgs')
            
        # Isotonic requires 1D input, Logistic requires 2D
        if method == 'isotonic':
            calibrator.fit(y_prob, y_true)
            y_calibrated = calibrator.transform(y_prob)
        else:
            calibrator.fit(y_prob.reshape(-1, 1), y_true)
            y_calibrated = calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]
            
        calibrators[cls] = calibrator
        
        # Brier Score After
        bs_after = brier_score_loss(y_true, y_calibrated)
        brier_scores[cls] = {
            "before": float(bs_before),
            "after": float(bs_after),
            "improvement": float(bs_before - bs_after),
            "uncertainty_ci_width": float(avg_ci_width[i]) if mc_samples > 1 else 0.0
        }
        
        # Plot Reliability Curve (Simplified)
        # In production, use sklearn.calibration.calibration_curve
        from sklearn.calibration import calibration_curve
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_calibrated, n_bins=10)
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"{cls} ({bs_after:.3f})")

    plt.ylabel("Fraction of positives")
    plt.xlabel("Mean predicted value")
    plt.title(f"Calibration Plots ({method})")
    plt.legend()
    
    # 5. Save Artifacts
    # Save Plot
    plot_filename = "calibration_plot.png"
    plt.savefig(plot_filename)
    
    # Save Calibrators (Pickle)
    calibrators_filename = "calibrators.pkl"
    with open(calibrators_filename, 'wb') as f:
        pickle.dump(calibrators, f)
        
    # Save Metrics
    metrics_filename = "brier_scores.json"
    with open(metrics_filename, 'w') as f:
        json.dump(brier_scores, f, indent=4)
        
    # Upload to GCS
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(output_bucket)
    
    base_path = f"calibration/{run_id}"
    bucket.blob(f"{base_path}/{plot_filename}").upload_from_filename(plot_filename)
    bucket.blob(f"{base_path}/{calibrators_filename}").upload_from_filename(calibrators_filename)
    bucket.blob(f"{base_path}/{metrics_filename}").upload_from_filename(metrics_filename)
    
    # Cleanup
    os.remove(plot_filename)
    os.remove(calibrators_filename)
    os.remove(metrics_filename)
    
    print(f"Calibration complete. Artifacts uploaded to gs://{output_bucket}/{base_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--output_bucket', required=True)
    parser.add_argument('--run_id', required=True)
    parser.add_argument('--method', default='isotonic', choices=['isotonic', 'platt'])
    parser.add_argument('--mc_samples', type=int, default=1, help="Number of MC Dropout samples (default: 1 for deterministic)")
    
    args = parser.parse_args()
    calibrate_model(
        args.project_id, args.model_path, args.data_path, 
        args.output_bucket, args.run_id, args.method, args.mc_samples
    )
