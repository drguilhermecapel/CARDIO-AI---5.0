import argparse
import json
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, recall_score, precision_score
from scipy import signal as scipy_signal
from google.cloud import storage

def load_data(gcs_path):
    """Loads validation data from GCS."""
    # Mock implementation for demo
    print(f"Loading validation data from {gcs_path}...")
    # In reality: download blob, np.load()
    X = np.random.randn(200, 5000, 12).astype(np.float32)
    y = np.random.randint(0, 2, size=(200, 5)) # Multi-label
    return X, y

def add_noise(signal, snr_db):
    """Adds Gaussian noise to the signal for a given SNR."""
    # Signal power
    s_power = np.mean(signal ** 2, axis=1, keepdims=True)
    # Noise power
    n_power = s_power / (10 ** (snr_db / 10))
    # Generate noise
    noise = np.random.normal(0, np.sqrt(n_power), signal.shape)
    return signal + noise

def drop_leads(signal, lead_indices):
    """Zeros out specified leads."""
    perturbed = signal.copy()
    # signal shape: (Batch, Time, Leads)
    for idx in lead_indices:
        perturbed[:, :, idx] = 0
    return perturbed

def resample_signal(sig, original_fs=500, target_fs=250):
    """Simulates lower sampling rate."""
    # Downsample
    num_samples = int(sig.shape[1] * target_fs / original_fs)
    resampled = scipy_signal.resample(sig, num_samples, axis=1)
    # Upsample back to original length for model input
    restored = scipy_signal.resample(resampled, sig.shape[1], axis=1)
    return restored

def calculate_metrics(y_true, y_pred_prob, threshold=0.5):
    y_pred = (y_pred_prob > threshold).astype(int)
    
    metrics = {
        "f1_macro": float(f1_score(y_true, y_pred, average='macro')),
        "sensitivity_macro": float(recall_score(y_true, y_pred, average='macro')),
        "precision_macro": float(precision_score(y_true, y_pred, average='macro'))
    }
    return metrics

def run_stress_test(project_id, model_path, data_path, output_bucket):
    print(f"Starting Stress Test for Model: {model_path}")
    
    # 1. Load Resources
    try:
        model = tf.keras.models.load_model(model_path)
    except:
        print("Model not found locally, using mock model.")
        model = None

    X_val, y_val = load_data(data_path)
    
    scenarios = {
        "Baseline": lambda x: x,
        "Noise_SNR_24dB": lambda x: add_noise(x, 24),
        "Noise_SNR_12dB": lambda x: add_noise(x, 12),
        "Lead_Loss_II": lambda x: drop_leads(x, [1]), # Lead II is index 1
        "Lead_Loss_Precordial": lambda x: drop_leads(x, [6,7,8,9,10,11]), # V1-V6
        "Resampling_250Hz": lambda x: resample_signal(x, 500, 250)
    }
    
    results = {}
    
    for name, perturbation_fn in scenarios.items():
        print(f"Running Scenario: {name}...")
        
        # Apply Perturbation
        X_perturbed = perturbation_fn(X_val)
        
        # Inference
        if model:
            preds = model.predict(X_perturbed)
            if isinstance(preds, dict):
                preds = preds['pathology']
        else:
            preds = np.random.rand(len(y_val), 5).astype(np.float32)
            
        # Metrics
        metrics = calculate_metrics(y_val, preds)
        results[name] = metrics
        print(f"  F1: {metrics['f1_macro']:.4f}")

    # Calculate Drop from Baseline
    baseline_f1 = results["Baseline"]["f1_macro"]
    for name in results:
        if name == "Baseline": continue
        drop = baseline_f1 - results[name]["f1_macro"]
        results[name]["performance_drop_f1"] = float(drop)
        results[name]["is_robust"] = drop < 0.10 # Threshold for robustness

    # Save Report
    report_filename = "stress_test_report.json"
    with open(report_filename, 'w') as f:
        json.dump(results, f, indent=4)
        
    # Upload
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(output_bucket)
    bucket.blob(f"evaluation/stress_tests/{report_filename}").upload_from_filename(report_filename)
    
    os.remove(report_filename)
    print(f"Stress Test Complete. Report uploaded to gs://{output_bucket}/evaluation/stress_tests/{report_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--output_bucket', required=True)
    
    args = parser.parse_args()
    run_stress_test(args.project_id, args.model_path, args.data_path, args.output_bucket)
