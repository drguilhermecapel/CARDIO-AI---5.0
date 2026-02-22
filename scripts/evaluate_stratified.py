import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score, brier_score_loss, precision_recall_curve
from google.cloud import storage
import tensorflow as tf

def evaluate_model(model_path, data_path, output_bucket, run_id):
    """
    Evaluates a trained model on a test set, stratified by patient/center.
    """
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # Load Test Data (Mock for now, replace with BigQuery/GCS read)
    # Assume data_path is a CSV with 'gcs_uri', 'label', 'patient_id', 'center_id'
    print(f"Loading test data from {data_path}...")
    # df = pd.read_csv(data_path)
    # For demo, generate mock dataframe
    num_samples = 200
    df = pd.DataFrame({
        'patient_id': np.random.randint(0, 50, num_samples),
        'center_id': np.random.choice(['CenterA', 'CenterB', 'CenterC'], num_samples),
        'label': np.random.randint(0, 5, num_samples) # 5 classes
    })
    # Mock predictions
    y_true = tf.keras.utils.to_categorical(df['label'], num_classes=5)
    y_pred = np.random.rand(num_samples, 5)
    y_pred = y_pred / y_pred.sum(axis=1, keepdims=True) # Normalize
    
    # --- Global Metrics ---
    print("Calculating Global Metrics...")
    global_metrics = {}
    
    # Macro F1
    y_pred_classes = np.argmax(y_pred, axis=1)
    global_metrics['f1_macro'] = f1_score(df['label'], y_pred_classes, average='macro')
    
    # Brier Score (Multi-class Brier is mean squared error of probabilities)
    # We calculate per-class Brier and average
    brier_scores = []
    for i in range(5):
        bs = brier_score_loss(y_true[:, i], y_pred[:, i])
        brier_scores.append(bs)
    global_metrics['brier_score'] = np.mean(brier_scores)
    
    # AUC per Class
    aucs = {}
    plt.figure(figsize=(10, 8))
    for i in range(5):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        aucs[f'class_{i}'] = roc_auc
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
        
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves by Class')
    plt.legend(loc="lower right")
    
    # Save ROC Plot
    roc_filename = f"evaluation/{run_id}/roc_curves.png"
    plt.savefig('/tmp/roc_curves.png')
    plt.close()
    
    global_metrics['auc_per_class'] = aucs
    
    # --- Stratified Evaluation ---
    print("Calculating Stratified Metrics...")
    stratified_metrics = {'by_center': {}, 'by_patient': {}}
    
    # By Center
    for center in df['center_id'].unique():
        mask = df['center_id'] == center
        if mask.sum() > 0:
            y_true_c = df.loc[mask, 'label']
            y_pred_c = y_pred_classes[mask]
            f1 = f1_score(y_true_c, y_pred_c, average='macro')
            stratified_metrics['by_center'][center] = {'f1_macro': f1, 'count': int(mask.sum())}
            
    # By Patient (Aggregate predictions per patient?)
    # Or just average metric across patients
    patient_f1s = []
    for patient in df['patient_id'].unique():
        mask = df['patient_id'] == patient
        if mask.sum() > 0:
            y_true_p = df.loc[mask, 'label']
            y_pred_p = y_pred_classes[mask]
            # F1 might be undefined if only 1 sample, handle gracefully
            try:
                f1 = f1_score(y_true_p, y_pred_p, average='macro', zero_division=0)
                patient_f1s.append(f1)
            except:
                pass
    stratified_metrics['by_patient']['avg_f1_macro'] = np.mean(patient_f1s) if patient_f1s else 0.0

    # --- Upload to GCS ---
    client = storage.Client()
    bucket = client.bucket(output_bucket)
    
    # Upload Metrics JSON
    metrics_blob = bucket.blob(f"evaluation/{run_id}/metrics.json")
    full_report = {
        'run_id': run_id,
        'global': global_metrics,
        'stratified': stratified_metrics
    }
    metrics_blob.upload_from_string(json.dumps(full_report, indent=2))
    
    # Upload ROC Plot
    roc_blob = bucket.blob(roc_filename)
    roc_blob.upload_from_filename('/tmp/roc_curves.png')
    
    print(f"Evaluation complete. Report saved to gs://{output_bucket}/evaluation/{run_id}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--output_bucket', required=True)
    parser.add_argument('--run_id', required=True)
    
    args = parser.parse_args()
    evaluate_model(args.model_path, args.data_path, args.output_bucket, args.run_id)
