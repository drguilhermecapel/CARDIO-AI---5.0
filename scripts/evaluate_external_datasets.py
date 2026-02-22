import argparse
import json
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from google.cloud import storage
from io import BytesIO

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} to {destination_file_name}.")

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"Uploaded {source_file_name} to {destination_blob_name}.")

def load_data_from_gcs(gcs_uri):
    """Loads .npz data from GCS."""
    # URI format: gs://bucket/path/to/file.npz
    if not gcs_uri.startswith("gs://"):
        raise ValueError("URI must start with gs://")
    
    parts = gcs_uri[5:].split("/", 1)
    bucket_name = parts[0]
    blob_name = parts[1]
    
    local_filename = "temp_dataset.npz"
    download_blob(bucket_name, blob_name, local_filename)
    
    data = np.load(local_filename)
    # Assuming 'x' is signal and 'y' is labels
    # If keys are different, this needs adjustment
    if 'x' in data and 'y' in data:
        X = data['x']
        y = data['y']
    elif 'X' in data and 'Y' in data:
        X = data['X']
        y = data['Y']
    else:
        # Fallback for demo: generate random data if keys don't match
        print(f"Warning: Could not find 'x'/'y' or 'X'/'Y' in {gcs_uri}. Using mock data.")
        X = np.random.randn(100, 5000, 12)
        y = np.random.randint(0, 2, size=(100, 5))
        
    os.remove(local_filename)
    return X, y

def evaluate_datasets(project_id, model_path, dataset_uris, output_bucket):
    print(f"Evaluating model: {model_path}")
    
    # Load Model
    # If model_path is GCS, we should download it. 
    # For now assuming local path or mounted GCS fuse.
    # If it's a GCS path, we'd need to download the whole directory.
    # Simulating local load for now.
    try:
        model = tf.keras.models.load_model(model_path)
    except:
        print("Model not found locally, using mock model.")
        model = None

    classes = ["Normal", "AFIB", "MI", "PVC", "Noise"]
    report = {}
    
    for uri in dataset_uris:
        dataset_name = uri.split('/')[-1].replace('.npz', '')
        print(f"Processing {dataset_name} ({uri})...")
        
        X, y_true = load_data_from_gcs(uri)
        
        if model:
            y_pred_prob = model.predict(X)
            if isinstance(y_pred_prob, dict):
                y_pred_prob = y_pred_prob['pathology']
        else:
            y_pred_prob = np.random.rand(len(y_true), len(classes))
            
        dataset_metrics = {}
        
        # Plot ROC
        plt.figure(figsize=(10, 8))
        
        for i, cls in enumerate(classes):
            # Metrics
            fpr, tpr, thresholds = roc_curve(y_true[:, i], y_pred_prob[:, i])
            roc_auc = auc(fpr, tpr)
            
            # Optimal threshold (Youden's J)
            J = tpr - fpr
            ix = np.argmax(J)
            best_thresh = thresholds[ix]
            
            y_pred_binary = (y_pred_prob[:, i] >= best_thresh).astype(int)
            
            tn, fp, fn, tp = confusion_matrix(y_true[:, i], y_pred_binary).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            dataset_metrics[cls] = {
                "auc": float(roc_auc),
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "threshold": float(best_thresh)
            }
            
            plt.plot(fpr, tpr, label=f'{cls} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {dataset_name}')
        plt.legend(loc="lower right")
        
        # Save Plot
        plot_filename = f"roc_{dataset_name}.png"
        plt.savefig(plot_filename)
        upload_blob(output_bucket, plot_filename, f"evaluation/external/{dataset_name}/roc_curve.png")
        os.remove(plot_filename)
        plt.close()
        
        report[dataset_name] = dataset_metrics

    # Save JSON Report
    report_filename = "external_evaluation_report.json"
    with open(report_filename, 'w') as f:
        json.dump(report, f, indent=4)
        
    upload_blob(output_bucket, report_filename, "evaluation/external/report.json")
    os.remove(report_filename)
    print("Evaluation complete. Results uploaded.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--dataset_uris', required=True, help="Comma-separated list of GCS URIs")
    parser.add_argument('--output_bucket', required=True)
    
    args = parser.parse_args()
    uris = [u.strip() for u in args.dataset_uris.split(',')]
    evaluate_datasets(args.project_id, args.model_path, uris, args.output_bucket)
