import kfp
from kfp import dsl
from kfp.v2.dsl import component, Input, Output, Dataset, Artifact

@component(
    base_image="python:3.9",
    packages_to_install=["google-cloud-storage", "google-cloud-bigquery", "pandas", "scipy", "numpy", "wfdb"]
)
def resample_ecg_signals(
    input_dataset: Input[Dataset],
    output_dataset: Output[Dataset],
    project_id: str,
    dataset_id: str,
    table_id: str,
    processed_bucket: str,
    target_fs: int = 500
):
    from google.cloud import storage
    from google.cloud import bigquery
    import pandas as pd
    import numpy as np
    import scipy.signal as signal
    import os
    import tempfile
    import json
    from datetime import datetime

    client = storage.Client()
    bq_client = bigquery.Client(project=project_id)
    proc_bucket = client.bucket(processed_bucket)
    
    df = pd.read_csv(input_dataset.path)
    processed_records = []
    transform_logs = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for index, row in df.iterrows():
            try:
                # Determine source URI (could be raw or already processed)
                gcs_uri = row.get('processed_uri') or row.get('gcs_uri')
                record_name = row['record_name']
                
                if not gcs_uri:
                    continue
                
                # Download
                bucket_name = gcs_uri.split('/')[2]
                blob_path = '/'.join(gcs_uri.split('/')[3:])
                
                src_bucket = client.bucket(bucket_name)
                blob = src_bucket.blob(blob_path)
                
                local_path = os.path.join(temp_dir, os.path.basename(blob_path))
                blob.download_to_filename(local_path)
                
                # Load Signal
                if local_path.endswith('.npy'):
                    sig = np.load(local_path)
                    # Assume FS is in row metadata or default to 500 if processed? 
                    # If coming from raw WFDB, we need to read header.
                    # For this component, let's assume 'sampling_rate' is in the input DF 
                    # (populated by ingestion/validation pipeline)
                    orig_fs = int(row.get('sampling_rate', 500)) 
                else:
                    # Fallback or error
                    print(f"Unsupported format for resampling: {local_path}")
                    continue
                
                # Resample if needed
                if orig_fs != target_fs:
                    orig_len = len(sig)
                    # Calculate new length
                    num_samples = int(orig_len * target_fs / orig_fs)
                    
                    # Deterministic Resampling (Polyphase)
                    # Up/Down sampling
                    # gcd = np.gcd(orig_fs, target_fs)
                    # up = target_fs // gcd
                    # down = orig_fs // gcd
                    # resampled_sig = signal.resample_poly(sig, up, down, axis=0)
                    
                    # Or simple Fourier method (scipy.signal.resample) - often faster for non-integer ratios
                    resampled_sig = signal.resample(sig, num_samples, axis=0)
                    
                    new_len = len(resampled_sig)
                    scale_factor = target_fs / orig_fs
                    
                    # Save
                    out_filename = f"resampled/{record_name}_{target_fs}hz.npy"
                    local_out = os.path.join(temp_dir, f"{record_name}_resampled.npy")
                    np.save(local_out, resampled_sig)
                    
                    out_blob = proc_bucket.blob(out_filename)
                    out_blob.upload_from_filename(local_out)
                    
                    # Update Row
                    row['processed_uri'] = f"gs://{processed_bucket}/{out_filename}"
                    row['sampling_rate'] = target_fs
                    row['length_samples'] = new_len
                    
                    # Log Transformation
                    transform_logs.append({
                        "record_id": record_name,
                        "transform_type": "RESAMPLE",
                        "original_fs": orig_fs,
                        "target_fs": target_fs,
                        "original_length": orig_len,
                        "new_length": new_len,
                        "scale_factor": scale_factor,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                else:
                    # No resampling needed, but maybe pass through
                    pass
                
                processed_records.append(row)
                
            except Exception as e:
                print(f"Error processing {row.get('record_name')}: {e}")

    # Output Dataset
    out_df = pd.DataFrame(processed_records)
    out_df.to_csv(output_dataset.path, index=False)
    
    # Write Logs to BigQuery
    if transform_logs:
        table_ref = f"{project_id}.{dataset_id}.{table_id}"
        errors = bq_client.insert_rows_json(table_ref, transform_logs)
        if errors:
            print(f"BQ Errors: {errors}")
        else:
            print(f"Logged {len(transform_logs)} transformations.")

@dsl.pipeline(
    name="resampling-pipeline",
    description="Standardize ECG sampling rates."
)
def resampling_pipeline(
    input_dataset_uri: str,
    processed_bucket: str,
    project_id: str,
    dataset_id: str,
    table_id: str
):
    pass
