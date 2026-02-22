import kfp
from kfp import dsl
from kfp.v2.dsl import component, Input, Output, Dataset, Metrics, Artifact

@component(
    base_image="python:3.9",
    packages_to_install=["google-cloud-storage", "pandas", "scipy", "numpy", "wfdb"]
)
def adaptive_ecg_filter(
    input_dataset: Input[Dataset],
    output_dataset: Output[Dataset],
    metrics: Output[Metrics],
    processed_bucket: str,
    sampling_rate: int = 500
):
    from google.cloud import storage
    import pandas as pd
    import numpy as np
    import scipy.signal as signal
    import wfdb
    import os
    import tempfile

    def apply_filters(sig, fs):
        # 1. Baseline Wander Removal (High-pass Butterworth, 0.5Hz, Zero-phase)
        b, a = signal.butter(4, 0.5 / (fs / 2), 'highpass')
        sig_base = signal.filtfilt(b, a, sig, axis=0)
        
        # 2. Notch Filter (50Hz and 60Hz)
        for freq in [50, 60]:
            b, a = signal.iirnotch(freq, 30, fs)
            sig_base = signal.filtfilt(b, a, sig_base, axis=0)
            
        # 3. Lowpass Filter (40Hz or 100Hz depending on need, let's use 100Hz for diagnostic)
        b, a = signal.butter(4, 100 / (fs / 2), 'lowpass')
        sig_clean = signal.filtfilt(b, a, sig_base, axis=0)
        
        return sig_clean

    def calculate_snr(original, cleaned):
        # Estimate noise as the difference
        noise = original - cleaned
        p_signal = np.sum(cleaned ** 2)
        p_noise = np.sum(noise ** 2)
        if p_noise == 0: return 100 # Perfect signal
        return 10 * np.log10(p_signal / p_noise)

    def calculate_rmse(original, cleaned):
        return np.sqrt(np.mean((original - cleaned) ** 2))

    # Setup GCS
    client = storage.Client()
    proc_bucket = client.bucket(processed_bucket)
    
    df = pd.read_csv(input_dataset.path)
    processed_records = []
    
    total_snr = 0
    total_rmse = 0
    count = 0
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for index, row in df.iterrows():
            try:
                gcs_uri = row['gcs_uri']
                record_name = row['record_name']
                
                # Parse GCS URI
                bucket_name = gcs_uri.split('/')[2]
                blob_path = '/'.join(gcs_uri.split('/')[3:])
                
                # Download
                src_bucket = client.bucket(bucket_name)
                blob_hea = src_bucket.blob(blob_path + '.hea')
                blob_dat = src_bucket.blob(blob_path + '.dat')
                
                local_base = os.path.join(temp_dir, record_name)
                blob_hea.download_to_filename(local_base + '.hea')
                if blob_dat.exists():
                    blob_dat.download_to_filename(local_base + '.dat')
                    
                    # Read
                    record = wfdb.rdrecord(local_base)
                    raw_signals = record.p_signal
                    fs = record.fs
                    
                    # Process
                    clean_signals = apply_filters(raw_signals, fs)
                    
                    # Metrics
                    snr = calculate_snr(raw_signals, clean_signals)
                    rmse = calculate_rmse(raw_signals, clean_signals)
                    
                    total_snr += snr
                    total_rmse += rmse
                    count += 1
                    
                    # Save Processed
                    # We can't easily write back to WFDB format without some work, 
                    # but we can save as .npy or CSV for ML. Let's save as .npy for efficiency.
                    out_filename = f"processed/{record_name}.npy"
                    local_out = os.path.join(temp_dir, f"{record_name}.npy")
                    np.save(local_out, clean_signals)
                    
                    # Upload
                    out_blob = proc_bucket.blob(out_filename)
                    out_blob.upload_from_filename(local_out)
                    
                    row['processed_uri'] = f"gs://{processed_bucket}/{out_filename}"
                    row['snr'] = snr
                    row['rmse'] = rmse
                    processed_records.append(row)
                    
            except Exception as e:
                print(f"Error processing {row.get('record_name')}: {e}")
    
    # Output Dataset
    out_df = pd.DataFrame(processed_records)
    out_df.to_csv(output_dataset.path, index=False)
    
    # Log Aggregate Metrics
    if count > 0:
        avg_snr = total_snr / count
        avg_rmse = total_rmse / count
        metrics.log_metric("avg_snr_db", avg_snr)
        metrics.log_metric("avg_rmse_loss", avg_rmse)
        metrics.log_metric("processed_count", count)

