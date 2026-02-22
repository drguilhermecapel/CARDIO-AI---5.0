import kfp
from kfp import dsl
from kfp.v2.dsl import component, Input, Output, Dataset, Metrics, Artifact

@component(
    base_image="python:3.9",
    packages_to_install=["google-cloud-bigquery", "pandas", "scipy", "numpy", "wfdb"]
)
def detect_artifacts_batch(
    input_dataset: Input[Dataset],
    output_report: Output[Dataset],
    project_id: str,
    dataset_id: str,
    table_id: str
):
    import pandas as pd
    import numpy as np
    import wfdb
    import os
    import tempfile
    from google.cloud import bigquery
    from google.cloud import storage
    
    # Inline the LightweightArtifactDetector class since we can't easily import local files in KFP component
    # without building a custom container.
    import scipy.stats as stats
    
    class LightweightArtifactDetector:
        def __init__(self, sampling_rate=500):
            self.fs = sampling_rate
            self.window_size = int(0.6 * sampling_rate)
            self.template_buffer = []
            self.max_template_size = 50

        def detect_beats(self, signal):
            diff = np.diff(signal)
            squared = diff ** 2
            window = np.ones(int(0.15 * self.fs)) / int(0.15 * self.fs)
            integrated = np.convolve(squared, window, mode='same')
            threshold = np.mean(integrated) * 3
            peaks = []
            last_peak = 0
            refractory_period = int(0.2 * self.fs)
            for i, val in enumerate(integrated):
                if val > threshold and (i - last_peak) > refractory_period:
                    peaks.append(i)
                    last_peak = i
            return np.array(peaks)

        def analyze_beat_quality(self, beat_signal):
            if len(beat_signal) < self.window_size // 2:
                return False, 0.0
            kurtosis = stats.kurtosis(beat_signal)
            correlation = 0.0
            if self.template_buffer:
                avg_template = np.mean(self.template_buffer, axis=0)
                min_len = min(len(beat_signal), len(avg_template))
                correlation = np.corrcoef(beat_signal[:min_len], avg_template[:min_len])[0, 1]
                if correlation < 0.8:
                    return False, correlation
            return (kurtosis > 3.0), correlation

        def process_record(self, signal_array):
            qrs_indices = self.detect_beats(signal_array)
            results = []
            half_window = self.window_size // 2
            for r_peak in qrs_indices:
                start = r_peak - half_window
                end = r_peak + half_window
                if start < 0 or end >= len(signal_array): continue
                beat = signal_array[start:end]
                if len(self.template_buffer) < 5: self.template_buffer.append(beat)
                is_valid, corr = self.analyze_beat_quality(beat)
                if is_valid:
                    if len(self.template_buffer) >= self.max_template_size: self.template_buffer.pop(0)
                    self.template_buffer.append(beat)
                results.append({"is_valid": is_valid, "correlation": corr})
            return results

    # Processing Logic
    storage_client = storage.Client()
    bq_client = bigquery.Client(project=project_id)
    
    df = pd.read_csv(input_dataset.path)
    report_rows = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for _, row in df.iterrows():
            try:
                gcs_uri = row.get('processed_uri') or row.get('gcs_uri')
                if not gcs_uri: continue
                
                bucket_name = gcs_uri.split('/')[2]
                blob_path = '/'.join(gcs_uri.split('/')[3:])
                
                # Download
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_path)
                local_path = os.path.join(temp_dir, os.path.basename(blob_path))
                blob.download_to_filename(local_path)
                
                # Load Signal
                if local_path.endswith('.npy'):
                    signals = np.load(local_path)
                    # Assuming shape (samples, leads)
                    lead_ii = signals[:, 1] if signals.shape[1] > 1 else signals[:, 0]
                else:
                    # Fallback for WFDB
                    # (Simplified for this example)
                    continue

                # Detect
                detector = LightweightArtifactDetector(sampling_rate=500)
                beat_results = detector.process_record(lead_ii)
                
                total = len(beat_results)
                valid = sum(1 for b in beat_results if b['is_valid'])
                artifacts = total - valid
                avg_corr = np.mean([b['correlation'] for b in beat_results]) if beat_results else 0
                
                report_rows.append({
                    "record_id": row.get('record_name', 'unknown'),
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "total_beats": total,
                    "valid_beats": valid,
                    "artifact_beats": artifacts,
                    "quality_score": (valid / total) if total > 0 else 0.0,
                    "average_sqi": avg_corr
                })
                
            except Exception as e:
                print(f"Error processing {row.get('record_name')}: {e}")

    # Upload to BigQuery
    if report_rows:
        errors = bq_client.insert_rows_json(f"{project_id}.{dataset_id}.{table_id}", report_rows)
        if errors:
            print(f"BQ Errors: {errors}")
        else:
            print(f"Inserted {len(report_rows)} quality reports.")
            
    # Output CSV
    out_df = pd.DataFrame(report_rows)
    out_df.to_csv(output_report.path, index=False)

@dsl.pipeline(
    name="beat-quality-pipeline",
    description="Detect artifacts and rate ECG quality per beat."
)
def quality_pipeline(
    input_dataset_uri: str,
    project_id: str,
    dataset_id: str,
    table_id: str
):
    # This is a standalone pipeline example, usually integrated into the main one
    # We mock the input dataset for the component signature
    pass 
