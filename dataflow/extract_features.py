import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions
from apache_beam.io import fileio
import argparse
import logging
import numpy as np
import scipy.signal as signal
import scipy.stats as stats

class ExtractFeaturesDoFn(beam.DoFn):
    def process(self, element):
        """
        Element is a dictionary from BigQuery or GCS file metadata.
        For this pipeline, let's assume we read from the 'validated_ecgs' BQ table 
        or directly from GCS files if we want to re-process.
        
        Let's assume input is a dictionary: {'record_name': ..., 'gcs_uri': ...}
        """
        from google.cloud import storage
        import os
        import tempfile
        
        record_name = element.get('record_name')
        gcs_uri = element.get('gcs_uri') or element.get('processed_uri')
        
        if not gcs_uri:
            return

        try:
            # Download Signal
            client = storage.Client()
            bucket_name = gcs_uri.split('/')[2]
            blob_path = '/'.join(gcs_uri.split('/')[3:])
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            
            with tempfile.NamedTemporaryFile(suffix='.npy') as tmp:
                blob.download_to_filename(tmp.name)
                # Load Signal (samples, leads)
                # If it's a .dat (WFDB), we'd need wfdb.rdrecord. 
                # Assuming .npy from previous steps for speed/standardization.
                if gcs_uri.endswith('.npy'):
                    ecg_signal = np.load(tmp.name)
                else:
                    # Skip non-npy for this specific feature extractor to keep it simple
                    return

            # Assume 500Hz (standardized)
            fs = 500
            
            # --- Feature Extraction Logic ---
            
            # 1. Representative Beat (Average Beat)
            # We'll use Lead II (index 1 usually) for rhythm features
            lead_idx = 1 if ecg_signal.shape[1] > 1 else 0
            lead_ii = ecg_signal[:, lead_idx]
            
            # Detect R-peaks (Simple heuristic)
            # Bandpass 5-15Hz for QRS energy
            sos = signal.butter(3, [5, 15], 'bandpass', fs=fs, output='sos')
            filtered = signal.sosfiltfilt(sos, lead_ii)
            # Square and integrate
            energy = filtered ** 2
            # Find peaks
            peaks, _ = signal.find_peaks(energy, height=np.mean(energy)*3, distance=int(0.2*fs))
            
            if len(peaks) < 2:
                return # Cannot extract intervals
            
            # RR Intervals
            rr_intervals = np.diff(peaks) / fs * 1000 # ms
            mean_rr = np.mean(rr_intervals)
            std_rr = np.std(rr_intervals)
            heart_rate = 60000 / mean_rr
            
            # QRS Duration (Approximate based on peak width at half height of energy envelope)
            qrs_durations = []
            for p in peaks:
                # Search bounds
                start = max(0, p - int(0.1*fs))
                end = min(len(energy), p + int(0.1*fs))
                window = energy[start:end]
                half_height = energy[p] / 2
                # Find crossings
                crossings = np.where(window > half_height)[0]
                if len(crossings) > 0:
                    width = crossings[-1] - crossings[0]
                    qrs_durations.append(width / fs * 1000) # ms
            
            mean_qrs = np.mean(qrs_durations) if qrs_durations else 0
            
            # QT Interval (Heuristic: R-peak to T-end)
            # This is very hard to do robustly without a delineator. 
            # We will use Bazett's formula on a dummy QT if we can't find T.
            # For this demo, let's use a statistical proxy or skip if too complex.
            # Proxy: QT is roughly 40% of RR usually. Let's try to find T-wave max.
            qt_intervals = []
            for p in peaks:
                # Search 200ms to 600ms after R
                start_t = p + int(0.2*fs)
                end_t = p + int(0.6*fs)
                if end_t < len(lead_ii):
                    t_window = lead_ii[start_t:end_t]
                    if len(t_window) > 0:
                        t_peak_idx = np.argmax(np.abs(t_window))
                        # T-end is roughly where it returns to baseline. 
                        # Let's just take R-to-Tpeak + 50ms as a rough proxy for QT
                        qt = (0.2*1000) + (t_peak_idx / fs * 1000) + 50 
                        qt_intervals.append(qt)
            
            mean_qt = np.mean(qt_intervals) if qt_intervals else 400
            qtc = mean_qt / np.sqrt(mean_rr / 1000) # Bazett
            
            # Amplitudes (Per Lead)
            amplitudes = {}
            for i in range(ecg_signal.shape[1]):
                lead_data = ecg_signal[:, i]
                amplitudes[f"lead_{i}_min"] = np.min(lead_data)
                amplitudes[f"lead_{i}_max"] = np.max(lead_data)
                amplitudes[f"lead_{i}_std"] = np.std(lead_data)

            # --- Spectral Features (Welch) ---
            freqs, psd = signal.welch(lead_ii, fs=fs, nperseg=1024)
            
            # Band Power
            # LF (0.04-0.15 Hz) - HRV band, but on short ECG implies baseline/respiration
            # HF (0.15-0.4 Hz) - HRV band
            # QRS Freq (10-25 Hz)
            
            def band_power(low, high):
                idx = np.logical_and(freqs >= low, freqs <= high)
                return np.trapz(psd[idx], freqs[idx])
            
            p_lf = band_power(0.04, 0.15)
            p_hf = band_power(0.15, 0.4)
            p_qrs = band_power(10, 25)
            
            # Dominant Freq
            dom_freq = freqs[np.argmax(psd)]
            
            # Construct Row
            row = {
                'record_id': record_name,
                'timestamp': datetime.utcnow().isoformat(),
                # Classical
                'heart_rate': float(heart_rate),
                'mean_rr_ms': float(mean_rr),
                'std_rr_ms': float(std_rr),
                'mean_qrs_ms': float(mean_qrs),
                'qtc_ms': float(qtc),
                # Spectral
                'power_lf': float(p_lf),
                'power_hf': float(p_hf),
                'power_qrs': float(p_qrs),
                'dominant_freq_hz': float(dom_freq),
                # Amplitudes (JSON string for flexibility or separate cols)
                'lead_amplitudes_json': json.dumps(amplitudes)
            }
            
            yield row

        except Exception as e:
            logging.error(f"Failed to extract features for {record_name}: {e}")

from datetime import datetime
import json

def run(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_table', required=True, help='BigQuery table with validated ECGs')
    parser.add_argument('--output_table', required=True, help='BigQuery table for features')
    parser.add_argument('--project', required=True)
    parser.add_argument('--region', default='us-central1')
    
    known_args, pipeline_args = parser.parse_known_args(argv)
    
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True
    pipeline_options.view_as(SetupOptions).requirements_file = 'requirements.txt'

    with beam.Pipeline(options=pipeline_options) as p:
        (
            p
            | "ReadValidatedECGs" >> beam.io.ReadFromBigQuery(
                query=f"SELECT record_name, gcs_uri FROM `{known_args.input_table}` WHERE status = 'VALIDATED'",
                use_standard_sql=True,
                project=known_args.project
            )
            | "ExtractFeatures" >> beam.ParDo(ExtractFeaturesDoFn())
            | "WriteFeatures" >> beam.io.WriteToBigQuery(
                known_args.output_table,
                schema='record_id:STRING, timestamp:TIMESTAMP, heart_rate:FLOAT, mean_rr_ms:FLOAT, std_rr_ms:FLOAT, mean_qrs_ms:FLOAT, qtc_ms:FLOAT, power_lf:FLOAT, power_hf:FLOAT, power_qrs:FLOAT, dominant_freq_hz:FLOAT, lead_amplitudes_json:STRING',
                create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND
            )
        )

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()
