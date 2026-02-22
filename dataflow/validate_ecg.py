import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions
from apache_beam.io import fileio
import argparse
import logging
import os

# Define validation criteria
MIN_SAMPLING_RATE = 500
REQUIRED_LEADS = 12
REQUIRED_DURATION_SEC = 10

class ValidateEcgDoFn(beam.DoFn):
    def __init__(self, quarantine_bucket):
        self.quarantine_bucket = quarantine_bucket

    def process(self, file_metadata):
        from google.cloud import storage
        import wfdb
        import os
        import tempfile
        
        gcs_uri = file_metadata.path
        if not gcs_uri.endswith('.hea'):
            return

        bucket_name = gcs_uri.split('/')[2]
        blob_name = '/'.join(gcs_uri.split('/')[3:])
        record_name = blob_name.replace('.hea', '')
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Create temp dir for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            local_header_path = os.path.join(temp_dir, os.path.basename(blob_name))
            local_record_base = os.path.splitext(local_header_path)[0]
            
            # Download header
            blob = bucket.blob(blob_name)
            blob.download_to_filename(local_header_path)
            
            # Download signal (.dat) if exists
            dat_blob_name = blob_name.replace('.hea', '.dat')
            dat_blob = bucket.blob(dat_blob_name)
            local_dat_path = os.path.join(temp_dir, os.path.basename(dat_blob_name))
            
            has_signal = False
            if dat_blob.exists():
                dat_blob.download_to_filename(local_dat_path)
                has_signal = True
            
            valid = True
            reasons = []

            try:
                # Read header only first to check metadata
                # If signal is needed for length check, we need to read it.
                # wfdb.rdheader is lighter
                header = wfdb.rdheader(local_record_base)
                
                # 1. Check Sampling Rate
                if header.fs < MIN_SAMPLING_RATE:
                    valid = False
                    reasons.append(f"Low sampling rate: {header.fs}Hz < {MIN_SAMPLING_RATE}Hz")
                
                # 2. Check Leads
                if header.n_sig != REQUIRED_LEADS:
                    valid = False
                    reasons.append(f"Invalid lead count: {header.n_sig} != {REQUIRED_LEADS}")
                
                # 3. Check Duration (Signal Length / FS)
                duration = header.sig_len / header.fs
                if duration < REQUIRED_DURATION_SEC:
                    valid = False
                    reasons.append(f"Short duration: {duration}s < {REQUIRED_DURATION_SEC}s")
                
                if not has_signal:
                    valid = False
                    reasons.append("Missing .dat signal file")

            except Exception as e:
                valid = False
                reasons.append(f"Read error: {str(e)}")

            if valid:
                yield beam.pvalue.TaggedOutput('valid', {
                    'record': record_name,
                    'uri': gcs_uri,
                    'fs': header.fs,
                    'duration': duration
                })
            else:
                # Move to Quarantine
                quarantine_prefix = f"quarantine/{os.path.dirname(blob_name)}/"
                quarantine_blob_name = f"{quarantine_prefix}{os.path.basename(blob_name)}"
                
                # Copy and Delete (Move) Header
                bucket.copy_blob(blob, self.quarantine_bucket, quarantine_blob_name)
                blob.delete()
                
                # Move Signal if exists
                if has_signal:
                    quarantine_dat_name = f"{quarantine_prefix}{os.path.basename(dat_blob_name)}"
                    bucket.copy_blob(dat_blob, self.quarantine_bucket, quarantine_dat_name)
                    dat_blob.delete()
                
                yield beam.pvalue.TaggedOutput('invalid', {
                    'record': record_name,
                    'reasons': reasons,
                    'original_uri': gcs_uri,
                    'quarantine_uri': f"gs://{self.quarantine_bucket.name}/{quarantine_blob_name}"
                })

def run(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_bucket', required=True, help='Input GCS Bucket to scan')
    parser.add_argument('--quarantine_bucket', required=True, help='Bucket to move invalid files to')
    parser.add_argument('--output_table', required=True, help='BigQuery table for valid records')
    
    known_args, pipeline_args = parser.parse_known_args(argv)
    
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True
    # Ensure workers have wfdb installed
    pipeline_options.view_as(SetupOptions).requirements_file = 'requirements.txt'

    with beam.Pipeline(options=pipeline_options) as p:
        # Read all .hea files
        files = p | "MatchFiles" >> fileio.MatchFiles(f"gs://{known_args.input_bucket}/**/*.hea")
        
        # Validate and Split
        results = (
            files 
            | "ReadMatches" >> fileio.ReadMatches()
            | "ValidateECG" >> beam.ParDo(ValidateEcgDoFn(known_args.quarantine_bucket)).with_outputs('valid', 'invalid')
        )
        
        # Log Invalid
        (
            results.invalid 
            | "FormatInvalidLog" >> beam.Map(lambda x: f"QUARANTINED: {x['record']} - {x['reasons']}")
            | "LogInvalid" >> beam.Map(logging.warning)
        )
        
        # Write Valid to BigQuery
        (
            results.valid
            | "FormatValidRow" >> beam.Map(lambda x: {
                'record_name': x['record'],
                'gcs_uri': x['uri'],
                'sampling_rate': x['fs'],
                'duration_sec': x['duration'],
                'status': 'VALIDATED'
            })
            | "WriteToBQ" >> beam.io.WriteToBigQuery(
                known_args.output_table,
                schema='record_name:STRING, gcs_uri:STRING, sampling_rate:INTEGER, duration_sec:FLOAT, status:STRING',
                create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND
            )
        )

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()
