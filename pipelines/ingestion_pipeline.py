import kfp
from kfp import dsl
from kfp.v2.dsl import component, Input, Output, Dataset, Artifact, HTML

@component(
    base_image="python:3.9",
    packages_to_install=["google-cloud-storage", "pandas", "wfdb", "numpy"]
)
def process_wfdb_files(
    gcs_source_bucket: str,
    gcs_source_prefix: str,
    output_dataset: Output[Dataset]
):
    from google.cloud import storage
    import pandas as pd
    import wfdb
    import os
    import tempfile
    import ast
    
    # Setup GCS client
    client = storage.Client()
    bucket = client.bucket(gcs_source_bucket)
    blobs = list(bucket.list_blobs(prefix=gcs_source_prefix))
    
    processed_data = []
    
    # Create temp dir for downloading
    temp_dir = tempfile.mkdtemp()
    
    # Group files by record name (assuming WFDB structure: record.dat, record.hea)
    records = set()
    for blob in blobs:
        if blob.name.endswith('.hea'):
            records.add(blob.name.replace('.hea', ''))
            
    print(f"Found {len(records)} records.")
    
    for record_path in records:
        try:
            # Download .hea and .dat
            record_name = os.path.basename(record_path)
            local_path_base = os.path.join(temp_dir, record_name)
            
            # Download header
            blob_hea = bucket.blob(record_path + '.hea')
            blob_hea.download_to_filename(local_path_base + '.hea')
            
            # Download signal (try .dat)
            blob_dat = bucket.blob(record_path + '.dat')
            if blob_dat.exists():
                blob_dat.download_to_filename(local_path_base + '.dat')
                
                # Read WFDB
                # wfdb.rdrecord reads the header and data
                record = wfdb.rdrecord(local_path_base)
                
                # Extract Metadata
                meta = {
                    'record_name': record_name,
                    'fs': record.fs,
                    'n_sig': record.n_sig,
                    'sig_len': record.sig_len,
                    'comments': str(record.comments), # Serialize list to string
                    'gcs_uri': f"gs://{gcs_source_bucket}/{record_path}"
                }
                
                # Normalize Labels (Simple heuristic based on comments/diagnosis)
                # Assuming DX in comments like "Dx: 1234, 5678" or SNOMED codes
                diagnosis = []
                for comment in record.comments:
                    if 'Dx:' in comment:
                        dx_codes = comment.split('Dx:')[1].strip().split(',')
                        diagnosis.extend([d.strip() for d in dx_codes])
                
                meta['diagnosis_codes'] = str(diagnosis) # Serialize list
                processed_data.append(meta)
                
        except Exception as e:
            print(f"Error processing {record_path}: {e}")
            
    # Convert to DataFrame
    df = pd.DataFrame(processed_data)
    df.to_csv(output_dataset.path, index=False)


@component(
    base_image="python:3.9",
    packages_to_install=["google-cloud-bigquery", "pandas", "pandas-gbq"]
)
def upload_to_bigquery(
    input_dataset: Input[Dataset],
    project_id: str,
    dataset_id: str,
    table_id: str
):
    import pandas as pd
    from google.cloud import bigquery
    
    df = pd.read_csv(input_dataset.path)
    
    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND",
        schema_update_options=["ALLOW_FIELD_ADDITION"],
        autodetect=True
    )
    
    job = client.load_table_from_dataframe(
        df, table_ref, job_config=job_config
    )
    job.result()
    print(f"Loaded {len(df)} rows into {table_ref}")


@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "jinja2"]
)
def generate_catalog_report(
    input_dataset: Input[Dataset],
    catalog_html: Output[HTML]
):
    import pandas as pd
    import ast
    
    df = pd.read_csv(input_dataset.path)
    
    # Build Hierarchy: Root -> Pathology -> Record
    catalog = {}
    
    for _, row in df.iterrows():
        # Handle stringified list
        try:
            dx_list = ast.literal_eval(row['diagnosis_codes']) if isinstance(row['diagnosis_codes'], str) else []
        except:
            dx_list = ["Unknown"]
            
        if not dx_list:
            dx_list = ["Normal/Unknown"]
            
        for dx in dx_list:
            if dx not in catalog:
                catalog[dx] = []
            catalog[dx].append({
                "record": row['record_name'],
                "uri": row['gcs_uri'],
                "meta": row['comments']
            })
            
    # Generate HTML
    html_content = """
    <html>
    <head>
        <title>ECG Pathology Catalog</title>
        <style>
            body { font-family: 'Roboto', sans-serif; padding: 20px; background-color: #f8f9fa; }
            h1 { color: #333; }
            .pathology { margin-bottom: 20px; border: 1px solid #ddd; padding: 15px; border-radius: 8px; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
            .pathology h3 { margin-top: 0; color: #0288d1; border-bottom: 1px solid #eee; padding-bottom: 10px; }
            .record { font-size: 0.9em; margin-left: 20px; color: #555; }
            a { color: #0288d1; text-decoration: none; }
            a:hover { text-decoration: underline; }
            .badge { background: #e1f5fe; color: #0277bd; padding: 2px 6px; border-radius: 4px; font-size: 0.8em; }
        </style>
    </head>
    <body>
        <h1>ECG Dataset Hierarchical Catalog</h1>
        <p>Generated by Vertex AI Ingestion Pipeline</p>
    """
    
    # Sort by number of samples
    sorted_catalog = dict(sorted(catalog.items(), key=lambda item: len(item[1]), reverse=True))
    
    for pathology, records in sorted_catalog.items():
        html_content += f"""
        <div class="pathology">
            <h3>{pathology} <span class="badge">{len(records)} samples</span></h3>
            <ul>
        """
        for rec in records[:10]: # Limit to 10 preview
            html_content += f"<li class='record'><strong>{rec['record']}</strong> - <a href='{rec['uri']}'>View Source</a> <br><small>{rec['meta']}</small></li>"
        
        if len(records) > 10:
            html_content += f"<li class='record' style='color: #888;'>... and {len(records)-10} more records</li>"
            
        html_content += """
            </ul>
        </div>
        """
        
    html_content += "</body></html>"
    
    with open(catalog_html.path, 'w') as f:
        f.write(html_content)


@dsl.pipeline(
    name="wfdb-ingestion-pipeline",
    description="Ingest WFDB files from GCS, normalize, and catalog."
)
def ingestion_pipeline(
    gcs_bucket: str,
    gcs_prefix: str,
    project_id: str,
    bq_dataset: str,
    bq_table: str
):
    process_task = process_wfdb_files(
        gcs_source_bucket=gcs_bucket,
        gcs_source_prefix=gcs_prefix
    )
    
    upload_task = upload_to_bigquery(
        input_dataset=process_task.outputs["output_dataset"],
        project_id=project_id,
        dataset_id=bq_dataset,
        table_id=bq_table
    )
    
    catalog_task = generate_catalog_report(
        input_dataset=process_task.outputs["output_dataset"]
    )

if __name__ == "__main__":
    kfp.v2.compiler.Compiler().compile(
        pipeline_func=ingestion_pipeline,
        package_path="ingestion_pipeline.json"
    )
