import argparse
import json
import pandas as pd
from google.cloud import bigquery
from google.cloud import storage

def export_for_labeling(project_id, source_table, output_bucket, sample_size=1000):
    """
    Exports samples for labeling, prioritizing rare pathologies based on model predictions.
    """
    client = bigquery.Client(project=project_id)
    
    # Define Rare Classes and Sampling Ratios
    # We assume the source table has 'predicted_label' and 'confidence' columns
    # We want low confidence samples generally, but also specific rare classes.
    
    # Strategy:
    # 1. Fetch counts per class.
    # 2. Calculate quotas.
    
    print("Fetching distribution...")
    query_dist = f"""
    SELECT predicted_label, COUNT(*) as count
    FROM `{source_table}`
    GROUP BY predicted_label
    """
    df_dist = client.query(query_dist).to_dataframe()
    print(df_dist)
    
    # Define sampling targets
    # Rare: MI, PVC (Take more of these)
    # Common: Normal, Noise (Take fewer)
    
    samples = []
    
    for _, row in df_dist.iterrows():
        label = row['predicted_label']
        count = row['count']
        
        if label in ['MI', 'PVC', 'AFIB']:
            # Rare/Critical: Take up to 200 each
            limit = 200
        else:
            # Common: Take up to 50 each
            limit = 50
            
        print(f"Sampling {limit} from {label}...")
        
        query_sample = f"""
        SELECT gcs_uri, predicted_label, confidence
        FROM `{source_table}`
        WHERE predicted_label = '{label}'
        ORDER BY confidence ASC -- Uncertainty Sampling (Active Learning)
        LIMIT {limit}
        """
        df_sample = client.query(query_sample).to_dataframe()
        samples.append(df_sample)
        
    final_df = pd.concat(samples)
    print(f"Total samples selected: {len(final_df)}")
    
    # Generate JSONL for Vertex AI
    # Format: {"imageGcsUri": "gs://bucket/file.png"} 
    # Assuming we have plotted images. If raw signals, we might need a custom task.
    # Let's assume we point to the raw file or a plot.
    
    jsonl_data = []
    for _, row in final_df.iterrows():
        jsonl_data.append({
            "imageGcsUri": row['gcs_uri'], # Assuming pre-generated plots
            "classificationAnnotation": {
                "displayName": row['predicted_label'] # Pre-fill suggestion
            }
        })
        
    output_filename = "labeling_request.jsonl"
    with open(output_filename, 'w') as f:
        for entry in jsonl_data:
            json.dump(entry, f)
            f.write('\n')
            
    # Upload to GCS
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(output_bucket)
    blob = bucket.blob(f"labeling/requests/{output_filename}")
    blob.upload_from_filename(output_filename)
    
    print(f"Exported to gs://{output_bucket}/labeling/requests/{output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', required=True)
    parser.add_argument('--source_table', required=True, help="BQ table with predictions")
    parser.add_argument('--output_bucket', required=True)
    
    args = parser.parse_args()
    export_for_labeling(args.project_id, args.source_table, args.output_bucket)
