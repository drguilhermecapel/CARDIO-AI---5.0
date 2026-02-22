import functions_framework
import json
import os
from google.cloud import bigquery
from google.cloud import storage
import pandas as pd

# Config
CONSENSUS_THRESHOLD = float(os.environ.get("CONSENSUS_THRESHOLD", 0.6)) # e.g., 3/5 annotators
MIN_ANNOTATORS = int(os.environ.get("MIN_ANNOTATORS", 3))
DESTINATION_TABLE = os.environ.get("DESTINATION_TABLE", "cardio_analytics.gold_standard_labels")

@functions_framework.http
def consolidate_labels(request):
    """
    Cloud Function to process labeling results, apply consensus, and save to BQ.
    Triggered by HTTP or GCS Event (if adapted).
    """
    request_json = request.get_json(silent=True)
    
    if not request_json or 'gcs_uri' not in request_json:
        return 'Error: "gcs_uri" missing in payload', 400
        
    gcs_uri = request_json['gcs_uri']
    print(f"Processing labeling export from: {gcs_uri}")
    
    # 1. Read Data from GCS
    # Vertex AI exports JSONL where each line is a data item with 'labelAnnotations'
    storage_client = storage.Client()
    
    # Parse URI
    parts = gcs_uri.replace("gs://", "").split("/", 1)
    bucket_name = parts[0]
    blob_name = parts[1]
    
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    content = blob.download_as_string().decode('utf-8')
    
    labeled_items = []
    for line in content.strip().split('\n'):
        labeled_items.append(json.loads(line))
        
    print(f"Loaded {len(labeled_items)} items.")
    
    # 2. Apply Consensus Rule
    consolidated_data = []
    
    for item in labeled_items:
        # Item structure depends on Vertex AI export format
        # Simplified assumption:
        # {
        #   "imageGcsUri": "...",
        #   "labelAnnotations": [
        #       {"displayName": "AFIB", "annotationSpecId": "1"},
        #       {"displayName": "AFIB", "annotationSpecId": "1"},
        #       {"displayName": "Normal", "annotationSpecId": "2"}
        #   ]
        # }
        
        uri = item.get('imageGcsUri') or item.get('textPayload')
        annotations = item.get('labelAnnotations', [])
        
        if len(annotations) < MIN_ANNOTATORS:
            print(f"Skipping {uri}: Insufficient annotators ({len(annotations)} < {MIN_ANNOTATORS})")
            continue
            
        # Count votes
        votes = {}
        for ann in annotations:
            label = ann.get('displayName')
            votes[label] = votes.get(label, 0) + 1
            
        # Check Consensus
        total_votes = len(annotations)
        best_label = max(votes, key=votes.get)
        vote_count = votes[best_label]
        agreement = vote_count / total_votes
        
        status = 'ACCEPTED' if agreement >= CONSENSUS_THRESHOLD else 'DISCARDED'
        
        if status == 'ACCEPTED':
            consolidated_data.append({
                "gcs_uri": uri,
                "label": best_label,
                "agreement_score": agreement,
                "num_annotators": total_votes,
                "timestamp": pd.Timestamp.now().isoformat()
            })
            
    print(f"Consolidated {len(consolidated_data)} samples.")
    
    # 3. Write to BigQuery
    if consolidated_data:
        bq_client = bigquery.Client()
        errors = bq_client.insert_rows_json(DESTINATION_TABLE, consolidated_data)
        if errors:
            print(f"BQ Errors: {errors}")
            return f"Error inserting rows: {errors}", 500
            
    return {"status": "success", "processed": len(labeled_items), "accepted": len(consolidated_data)}
