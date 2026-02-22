import functions_framework
from google.cloud import bigquery
from google.cloud import storage
import json
import logging
from collections import Counter

# Configuration
PROJECT_ID = "your-project-id"
DATASET_ID = "cardio_analytics"
REVIEWS_TABLE = "ecg_reviews"
GOLDEN_TABLE = "golden_labels"
CONSENSUS_THRESHOLD = 0.66  # 2/3 majority required

@functions_framework.cloud_event
def process_labeling_export(cloud_event):
    """
    Triggered by a file upload to GCS (Labeling Service Export).
    Parses JSONL, ingests reviews, calculates consensus, updates BigQuery.
    """
    data = cloud_event.data
    bucket_name = data["bucket"]
    file_name = data["name"]
    
    if not file_name.endswith(".jsonl"):
        logging.info(f"Skipping non-JSONL file: {file_name}")
        return

    logging.info(f"Processing labeling export: gs://{bucket_name}/{file_name}")
    
    storage_client = storage.Client()
    bq_client = bigquery.Client()
    
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    
    # Download and parse JSONL
    content = blob.download_as_text()
    lines = content.strip().split('\n')
    
    reviews_to_insert = []
    records_to_update = set()
    
    for line in lines:
        try:
            # Assuming Vertex AI Data Labeling / Standard JSONL format
            # { "imageGcsUri": "...", "labelAnnotations": [ { "displayName": "AFIB", "annotationSpecId": "..." } ], "dataItemResourceLabels": { ... } }
            entry = json.loads(line)
            
            gcs_uri = entry.get("imageGcsUri") or entry.get("textPayload") # Adjust based on actual format
            # Extract Record ID from URI (e.g., gs://bucket/ecgs/RECORD_123.png)
            record_id = gcs_uri.split('/')[-1].replace('.png', '').replace('.json', '')
            
            # Extract Annotations (Simulating multiple workers per item if present, or single)
            # In Vertex AI export, usually one line per data item with aggregated or individual annotations.
            # Let's assume we get individual worker responses or we parse the 'confidences'.
            
            # Simplified Logic: 
            # If the export contains "humanAnnotationPayload", use that.
            annotations = entry.get("labelAnnotations", [])
            
            for ann in annotations:
                label = ann.get("displayName")
                reviewer_id = ann.get("annotationSpecId", "unknown_worker") # Or worker_id if available
                
                review = {
                    "review_id": f"{record_id}_{reviewer_id}_{len(reviews_to_insert)}",
                    "record_id": record_id,
                    "reviewer_id": reviewer_id,
                    "label": label,
                    "confidence": 1.0, # Human labels usually 1.0 unless specified
                    "review_timestamp": "TIMESTAMP(CURRENT_TIMESTAMP())", # Placeholder for SQL
                    "comments": "",
                    "labeling_job_id": file_name
                }
                reviews_to_insert.append(review)
                records_to_update.add(record_id)
                
        except Exception as e:
            logging.error(f"Error parsing line: {e}")

    # 1. Insert Raw Reviews
    if reviews_to_insert:
        errors = bq_client.insert_rows_json(f"{PROJECT_ID}.{DATASET_ID}.{REVIEWS_TABLE}", reviews_to_insert)
        if errors:
            logging.error(f"BQ Insert Errors: {errors}")
        else:
            logging.info(f"Inserted {len(reviews_to_insert)} reviews.")

    # 2. Run Consensus Logic for affected records
    # We'll do this via a BQ Merge Query for efficiency/atomicity
    if records_to_update:
        run_consensus_update(bq_client, list(records_to_update))

def run_consensus_update(client, record_ids):
    """
    Calculates consensus for the given records and updates the Golden Table.
    """
    # Parameterized query to prevent injection (though IDs are internal)
    # We use a temporary table or just a WHERE IN clause if list is small.
    # For scalability, we'll run a query that recalculates for ALL modified records.
    
    query = f"""
    MERGE `{PROJECT_ID}.{DATASET_ID}.{GOLDEN_TABLE}` T
    USING (
        SELECT
            record_id,
            APPROX_TOP_COUNT(label, 1)[OFFSET(0)].value as consensus_label,
            COUNT(*) as total_reviews,
            -- Calculate Agreement Score: Count of Majority Label / Total
            (APPROX_TOP_COUNT(label, 1)[OFFSET(0)].count / COUNT(*)) as agreement_score
        FROM `{PROJECT_ID}.{DATASET_ID}.{REVIEWS_TABLE}`
        WHERE record_id IN UNNEST(@record_ids)
        GROUP BY record_id
    ) S
    ON T.record_id = S.record_id
    WHEN MATCHED THEN
        UPDATE SET
            consensus_label = S.consensus_label,
            agreement_score = S.agreement_score,
            total_reviews = S.total_reviews,
            last_updated = CURRENT_TIMESTAMP(),
            status = CASE 
                WHEN S.agreement_score >= {CONSENSUS_THRESHOLD} THEN 'FINAL'
                ELSE 'NEEDS_SUPER_REVIEW'
            END
    WHEN NOT MATCHED THEN
        INSERT (record_id, consensus_label, consensus_confidence, agreement_score, total_reviews, last_updated, status)
        VALUES (
            S.record_id, 
            S.consensus_label, 
            1.0, -- Human consensus assumed high confidence
            S.agreement_score, 
            S.total_reviews, 
            CURRENT_TIMESTAMP(),
            CASE 
                WHEN S.agreement_score >= {CONSENSUS_THRESHOLD} THEN 'FINAL'
                ELSE 'NEEDS_SUPER_REVIEW'
            END
        )
    """
    
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("record_ids", "STRING", record_ids)
        ]
    )
    
    job = client.query(query, job_config=job_config)
    job.result()
    logging.info(f"Updated consensus for {len(record_ids)} records.")
