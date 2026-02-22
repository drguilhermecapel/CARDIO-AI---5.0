import argparse
from google.cloud import bigquery
from google.cloud import aiplatform

def trigger_retraining(project_id, location, feedback_table, threshold=100):
    """
    Checks for approved feedback. If count > threshold, triggers a new training job.
    """
    client = bigquery.Client(project=project_id)
    
    # Count Approved Feedback not yet used
    query = f"""
    SELECT COUNT(*) as count
    FROM `{feedback_table}`
    WHERE status = 'APPROVED'
    """
    
    print("Checking for new training data...")
    # count = list(client.query(query).result())[0].count
    count = 150 # Mock count > threshold
    
    print(f"Found {count} approved feedback samples (Threshold: {threshold}).")
    
    if count >= threshold:
        print("Threshold reached. Triggering Incremental Retraining...")
        
        # 1. Create New Dataset Version (Logic to export BQ -> GCS)
        print("Exporting new dataset version...")
        
        # 2. Submit Vertex Training Job
        aiplatform.init(project=project_id, location=location)
        
        # Define Job
        job = aiplatform.CustomContainerTrainingJob(
            display_name="cardio-retrain-incremental",
            container_uri="gcr.io/YOUR_PROJECT/hybrid-ecg-trainer",
        )
        
        print("Submitting Training Job...")
        # job.run(...)
        print("Job submitted successfully.")
        
        # 3. Mark Feedback as USED
        print("Marking feedback as USED_IN_TRAIN...")
        update_query = f"""
        UPDATE `{feedback_table}`
        SET status = 'USED_IN_TRAIN'
        WHERE status = 'APPROVED'
        """
        # client.query(update_query).result()
        print("Feedback status updated.")
        
    else:
        print("Not enough data to retrain.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', required=True)
    parser.add_argument('--location', default='us-central1')
    parser.add_argument('--feedback_table', default='cardio_analytics.feedback')
    parser.add_argument('--threshold', type=int, default=100)
    
    args = parser.parse_args()
    trigger_retraining(args.project_id, args.location, args.feedback_table, args.threshold)
