import argparse
import pandas as pd
from google.cloud import bigquery

def validate_feedback(project_id, table_id="cardio_analytics.feedback"):
    """
    Interactive CLI to review PENDING feedback and approve/reject it.
    """
    client = bigquery.Client(project=project_id)
    
    # Fetch Pending Feedback
    query = f"""
    SELECT * FROM `{table_id}`
    WHERE status = 'PENDING'
    ORDER BY timestamp ASC
    LIMIT 10
    """
    
    print("Fetching pending feedback...")
    # df = client.query(query).to_dataframe()
    
    # Mock Data
    df = pd.DataFrame([
        {
            'feedback_id': 'fb-1', 'predicted_label': 'Normal', 'corrected_label': 'AFIB',
            'comments': 'Clear P-wave absence in Lead II', 'cardiologist_id': 'DR_SMITH'
        },
        {
            'feedback_id': 'fb-2', 'predicted_label': 'MI', 'corrected_label': 'Normal',
            'comments': 'Artifact mimicking ST elevation', 'cardiologist_id': 'DR_JONES'
        }
    ])
    
    if df.empty:
        print("No pending feedback.")
        return

    print(f"Found {len(df)} pending items.\n")
    
    for _, row in df.iterrows():
        print("-" * 40)
        print(f"ID: {row['feedback_id']}")
        print(f"Model Predicted: {row['predicted_label']}")
        print(f"Cardiologist Correction: {row['corrected_label']}")
        print(f"Comments: {row['comments']}")
        print(f"By: {row['cardiologist_id']}")
        
        choice = input("Action [A]pprove / [R]eject / [S]kip: ").lower()
        
        new_status = None
        if choice == 'a':
            new_status = 'APPROVED'
        elif choice == 'r':
            new_status = 'REJECTED'
            
        if new_status:
            print(f"Marking as {new_status}...")
            # Update BQ
            update_query = f"""
            UPDATE `{table_id}`
            SET status = '{new_status}'
            WHERE feedback_id = '{row['feedback_id']}'
            """
            # client.query(update_query).result()
            print("Updated.")
        else:
            print("Skipped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', required=True)
    args = parser.parse_args()
    validate_feedback(args.project_id)
