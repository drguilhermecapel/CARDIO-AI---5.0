import argparse
import time
import pandas as pd
from google.cloud import bigquery
from google.cloud import monitoring_v3

def compute_and_publish(project_id, feedback_table="cardio_analytics.feedback"):
    client = bigquery.Client(project=project_id)
    
    # 1. Fetch Data (Approved Feedback only)
    # We need predictions vs corrected labels
    # We'll calculate metrics for two windows: [Now-7d, Now] and [Now-14d, Now-7d]
    
    query = f"""
    SELECT 
        predicted_label, 
        corrected_label, 
        timestamp
    FROM `{feedback_table}`
    WHERE status IN ('APPROVED', 'USED_IN_TRAIN')
    AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 14 DAY)
    """
    
    try:
        df = client.query(query).to_dataframe()
    except Exception as e:
        print(f"Error querying BigQuery: {e}")
        return

    if df.empty:
        print("No feedback data found in the last 14 days.")
        return

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    now = pd.Timestamp.now(tz='UTC')
    window_current = df[df['timestamp'] >= (now - pd.Timedelta(days=7))]
    window_past = df[(df['timestamp'] < (now - pd.Timedelta(days=7))) & (df['timestamp'] >= (now - pd.Timedelta(days=14)))]
    
    classes = ["MI", "AFIB", "PVC", "Normal"]
    MIN_SAMPLES = 5
    
    client_metric = monitoring_v3.MetricServiceClient()
    project_name = f"projects/{project_id}"
    
    for cls in classes:
        # Helper to calc sensitivity
        def calc_sens(d):
            truth_pos = d[d['corrected_label'] == cls]
            if len(truth_pos) < MIN_SAMPLES:
                return None # Insufficient data
            
            tp = len(truth_pos[truth_pos['predicted_label'] == cls])
            return tp / len(truth_pos)

        sens_curr = calc_sens(window_current)
        sens_past = calc_sens(window_past)
        
        if sens_curr is not None:
            print(f"[{cls}] Current Sens: {sens_curr:.2f}")
            # Publish Current Sensitivity
            series = monitoring_v3.TimeSeries()
            series.metric.type = "custom.googleapis.com/cardio_ai/sensitivity"
            series.metric.labels["pathology"] = cls
            series.resource.type = "global"
            point = series.points.add()
            point.value.double_value = sens_curr
            point.interval.end_time.seconds = int(time.time())
            client_metric.create_time_series(name=project_name, time_series=[series])
        else:
            print(f"[{cls}] Current Sens: Insufficient Data (<{MIN_SAMPLES} samples)")

        if sens_curr is not None and sens_past is not None:
            drop = sens_past - sens_curr
            print(f"[{cls}] Drop: {drop:.2f} (Past: {sens_past:.2f})")
            
            # Publish Drop
            series_drop = monitoring_v3.TimeSeries()
            series_drop.metric.type = "custom.googleapis.com/cardio_ai/sensitivity_drop_7d"
            series_drop.metric.labels["pathology"] = cls
            series_drop.resource.type = "global"
            point_drop = series_drop.points.add()
            point_drop.value.double_value = drop
            point_drop.interval.end_time.seconds = int(time.time())
            client_metric.create_time_series(name=project_name, time_series=[series_drop])
        else:
            print(f"[{cls}] Drop: Cannot calculate (insufficient history)")

    print("Metrics published to Cloud Monitoring.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', required=True)
    parser.add_argument('--feedback_table', default="cardio_analytics.feedback")
    
    args = parser.parse_args()
    compute_and_publish(args.project_id, args.feedback_table)
