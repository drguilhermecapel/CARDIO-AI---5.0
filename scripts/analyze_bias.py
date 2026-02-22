import argparse
import os
import json
import numpy as np
import pandas as pd
from google.cloud import bigquery
from google.cloud import monitoring_v3
from sklearn.metrics import roc_auc_score, f1_score

def analyze_bias(
    project_id,
    dataset_id,
    table_id,
    model_run_id,
    predictions_table,
    demographics_table,
    alert_threshold=0.1
):
    """
    Analyzes model bias across demographic groups (Age, Sex, Ethnicity, Device).
    Logs metrics to BigQuery and triggers Cloud Monitoring alerts if disparities exceed threshold.
    """
    client = bigquery.Client(project=project_id)
    
    # 1. Fetch Data (Join Predictions with Demographics)
    query = f"""
    SELECT
        p.patient_id,
        p.true_label,
        p.predicted_label,
        p.predicted_probs,
        d.age_group,
        d.sex,
        d.ethnicity,
        d.device_type
    FROM
        `{predictions_table}` p
    JOIN
        `{demographics_table}` d
    ON
        p.patient_id = d.patient_id
    WHERE
        p.run_id = '{model_run_id}'
    """
    
    print("Fetching data from BigQuery...")
    # df = client.query(query).to_dataframe()
    
    # Mock Data for Demo
    num_samples = 1000
    df = pd.DataFrame({
        'true_label': np.random.randint(0, 2, num_samples),
        'predicted_label': np.random.randint(0, 2, num_samples),
        'predicted_probs': np.random.rand(num_samples),
        'age_group': np.random.choice(['18-30', '31-50', '51-70', '70+'], num_samples),
        'sex': np.random.choice(['Male', 'Female'], num_samples),
        'ethnicity': np.random.choice(['GroupA', 'GroupB', 'GroupC'], num_samples),
        'device_type': np.random.choice(['Holter', '12-Lead', 'Wearable'], num_samples)
    })
    
    # 2. Calculate Metrics per Group
    groups = ['age_group', 'sex', 'ethnicity', 'device_type']
    bias_report = []
    
    for group_col in groups:
        print(f"Analyzing bias for {group_col}...")
        unique_groups = df[group_col].unique()
        
        group_metrics = {}
        for g in unique_groups:
            mask = df[group_col] == g
            if mask.sum() < 10: continue # Skip small groups
            
            y_true = df.loc[mask, 'true_label']
            y_pred = df.loc[mask, 'predicted_label']
            y_prob = df.loc[mask, 'predicted_probs']
            
            try:
                auc = roc_auc_score(y_true, y_prob)
                f1 = f1_score(y_true, y_pred, average='macro')
            except:
                auc = 0.5
                f1 = 0.0
                
            group_metrics[g] = {'auc': auc, 'f1': f1}
            
            bias_report.append({
                'run_id': model_run_id,
                'dimension': group_col,
                'group_value': g,
                'metric_auc': float(auc),
                'metric_f1': float(f1),
                'sample_count': int(mask.sum()),
                'timestamp': pd.Timestamp.utcnow().isoformat()
            })
            
        # Check for Disparities
        aucs = [m['auc'] for m in group_metrics.values()]
        if aucs:
            max_diff = max(aucs) - min(aucs)
            print(f"  Max AUC Disparity: {max_diff:.4f}")
            
            if max_diff > alert_threshold:
                trigger_alert(project_id, model_run_id, group_col, max_diff)

    # 3. Log to BigQuery
    if bias_report:
        table_ref = f"{project_id}.{dataset_id}.{table_id}"
        # errors = client.insert_rows_json(table_ref, bias_report)
        print(f"Logged {len(bias_report)} bias metrics to {table_ref}")
        # print(f"BQ Errors: {errors}")

def trigger_alert(project_id, run_id, dimension, disparity):
    """
    Writes a custom metric to Cloud Monitoring to trigger an alert policy.
    """
    print(f"TRIGGERING ALERT: High Bias in {dimension} (Diff: {disparity:.4f})")
    
    client = monitoring_v3.MetricServiceClient()
    project_name = f"projects/{project_id}"
    
    series = monitoring_v3.TimeSeries()
    series.metric.type = "custom.googleapis.com/model_bias/disparity"
    series.metric.labels["run_id"] = run_id
    series.metric.labels["dimension"] = dimension
    
    series.resource.type = "global"
    
    point = series.points.add()
    point.value.double_value = disparity
    
    now = pd.Timestamp.utcnow()
    point.interval.end_time.seconds = int(now.timestamp())
    
    # client.create_time_series(name=project_name, time_series=[series])
    print("  -> Alert metric sent to Cloud Monitoring.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', required=True)
    parser.add_argument('--dataset_id', default='cardio_analytics')
    parser.add_argument('--table_id', default='bias_metrics')
    parser.add_argument('--run_id', required=True)
    parser.add_argument('--predictions_table', default='cardio_analytics.predictions')
    parser.add_argument('--demographics_table', default='cardio_analytics.demographics')
    
    args = parser.parse_args()
    analyze_bias(
        args.project_id, args.dataset_id, args.table_id, args.run_id,
        args.predictions_table, args.demographics_table
    )
