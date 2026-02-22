import argparse
import json
from google.cloud import bigquery

def calculate_weights(project_id, run_id, table_id='cardio_analytics.bias_metrics'):
    """
    Calculates reweighting factors based on bias metrics.
    Weight = 1 / (F1_Score + epsilon)
    """
    client = bigquery.Client(project=project_id)
    
    query = f"""
    SELECT dimension, group_value, metric_f1
    FROM `{table_id}`
    WHERE run_id = '{run_id}'
    """
    
    print(f"Fetching bias metrics for run {run_id}...")
    # df = client.query(query).to_dataframe()
    
    # Mock Data
    import pandas as pd
    df = pd.DataFrame([
        {'dimension': 'sex', 'group_value': 'Male', 'metric_f1': 0.85},
        {'dimension': 'sex', 'group_value': 'Female', 'metric_f1': 0.65}, # Disparity!
    ])
    
    weights = {}
    epsilon = 0.05
    
    # Normalize weights so mean is 1.0? Or just boost?
    # Let's boost lower performing groups.
    
    # Focus on Sex for this demo
    sex_metrics = df[df['dimension'] == 'sex']
    
    if not sex_metrics.empty:
        # Calculate inverse performance
        inv_perf = 1.0 / (sex_metrics['metric_f1'] + epsilon)
        # Normalize to mean 1.0
        norm_weights = inv_perf / inv_perf.mean()
        
        for i, row in sex_metrics.iterrows():
            group = row['group_value']
            w = norm_weights[i]
            weights[group] = float(w)
            print(f"  Group {group}: F1={row['metric_f1']:.2f} -> Weight={w:.2f}")
            
    return json.dumps(weights)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', required=True)
    parser.add_argument('--run_id', required=True)
    
    args = parser.parse_args()
    weights_json = calculate_weights(args.project_id, args.run_id)
    print(f"FAIRNESS_WEIGHTS={weights_json}")
