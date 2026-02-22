import kfp
from kfp import dsl
from kfp.v2.dsl import component, Input, Output, Dataset, Metrics, Artifact, HTML

@component(
    base_image="python:3.9",
    packages_to_install=["google-cloud-aiplatform", "pandas", "numpy", "scikit-learn", "matplotlib"]
)
def compare_models(
    project_id: str,
    location: str,
    experiment_name: str,
    validation_dataset: Input[Dataset],
    comparison_report: Output[HTML],
    best_model_metrics: Output[Metrics]
):
    from google.cloud import aiplatform
    import pandas as pd
    import numpy as np
    from sklearn.metrics import confusion_matrix, roc_auc_score
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO

    aiplatform.init(project=project_id, location=location, experiment=experiment_name)
    
    # 1. Fetch Experiment Runs
    # We assume runs have logged metrics like 'pathology_auc', 'val_loss', etc.
    # And potentially 'model_resource_name' as a parameter or artifact.
    df_runs = aiplatform.get_experiment_df(experiment=experiment_name)
    
    if df_runs.empty:
        print("No runs found in experiment.")
        return

    # Filter for successful runs
    df_runs = df_runs[df_runs['state'] == 'COMPLETE']
    
    # 2. Evaluate "Best" Candidates on Validation Set (if not already done)
    # Ideally, each run logs metrics on the same validation set.
    # Here we aggregate those logged metrics.
    
    # Extract relevant metrics
    # Assuming metrics are named 'metric.pathology_auc', 'metric.pathology_loss'
    metric_cols = [c for c in df_runs.columns if c.startswith('metric.')]
    param_cols = [c for c in df_runs.columns if c.startswith('param.')]
    
    summary = df_runs[['run_name'] + metric_cols + param_cols].copy()
    
    # Clean column names
    summary.columns = [c.replace('metric.', '').replace('param.', '') for c in summary.columns]
    
    # Sort by AUC
    if 'pathology_auc' in summary.columns:
        summary = summary.sort_values('pathology_auc', ascending=False)
    
    # Generate HTML Report
    html_content = """
    <html>
    <head>
        <title>Model Comparison Report</title>
        <style>
            body { font-family: sans-serif; padding: 20px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .best { background-color: #e6fffa; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>Model Comparison: Custom vs AutoML</h1>
        <p>Experiment: <strong>%s</strong></p>
        
        <h2>Leaderboard (Sorted by AUC)</h2>
        <table>
            <tr>
                <th>Run Name</th>
                <th>Architecture</th>
                <th>AUC</th>
                <th>Accuracy</th>
                <th>Loss</th>
            </tr>
    """ % experiment_name
    
    for i, row in summary.iterrows():
        row_class = "best" if i == summary.index[0] else ""
        arch = row.get('architecture', 'Unknown')
        auc = row.get('pathology_auc', 'N/A')
        acc = row.get('final_accuracy', row.get('accuracy', 'N/A'))
        loss = row.get('loss', 'N/A')
        
        html_content += f"""
            <tr class="{row_class}">
                <td>{row['run_name']}</td>
                <td>{arch}</td>
                <td>{auc}</td>
                <td>{acc}</td>
                <td>{loss}</td>
            </tr>
        """
        
    html_content += """
        </table>
        
        <h2>Detailed Analysis</h2>
        <p>Ranking based on Sensitivity (Recall) and Specificity per pathology is available in the detailed logs.</p>
    </body>
    </html>
    """
    
    with open(comparison_report.path, 'w') as f:
        f.write(html_content)
        
    # Log Best Model Metrics to Output
    best_run = summary.iloc[0]
    best_model_metrics.log_metric("best_run_id", best_run['run_name'])
    best_model_metrics.log_metric("best_auc", best_run.get('pathology_auc', 0))

@dsl.pipeline(
    name="model-comparison-pipeline",
    description="Compare AutoML and Custom models from Vertex Experiments."
)
def comparison_pipeline(
    project_id: str,
    location: str,
    experiment_name: str,
    validation_dataset_uri: str
):
    # Load Validation Data (Dummy step for pipeline graph)
    # In reality, you might load a specific golden dataset here
    
    compare_task = compare_models(
        project_id=project_id,
        location=location,
        experiment_name=experiment_name,
        validation_dataset=dsl.Importer(
            artifact_uri=validation_dataset_uri,
            artifact_class=Dataset
        ).output
    )
