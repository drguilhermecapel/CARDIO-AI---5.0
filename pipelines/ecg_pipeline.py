import kfp
from kfp import dsl
from kfp.v2.dsl import component, Input, Output, Dataset, Model, Metrics, Artifact

@component(
    base_image="python:3.9",
    packages_to_install=["google-cloud-bigquery", "pandas", "scikit-learn"]
)
def fetch_ecg_data(
    project_id: str,
    dataset_id: str,
    table_id: str,
    output_dataset: Output[Dataset]
):
    from google.cloud import bigquery
    import pandas as pd

    client = bigquery.Client(project=project_id)
    query = f"""
        SELECT * FROM `{project_id}.{dataset_id}.{table_id}`
        WHERE timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
    """
    df = client.query(query).to_dataframe()
    df.to_csv(output_dataset.path, index=False)

@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "scikit-learn"]
)
def train_drift_detector(
    input_dataset: Input[Dataset],
    drift_model: Output[Model],
    metrics: Output[Metrics]
):
    import pandas as pd
    from sklearn.ensemble import IsolationForest
    import pickle

    df = pd.read_csv(input_dataset.path)
    
    # Simple drift detection using Isolation Forest on numerical features
    features = ['pr_interval', 'qrs_duration', 'qtc_interval']
    X = df[features].dropna()
    
    clf = IsolationForest(contamination=0.05)
    clf.fit(X)
    
    with open(drift_model.path, 'wb') as f:
        pickle.dump(clf, f)
        
    metrics.log_metric("training_samples", len(X))
    metrics.log_metric("drift_threshold", 0.05)

@dsl.pipeline(
    name="ecg-monitoring-pipeline",
    description="Daily pipeline to monitor ECG data drift and quality."
)
def ecg_pipeline(
    project_id: str = "your-project-id",
    dataset_id: str = "cardio_analytics",
    table_id: str = "predictions"
):
    fetch_task = fetch_ecg_data(
        project_id=project_id,
        dataset_id=dataset_id,
        table_id=table_id
    )
    
    train_task = train_drift_detector(
        input_dataset=fetch_task.outputs["output_dataset"]
    )

if __name__ == "__main__":
    kfp.v2.compiler.Compiler().compile(
        pipeline_func=ecg_pipeline,
        package_path="ecg_pipeline.json"
    )
