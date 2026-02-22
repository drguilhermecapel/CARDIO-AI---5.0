import argparse
from google.cloud import aiplatform
from google.cloud.aiplatform import model_monitoring

def enable_monitoring(
    project_id,
    location,
    endpoint_name,
    email_alert,
    sampling_rate=0.1,
    drift_threshold=0.1
):
    """
    Enables Vertex AI Model Monitoring for an Endpoint.
    Monitors for Prediction Drift on the 'diagnosis' output.
    """
    aiplatform.init(project=project_id, location=location)

    # 1. Find Endpoint
    endpoints = aiplatform.Endpoint.list(filter=f'display_name="{endpoint_name}"')
    if not endpoints:
        print(f"Endpoint {endpoint_name} not found.")
        return
    endpoint = endpoints[0]
    print(f"Configuring monitoring for Endpoint: {endpoint.resource_name}")

    # 2. Configure Skew/Drift Detection
    # We monitor the output 'diagnosis' (probabilities) for drift
    # If the distribution of predicted classes changes significantly, trigger alert.
    
    # Note: For custom containers, we usually monitor the raw prediction output.
    # Assuming output is a dictionary, we might need to specify the field.
    # Vertex Monitoring for custom containers analyzes the JSON payload logged to BQ.
    
    objective_config = model_monitoring.ObjectiveConfig(
        skew_detection_config=None, # Requires training data schema
        prediction_drift_detection_config=model_monitoring.PredictionDriftDetectionConfig(
            drift_thresholds={
                "diagnosis": model_monitoring.ThresholdConfig(value=drift_threshold)
            }
        )
    )

    # 3. Configure Alerting
    alert_config = model_monitoring.EmailAlertConfig(
        user_emails=[email_alert],
        enable_logging=True
    )

    # 4. Schedule Config (Hourly run)
    schedule_config = model_monitoring.ScheduleConfig(monitor_interval=1)

    # 5. Create Job
    # Note: This requires the Endpoint to have traffic logging enabled to BigQuery.
    # The job will analyze the BQ table associated with the endpoint.
    
    try:
        job = aiplatform.ModelDeploymentMonitoringJob.create(
            display_name=f"monitor-{endpoint_name}",
            logging_sampling_strategy=model_monitoring.RandomSampleConfig(sample_rate=sampling_rate),
            schedule_config=schedule_config,
            alert_config=alert_config,
            objective_configs=objective_config,
            endpoint=endpoint
        )
        print(f"Monitoring Job Created: {job.resource_name}")
    except Exception as e:
        print(f"Failed to create monitoring job: {e}")
        print("Ensure the Endpoint was deployed with 'enable_request_response_logging=True'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', required=True)
    parser.add_argument('--location', default='us-central1')
    parser.add_argument('--endpoint_name', required=True)
    parser.add_argument('--email_alert', required=True)
    
    args = parser.parse_args()
    enable_monitoring(
        args.project_id, args.location, args.endpoint_name, args.email_alert
    )
