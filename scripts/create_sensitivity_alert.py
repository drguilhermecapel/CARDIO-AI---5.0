import argparse
from google.cloud import monitoring_v3

def create_alert_policy(project_id, email_address):
    client = monitoring_v3.AlertPolicyServiceClient()
    notification_client = monitoring_v3.NotificationChannelServiceClient()
    project_name = f"projects/{project_id}"

    # 1. Create Notification Channel
    channel = monitoring_v3.NotificationChannel(
        type_="email",
        display_name="Cardio AI Team",
        labels={"email_address": email_address}
    )
    channel = notification_client.create_notification_channel(name=project_name, notification_channel=channel)
    print(f"Notification Channel created: {channel.name}")

    # 2. Create Alert Policy
    # Condition: sensitivity_drop_7d > 0.05 (5%)
    
    condition = monitoring_v3.AlertPolicy.Condition(
        display_name="Sensitivity Drop > 5%",
        condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
            filter='metric.type = "custom.googleapis.com/cardio_ai/sensitivity_drop_7d" AND resource.type = "global"',
            comparison=monitoring_v3.ComparisonType.COMPARISON_GT,
            threshold_value=0.05,
            duration={"seconds": 0}, # Instant trigger upon metric ingestion
            aggregations=[
                monitoring_v3.Aggregation(
                    alignment_period={"seconds": 60},
                    per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_MEAN
                )
            ]
        )
    )

    policy = monitoring_v3.AlertPolicy(
        display_name="Cardio AI - Clinical Regression Alert",
        conditions=[condition],
        combiner=monitoring_v3.AlertPolicy.ConditionCombinerType.AND,
        notification_channels=[channel.name],
        documentation=monitoring_v3.AlertPolicy.Documentation(
            content="Sensitivity for a pathology has dropped by >5% compared to last week. Investigate immediately.",
            mime_type="text/markdown"
        )
    )

    policy = client.create_alert_policy(name=project_name, alert_policy=policy)
    print(f"Alert Policy created: {policy.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', required=True)
    parser.add_argument('--email', required=True)
    
    args = parser.parse_args()
    create_alert_policy(args.project_id, args.email)
