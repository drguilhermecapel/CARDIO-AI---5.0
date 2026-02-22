CREATE TABLE IF NOT EXISTS `cardio_analytics.bias_metrics` (
    run_id STRING,
    dimension STRING, -- 'age_group', 'sex', 'ethnicity', 'device_type'
    group_value STRING,
    metric_auc FLOAT64,
    metric_f1 FLOAT64,
    sample_count INT64,
    timestamp TIMESTAMP
);
