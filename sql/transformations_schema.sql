CREATE TABLE IF NOT EXISTS `cardio_analytics.signal_transformations` (
    record_id STRING,
    transform_type STRING, -- 'RESAMPLE', 'FILTER', 'NORMALIZE'
    original_fs INT64,
    target_fs INT64,
    original_length INT64,
    new_length INT64,
    scale_factor FLOAT64,
    timestamp TIMESTAMP
);
