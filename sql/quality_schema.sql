CREATE TABLE IF NOT EXISTS `cardio_analytics.beat_quality_reports` (
    record_id STRING,
    timestamp TIMESTAMP,
    total_beats INT64,
    valid_beats INT64,
    artifact_beats INT64,
    quality_score FLOAT64, -- valid / total
    average_sqi FLOAT64, -- average correlation/kurtosis score
    processing_method STRING DEFAULT 'LightweightArtifactDetector_v1'
);
