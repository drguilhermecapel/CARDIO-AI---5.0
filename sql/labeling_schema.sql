CREATE TABLE IF NOT EXISTS `cardio_analytics.ecg_reviews` (
    review_id STRING,
    record_id STRING,
    reviewer_id STRING,
    label STRING,
    confidence FLOAT64,
    review_timestamp TIMESTAMP,
    comments STRING,
    labeling_job_id STRING
);

CREATE TABLE IF NOT EXISTS `cardio_analytics.golden_labels` (
    record_id STRING,
    consensus_label STRING,
    consensus_confidence FLOAT64,
    agreement_score FLOAT64, -- % of reviewers who agreed
    total_reviews INT64,
    last_updated TIMESTAMP,
    status STRING -- 'FINAL', 'NEEDS_SUPER_REVIEW', 'PENDING'
);
