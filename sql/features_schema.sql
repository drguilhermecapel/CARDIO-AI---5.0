CREATE TABLE IF NOT EXISTS `cardio_analytics.derived_features` (
    record_id STRING,
    timestamp TIMESTAMP,
    heart_rate FLOAT64,
    mean_rr_ms FLOAT64,
    std_rr_ms FLOAT64,
    mean_qrs_ms FLOAT64,
    qtc_ms FLOAT64,
    power_lf FLOAT64,
    power_hf FLOAT64,
    power_qrs FLOAT64,
    dominant_freq_hz FLOAT64,
    lead_amplitudes_json STRING -- JSON object with min/max/std per lead
);
