CREATE TABLE IF NOT EXISTS `cardio_analytics.feedback` (
    feedback_id STRING,
    prediction_id STRING,
    patient_id STRING,
    model_version STRING,
    predicted_label STRING,
    corrected_label STRING, -- The cardiologist's correction
    confidence_score FLOAT64, -- Cardiologist's confidence in correction (optional)
    cardiologist_id STRING,
    comments STRING,
    timestamp TIMESTAMP,
    status STRING -- 'PENDING', 'APPROVED', 'REJECTED', 'USED_IN_TRAIN'
);
