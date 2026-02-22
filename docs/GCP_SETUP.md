# GCP Integration Setup Guide

This application integrates with Google Cloud Platform (GCP) services for storage, analytics, and ML operations.

## Prerequisites

1.  **Google Cloud Project**: Create a project in the Google Cloud Console.
2.  **Service Account**: Create a service account with the necessary roles (see below).
3.  **Key File**: Download the JSON key file for the service account and set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to its path.

## Required IAM Roles

Grant the following roles to your service account:

-   **Cloud Storage**: `Storage Object Admin` (to read/write ECG images to buckets).
-   **BigQuery**: `BigQuery Data Editor` (to insert analysis rows) and `BigQuery Job User` (to run queries).
-   **Vertex AI**: `Vertex AI User` (to run pipelines and access endpoints).

## Configuration

### 1. Cloud Storage (GCS)
Create a bucket for storing ECG artifacts (images, PDFs).
-   **Bucket Name**: `cardio-ai-artifacts` (or update `GCS_BUCKET_NAME` in `.env`)
-   **Location**: Choose a region close to your users (e.g., `us-central1`).

### 2. BigQuery
Create a dataset for analytics.
-   **Dataset ID**: `cardio_analytics`
-   **Table ID**: `predictions`
-   **Schema**:
    -   `analysis_id`: STRING
    -   `timestamp`: TIMESTAMP
    -   `diagnosis`: STRING
    -   `confidence`: STRING
    -   `urgency`: STRING
    -   `heart_rate`: STRING
    -   `rhythm`: STRING
    -   `pr_interval`: FLOAT
    -   `qrs_duration`: FLOAT
    -   `qtc_interval`: FLOAT
    -   `gcs_uri`: STRING
    -   `model_version`: STRING
    -   `patient_hash`: STRING

### 3. Vertex AI Pipelines
The pipeline definition is located in `pipelines/ecg_pipeline.py`.
To deploy:
1.  Install the Kubeflow Pipelines SDK: `pip install kfp google-cloud-aiplatform`
2.  Compile the pipeline: `python pipelines/ecg_pipeline.py`
3.  Upload `ecg_pipeline.json` to Vertex AI Pipelines in the Cloud Console.

### 4. Ingestion Pipeline (WFDB/CSV)
The pipeline definition is located in `pipelines/ingestion_pipeline.py`.
This pipeline ingests raw WFDB files (PhysioNet format) from GCS, extracts metadata, registers them in BigQuery, and generates a hierarchical HTML catalog.

**To deploy:**
1.  Compile: `python pipelines/ingestion_pipeline.py`
2.  Upload `ingestion_pipeline.json` to Vertex AI.
3.  Run with parameters:
    -   `gcs_bucket`: Your bucket name (e.g., `cardio-ai-raw-data`)
    -   `gcs_prefix`: Path to WFDB files (e.g., `ptb-xl/`)
    -   `project_id`: Your GCP Project ID
    -   `bq_dataset`: `cardio_analytics`
    -   `bq_table`: `raw_ecg_metadata`

### 5. Dataflow Validation Job
The Dataflow job `dataflow/validate_ecg.py` validates ECG signal integrity (sampling rate, leads, duration) and quarantines invalid files.

**To run:**
1.  Navigate to `dataflow/` directory.
2.  Install dependencies: `pip install -r requirements.txt`
3.  Run the job (DirectRunner for testing, DataflowRunner for cloud):

```bash
python validate_ecg.py \
  --runner DataflowRunner \
  --project YOUR_PROJECT_ID \
  --region us-central1 \
  --temp_location gs://YOUR_BUCKET/temp \
  --input_bucket cardio-ai-raw-data \
  --quarantine_bucket cardio-ai-quarantine \
  --output_table cardio_analytics.validated_ecgs \
  --setup_file ./setup.py
```

### 6. Data Labeling & Consensus
See `docs/LABELING_WORKFLOW.md` for details on setting up the Human-in-the-Loop pipeline.
This includes a Cloud Function that ingests labeling results from GCS, calculates consensus (Majority Vote), and updates the "Golden Dataset" in BigQuery.

### 7. Signal Processing Pipeline Component
The component `pipelines/signal_processing.py` implements adaptive filtering (Baseline Wander, Notch, Lowpass) using zero-phase filtering to preserve ECG morphology.
It calculates and logs SNR and RMSE metrics to Vertex AI ML Metadata.

**Usage:**
Import the `adaptive_ecg_filter` component in your pipeline definition:

```python
from pipelines.signal_processing import adaptive_ecg_filter

# ... inside pipeline ...
process_task = adaptive_ecg_filter(
    input_dataset=ingestion_task.outputs["output_dataset"],
    processed_bucket="cardio-ai-processed-data",
    sampling_rate=500
)
```

### 8. Beat-Level Artifact Detection
The component `pipelines/beat_quality.py` implements a lightweight beat detector and quality classifier.
It generates a report in BigQuery table `beat_quality_reports`.

**Setup:**
1.  Create the table: `bq query --use_legacy_sql=false < sql/quality_schema.sql`
2.  Import `detect_artifacts_batch` in your pipeline.

**Logic:**
-   **QRS Detection**: Energy-based thresholding (Pan-Tompkins simplified).
-   **SQI Metrics**: Kurtosis (sharpness) and Template Matching (correlation).
-   **Classification**: Beats with low correlation (<0.8) or abnormal kurtosis are marked as artifacts.

### 9. Resampling & Transformation Tracking
The component `pipelines/resampling.py` standardizes sampling rates (e.g., to 500Hz) using deterministic resampling.
Crucially, it logs transformation metadata to `cardio_analytics.signal_transformations` to allow **coordinate inversion**.

**Setup:**
1.  Create table: `bq query --use_legacy_sql=false < sql/transformations_schema.sql`
2.  Use `utils/xai_utils.py` in your inference/XAI service to map model saliency back to the original raw signal.

### 10. Parallel Feature Extraction (Dataflow)
The job `dataflow/extract_features.py` extracts classical (PR, QRS, QTc) and spectral (PSD) features from validated ECGs in parallel.

**Setup:**
1.  Create table: `bq query --use_legacy_sql=false < sql/features_schema.sql`
2.  Run Dataflow job:

```bash
python dataflow/extract_features.py \
  --runner DataflowRunner \
  --project YOUR_PROJECT_ID \
  --region us-central1 \
  --temp_location gs://YOUR_BUCKET/temp \
  --input_table cardio_analytics.validated_ecgs \
  --output_table cardio_analytics.derived_features \
  --setup_file ./dataflow/setup.py
```

### 11. ECG Data Augmentation (TensorFlow)
The module `models/augmentations.py` implements GPU-accelerated augmentations for ECG signals, including:
-   **Baseline Wander**: Simulates breathing artifacts.
-   **Gaussian Noise**: Simulates sensor noise.
-   **Lead Dropout**: Simulates electrode failure.
-   **Time Stretch**: Simulates heart rate variability.

**Usage:**
Integrate into your `tf.data` pipeline:

```python
from models.augmentations import augment_ecg

dataset = tf.data.Dataset.from_tensor_slices(...)
# Apply only during training
dataset = dataset.map(augment_ecg, num_parallel_calls=tf.data.AUTOTUNE)
```

Run `python models/train_demo.py` to verify the pipeline.

### 12. Synthetic Data Generation (CVAE)
The `models/cvae` directory contains a Conditional Variational Autoencoder to generate synthetic ECGs for rare classes.

**Workflow:**
1.  **Train**: Submit a Vertex AI Custom Job using `models/cvae/Dockerfile`.
2.  **Generate**: Run `models/cvae/generate.py` to create samples in `gs://BUCKET/staging/`.
3.  **Validate**: Run `scripts/promote_synthetic.py` (simulating a cardiologist review tool) to move valid samples to `gs://BUCKET/approved/`.

**Training Command (Local Test):**
```bash
cd models/cvae
python task.py --job-dir /tmp/cvae_model
```

### 13. Hybrid CNN-Transformer Training (Vertex AI)
The `models/hybrid` directory contains the definition for a state-of-the-art ECG classifier combining CNNs (local features) and Transformers (global context).

**Features:**
-   **Multi-Task Learning**: Simultaneously predicts **Pathologies** (Multi-label) and **Lead Quality** (Per-lead).
-   **Weighted Loss**: Applies clinical importance weights (e.g., MI errors are penalized 5x more than Normal).
-   **Experiment Tracking**: Logs AUC, Precision, Recall, and Quality Accuracy to Vertex AI Experiments and TensorBoard.
-   **Checkpointing**: Saves model checkpoints to GCS during training.

**To Submit Job:**
1.  Build Docker image: `gcloud builds submit --tag gcr.io/YOUR_PROJECT/hybrid-ecg-trainer models/hybrid/`
2.  Submit Custom Job via Vertex AI Console or SDK, pointing to the image.

### 14. Regularization & Hyperparameter Tuning
The training script supports advanced regularization techniques, controllable via command-line flags (ideal for Vertex Hyperparameter Tuning):

-   **Mixup**: `--use_mixup` (Blends pairs of inputs/labels).
-   **Focal Loss**: `--loss_type focal` (Focuses on hard examples).
-   **Label Smoothing**: `--loss_type label_smoothing` (Prevents overconfidence).

**Example Tuning Config (`hptuning.yaml`):**
```yaml
studySpec:
  metrics:
  - metricId: pathology_auc
    goal: MAXIMIZE
  parameters:
  - parameterId: learning_rate
    scaleType: UNIT_LOG_SCALE
    doubleValueSpec:
      minValue: 1e-4
      maxValue: 1e-2
  - parameterId: loss_type
    categoricalValueSpec:
      values:
      - "weighted_bce"
      - "focal"
      - "label_smoothing"
  - parameterId: use_mixup
    categoricalValueSpec:
      values:
      - "true"
      - "false"
```

### 15. Model Comparison & Ranking
The pipeline `pipelines/comparison_pipeline.py` aggregates results from Vertex AI Experiments (both Custom and AutoML runs).
It generates an HTML leaderboard ranking models by AUC, Accuracy, and Loss.

**Usage:**
1.  Ensure all training jobs log to the same `experiment_name`.
2.  Compile and run the comparison pipeline:
    ```bash
    python pipelines/comparison_pipeline.py
    ```
3.  View the generated HTML report in the Vertex AI Pipelines UI (Artifacts tab).

### 16. Stratified Evaluation & Reporting
The script `scripts/evaluate_stratified.py` performs rigorous model evaluation:
-   **Metrics**: Macro F1, Brier Score, Per-Class AUC.
-   **Stratification**: Breaks down performance by **Center** (hospital) and **Patient** to detect bias.
-   **Visualization**: Generates ROC curves and uploads them to GCS.

**Usage:**
```bash
python scripts/evaluate_stratified.py \
  --model_path gs://YOUR_BUCKET/models/hybrid/final_model \
  --data_path gs://YOUR_BUCKET/data/test_set.csv \
  --output_bucket YOUR_BUCKET \
  --run_id run-123
```

### 17. Probability Calibration
The script `scripts/calibrate_model.py` applies post-training calibration (Platt Scaling or Isotonic Regression) to ensure predicted probabilities match true frequencies.
It generates Reliability Diagrams and logs Brier Scores per pathology.

**Usage:**
```bash
python scripts/calibrate_model.py \
  --model_path gs://YOUR_BUCKET/models/hybrid/final_model \
  --data_path gs://YOUR_BUCKET/data/val_set.csv \
  --output_bucket YOUR_BUCKET \
  --run_id run-123 \
  --method isotonic
```

**Artifacts:**
-   `calibrators.pkl`: Serialized sklearn calibrators (one per class).
-   `calibration_plot.png`: Reliability curves.
-   `brier_scores.json`: Brier scores before and after calibration.

### 18. Comprehensive Vertex AI Endpoint
The `serving/` directory now contains a unified API that provides:
1.  **Diagnosis**: Pathology probabilities.
2.  **Uncertainty**: MC Dropout confidence intervals.
3.  **Quality**: Signal SNR and valid beat percentage.
4.  **Explanation**: Saliency maps (optional via `explain=True`).

**Deployment:**
1.  Build Image:
    ```bash
    docker build -t gcr.io/YOUR_PROJECT/cardio-endpoint -f serving/Dockerfile .
    ```
2.  Push to GCR:
    ```bash
    docker push gcr.io/YOUR_PROJECT/cardio-endpoint
    ```
3.  Deploy to Vertex AI:
    -   Create Model pointing to this container image.
    -   Create Endpoint and deploy model.

**API Usage:**
POST `/predict`
```json
{
  "signal": [[...]], 
  "mc_samples": 20,
  "explain": true
}
```

### 19. Explainable AI (XAI) & Clinical Reporting
The system supports generating **Saliency Maps** (Integrated Gradients) to visualize which parts of the ECG contributed to the diagnosis.

**Components:**
1.  `scripts/upload_xai_model.py`: Uploads the model to Vertex AI with XAI configuration.
2.  `scripts/generate_clinical_report.py`: Generates visual reports with saliency overlays and QRS correlation metrics.

**Usage (Local Generation):**
```bash
python scripts/generate_clinical_report.py \
  --model_path gs://YOUR_BUCKET/models/hybrid/final_model \
  --output_bucket YOUR_BUCKET \
  --run_id run-123
```

**Artifacts:**
-   `xai_reports/run-123/sample_0_AFIB.png`: 12-lead plot with importance heatmap.

### 20. Clinical PDF Reporting
The script `scripts/generate_pdf_report.py` generates a comprehensive, cardiologist-friendly PDF report containing:
-   **Patient Info & Measurements**: HR, PR, QRS, QT/QTc, Axis.
-   **12-Lead ECG**: Standard layout with medical grid.
-   **AI Interpretation**: Diagnosis with confidence intervals and uncertainty.
-   **Visual Explanation**: Saliency map overlay on Lead II.

**Prerequisites:**
`pip install reportlab matplotlib`

**Usage:**
```bash
python scripts/generate_pdf_report.py \
  --record_id "PATIENT_001" \
  --signal_path "data/sample_ecg.npy" \
  --model_path "models/hybrid/final_model" \
  --output_path "reports/PATIENT_001_Report.pdf"
```

### 21. Automated Bias Analysis & Monitoring
The script `scripts/analyze_bias.py` evaluates model fairness across demographic groups (Age, Sex, Ethnicity, Device).
It calculates AUC/F1 per group, logs results to BigQuery, and sends custom metrics to Cloud Monitoring to trigger alerts if disparities exceed a threshold (default 0.1).

**Setup:**
1.  Create table: `bq query --use_legacy_sql=false < sql/bias_schema.sql`
2.  Run Analysis:
    ```bash
    python scripts/analyze_bias.py \
      --project_id YOUR_PROJECT_ID \
      --run_id run-123
    ```

**Monitoring:**
Create an Alert Policy in Cloud Monitoring for `custom.googleapis.com/model_bias/disparity` > 0.1.

### 22. Fairness-Aware Training Loop
The script `scripts/fairness_loop.py` automates the bias mitigation process:
1.  **Analyze**: Detects underperforming demographic groups.
2.  **Reweight**: Calculates inverse-performance weights (e.g., boosting "Female" samples if F1 is low).
3.  **Retrain**: Launches a new training job with `--group_weights`.
4.  **Verify**: Compares performance before and after.

**Usage:**
```bash
python scripts/fairness_loop.py \
  --project_id YOUR_PROJECT_ID \
  --bucket YOUR_BUCKET
```

### 23. Regulatory Compliance Artifacts (Model Card)
The script `scripts/generate_model_card.py` automatically generates a **Model Card** (Markdown) for regulatory submission.
It aggregates:
-   **Performance Metrics** (from GCS).
-   **Bias Analysis** (from BigQuery).
-   **Dataset Inventory** & **Limitations**.

**Usage:**
```bash
python scripts/generate_model_card.py \
  --project_id YOUR_PROJECT_ID \
  --run_id run-123 \
  --output_bucket YOUR_BUCKET
```

**Output:** `MODEL_CARD.md` (Ready for PDF conversion/submission).

### 25. Triage Mode & Webhook Alerts
The endpoint now includes a **Triage Logic** layer that evaluates urgency based on configurable thresholds.
-   **CRITICAL**: MI > 0.6 (Configurable via `THRESH_MI`).
-   **URGENT**: AFIB > 0.8 (Configurable via `THRESH_AFIB`).
-   **ROUTINE**: All else.

**Alerting:**
If a case is Critical or Urgent, the system sends a JSON payload to `ALERT_WEBHOOK_URL` (e.g., PagerDuty, Slack).

**Audit Logging:**
All requests are logged to `stdout` in JSON format for ingestion by Cloud Logging.

**Configuration:**
Set these env vars in your Docker run or Vertex Endpoint:
-   `THRESH_MI=0.6`
-   `THRESH_AFIB=0.8`
-   `ALERT_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK`

### 26. Model Monitoring & Drift Detection
The script `scripts/enable_model_monitoring.py` configures Vertex AI to monitor the endpoint for **Prediction Drift** (e.g., if the model starts predicting "AFIB" much more frequently than usual).

**Setup:**
1.  Ensure Endpoint has logging enabled (deploy with `enable_request_response_logging=True`).
2.  Run Configuration:
    ```bash
    python scripts/enable_model_monitoring.py \
      --project_id YOUR_PROJECT_ID \
      --endpoint_name "cardio-endpoint" \
      --email_alert "admin@example.com"
    ```

**Weekly Reporting:**
Generate a PDF summary of drift and traffic:
```bash
python scripts/generate_weekly_monitoring_report.py \
  --project_id YOUR_PROJECT_ID \
  --endpoint_id "1234567890"
```

### 27. Human Feedback & Incremental Learning
The system implements a **Human-in-the-Loop** pipeline to improve over time.

1.  **Feedback Submission**:
    Cardiologists submit corrections via the `/feedback` endpoint.
    ```bash
    curl -X POST http://localhost:8080/feedback -d '{
      "prediction_id": "pred-123",
      "predicted_label": "Normal",
      "corrected_label": "AFIB",
      "cardiologist_id": "DR_SMITH"
    }'
    ```

2.  **Validation (Senior Review)**:
    A senior cardiologist reviews pending feedback using the CLI tool:
    ```bash
    python scripts/validate_feedback.py --project_id YOUR_PROJECT_ID
    ```

3.  **Automated Retraining**:
    A scheduled script checks if enough approved feedback exists (>100 samples) to trigger a new training job:
    ```bash
    python scripts/trigger_retraining.py --project_id YOUR_PROJECT_ID
    ```

### 28. Governance & Compliance
A comprehensive governance framework has been established in the `governance/` directory.

-   **Policies**:
    -   `governance/DATA_GOVERNANCE.md`: Data classification, lineage, and retention.
    -   `governance/MODEL_GOVERNANCE.md`: Lifecycle, versioning, and retirement.
    -   `governance/CLINICAL_ACCEPTANCE_CRITERIA.md`: Strict performance thresholds for deployment.

-   **Templates**:
    -   `governance/templates/CLINICAL_REVIEW_TEMPLATE.md`: Checklist for cardiologist sign-off.
    -   `governance/templates/SECURITY_CHECKLIST.md`: Security pre-deployment verification.

**Usage:**
Before deploying a new model version, complete the **Clinical Review** and **Security Checklist** and store them in the project documentation (e.g., `docs/reviews/v1.2.0/`).

### 29. CI/CD Pipeline (Cloud Build)
The project includes a `cloudbuild.yaml` configuration for automated testing and deployment.

**Test Suite:**
-   **Unit Tests** (`tests/unit`): Validates preprocessing, artifact detection, and logic for short/noisy signals.
-   **Integration Tests** (`tests/integration`): Validates API response structure, triage logic, and error handling.
-   **Regression Tests** (`tests/regression`): (Optional) Validates model performance metrics against a golden dataset.

**Usage:**
Trigger a build manually:
```bash
gcloud builds submit --config cloudbuild.yaml .
```

**Pipeline Steps:**
1.  Install Dependencies.
2.  Run Unit Tests.
3.  Run Integration Tests.
4.  Build Docker Image.
5.  Push to Container Registry.

### 30. Automated PR Evaluation (Clinical Regression Check)
To ensure no code change negatively impacts patient safety, a dedicated Cloud Build pipeline (`cloudbuild-pr.yaml`) runs on every Pull Request.

**Mechanism:**
1.  **Baseline Definition**: `tests/baseline_metrics.json` defines the minimum acceptable Sensitivity/Specificity for MI and AFIB.
2.  **Evaluation Script**: `scripts/evaluate_pr.py` runs inference on a validation subset and compares current metrics against the baseline.
3.  **Blocking**: If **MI Sensitivity** drops below the baseline (e.g., 0.90), the build fails, preventing the merge.

**Usage:**
Configure your GitHub/Bitbucket trigger in Cloud Build to use `cloudbuild-pr.yaml` for Pull Request events.

### 31. External Dataset Evaluation
The script `scripts/evaluate_external_datasets.py` allows for rigorous validation against multiple external datasets stored in GCS.
It calculates Sensitivity, Specificity, and AUC for each pathology and generates ROC curves.

**Usage:**
```bash
python scripts/evaluate_external_datasets.py \
  --project_id YOUR_PROJECT_ID \
  --model_path "models/hybrid/final_model" \
  --dataset_uris "gs://bucket/data/external_1.npz,gs://bucket/data/external_2.npz,gs://bucket/data/external_3.npz" \
  --output_bucket YOUR_OUTPUT_BUCKET
```

**Outputs (in Output Bucket):**
-   `evaluation/external/report.json`: Aggregate metrics.
-   `evaluation/external/{dataset_name}/roc_curve.png`: ROC plots.

### 32. Post-Training Calibration & Uncertainty
The script `scripts/calibrate_model.py` applies **Isotonic Regression** or **Platt Scaling** to calibrate probabilities.
It also supports **MC Dropout** to estimate uncertainty (Confidence Intervals) during the calibration phase.

**Usage:**
```bash
python scripts/calibrate_model.py \
  --project_id YOUR_PROJECT_ID \
  --model_path gs://YOUR_BUCKET/models/hybrid/final_model \
  --data_path gs://YOUR_BUCKET/data/val_set.csv \
  --output_bucket YOUR_BUCKET \
  --run_id run-123 \
  --method isotonic \
  --mc_samples 20
```

**Artifacts:**
-   `calibrators.pkl`: Serialized sklearn calibrators.
-   `calibration_plot.png`: Reliability curves.
-   `brier_scores.json`: Brier scores and average CI widths.

### 33. Active Learning & Data Labeling Pipeline
This pipeline automates the selection of hard/rare samples for human labeling and consolidates the results.

1.  **Export for Labeling**:
    Selects samples from BigQuery (prioritizing rare classes like MI/PVC and low-confidence predictions) and generates a JSONL file for Vertex AI Data Labeling.
    ```bash
    python scripts/export_for_labeling.py \
      --project_id YOUR_PROJECT_ID \
      --source_table "cardio_analytics.predictions" \
      --output_bucket YOUR_BUCKET
    ```

2.  **Consolidate Labels (Cloud Function)**:
    After labeling is complete, this function processes the export, applies a **Consensus Rule** (e.g., 60% agreement), and saves high-quality labels to BigQuery.
    
    **Deploy Function:**
    ```bash
    gcloud functions deploy consolidate_labels \
      --runtime python39 \
      --trigger-http \
      --source functions/consolidate_labels \
      --set-env-vars CONSENSUS_THRESHOLD=0.6,MIN_ANNOTATORS=3,DESTINATION_TABLE=cardio_analytics.gold_standard_labels
    ```
    
    **Trigger:**
    ```bash
    curl -X POST https://REGION-PROJECT.cloudfunctions.net/consolidate_labels \
      -H "Content-Type: application/json" \
      -d '{"gcs_uri": "gs://YOUR_BUCKET/labeling/exports/export-123.jsonl"}'
    ```

### 34. Robustness Stress Testing
The script `scripts/stress_test_model.py` evaluates model performance under adverse conditions to ensure clinical safety.

**Scenarios Tested:**
1.  **Noise Injection**: Gaussian noise at SNR 24dB and 12dB.
2.  **Lead Loss**: Missing Lead II (critical for rhythm) or Precordial leads (V1-V6).
3.  **Sampling Variation**: Downsampling to 250Hz (simulating older Holter monitors).

**Usage:**
```bash
python scripts/stress_test_model.py \
  --project_id YOUR_PROJECT_ID \
  --model_path gs://YOUR_BUCKET/models/hybrid/final_model \
  --data_path gs://YOUR_BUCKET/data/val_set.csv \
  --output_bucket YOUR_BUCKET
```

**Output:**
-   `stress_test_report.json`: Metrics per scenario and "Robustness" pass/fail status (Drop < 10%).

### 35. Explainable AI (XAI) Clinical Reports
The script `scripts/generate_xai_report.py` generates a PDF report for a specific patient case, visualizing:
1.  **Saliency Map Overlay**: Highlights the exact segments (e.g., P-wave, ST-segment) that drove the model's decision.
2.  **Feature Correlation**: Automatically detects if the model is focusing on QRS complexes vs. diffuse rhythm irregularities.
3.  **Lead Importance**: Bar chart showing which of the 12 leads contributed most.

**Usage:**
```bash
python scripts/generate_xai_report.py \
  --project_id YOUR_PROJECT_ID \
  --model_path gs://YOUR_BUCKET/models/hybrid/final_model \
  --data_path gs://YOUR_BUCKET/data/val_set.csv \
  --output_bucket YOUR_BUCKET \
  --patient_id "PATIENT-123"
```

**Output:**
-   `reports/{patient_id}/xai_report_{patient_id}.pdf`: A clinician-friendly PDF summary.

### 36. Continuous Sensitivity Monitoring
To detect clinical regression in production, we monitor the **Sensitivity Drop** metric.
This requires the Feedback Loop to be active (populating `cardio_analytics.feedback`).

1.  **Compute Metrics**:
    Run this script daily (e.g., via Cloud Scheduler) to calculate sensitivity from approved feedback and publish to Cloud Monitoring.
    ```bash
    python scripts/compute_and_publish_metrics.py --project_id YOUR_PROJECT_ID
    ```

2.  **Configure Alerts**:
    One-time setup to create an Alert Policy that triggers if **Sensitivity drops > 5%** week-over-week.
    ```bash
    python scripts/create_sensitivity_alert.py \
      --project_id YOUR_PROJECT_ID \
      --email "clinical-ops@example.com"
    ```

## Environment Variables

Add these to your `.env` file:

```env
GOOGLE_APPLICATION_CREDENTIALS="./path/to/service-account-key.json"
GCS_BUCKET_NAME="cardio-ai-artifacts"
BQ_DATASET="cardio_analytics"
BQ_TABLE="predictions"
```

## Privacy & Security

-   **De-identification**: Patient IDs are hashed before storage in BigQuery.
-   **Access Control**: Use IAM roles to restrict access to the GCS bucket and BigQuery dataset.
-   **Encryption**: Data is encrypted at rest and in transit by default on GCP.
