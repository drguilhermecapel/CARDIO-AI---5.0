# Data Labeling & Consensus Workflow

This document outlines the process for integrating human review (Cardiologists) into the Cardio AI loop.

## Architecture

1.  **Labeling Service**: Use Vertex AI Data Labeling to create tasks for cardiologists.
2.  **Export**: When a labeling job is complete, export the results (JSONL) to the GCS bucket `cardio-ai-labels-export`.
3.  **Ingestion**: A Cloud Function (`functions/process_labeling_export`) triggers on the file upload.
4.  **Consensus**: The function calculates the "Golden Label" based on majority vote and updates BigQuery.

## Setup

### Automated Deployment (Recommended)

Run the provided shell script to create the BigQuery dataset, tables, GCS bucket, and deploy the Cloud Function:

```bash
chmod +x scripts/deploy_labeling.sh
./scripts/deploy_labeling.sh
```

### Manual Setup

#### 1. BigQuery Schema
Run the SQL script to create the necessary tables:

```bash
bq query --use_legacy_sql=false < sql/labeling_schema.sql
```

### 2. Deploy Cloud Function
Deploy the consensus processor:

```bash
cd functions/process_labeling_export
gcloud functions deploy process_labeling_export \
    --runtime python39 \
    --trigger-resource cardio-ai-labels-export \
    --trigger-event google.storage.object.finalize \
    --entry-point process_labeling_export \
    --set-env-vars PROJECT_ID=your-project-id
```

## Consensus Logic

The consensus engine uses a **Majority Vote** strategy with a configurable threshold (default 66%).

-   **FINAL**: Agreement Score >= 66% (e.g., 2/3 or 3/4 reviewers agree).
-   **NEEDS_SUPER_REVIEW**: Agreement Score < 66%. These records are flagged for a senior cardiologist (Super Reviewer) to resolve.

## Data Flow

1.  **Raw Reviews**: Stored in `cardio_analytics.ecg_reviews`. Contains every individual vote.
2.  **Golden Labels**: Stored in `cardio_analytics.golden_labels`. Contains the final, approved label for training.

## Future Improvements

-   **Weighted Consensus**: Assign higher weights to senior cardiologists in the `agreement_score` calculation.
-   **Active Learning**: Automatically send low-confidence model predictions to the Labeling Service.
