#!/bin/bash
set -e

# Configuration
# Override these with environment variables if needed
PROJECT_ID=${PROJECT_ID:-$(gcloud config get-value project)}
REGION=${REGION:-"us-central1"}
BUCKET_NAME=${BUCKET_NAME:-"cardio-ai-labels-export-${PROJECT_ID}"}
DATASET_ID="cardio_analytics"

echo "🚀 Deploying Labeling Infrastructure for Project: ${PROJECT_ID}"
echo "   Region: ${REGION}"
echo "   Bucket: ${BUCKET_NAME}"

# 1. Create BigQuery Dataset
echo -e "\n[1/4] Creating BigQuery Dataset '${DATASET_ID}'..."
if bq ls --project_id=${PROJECT_ID} | grep -q ${DATASET_ID}; then
    echo "Dataset already exists."
else
    bq --location=US mk -d --description "Cardio AI Analytics" ${PROJECT_ID}:${DATASET_ID}
    echo "Dataset created."
fi

# 2. Create Tables
echo -e "\n[2/4] Creating BigQuery Tables..."
# We use sed to replace the dataset placeholder if necessary, but the SQL file uses 'cardio_analytics' directly.
# Ensure we run this from the project root
bq query --project_id=${PROJECT_ID} --use_legacy_sql=false < sql/labeling_schema.sql
echo "Tables created/verified."

# 3. Create GCS Bucket for Exports
echo -e "\n[3/4] Creating GCS Bucket 'gs://${BUCKET_NAME}'..."
if gcloud storage buckets list gs://${BUCKET_NAME} --format="value(name)" > /dev/null 2>&1; then
    echo "Bucket already exists."
else
    gcloud storage buckets create gs://${BUCKET_NAME} --location=${REGION}
    echo "Bucket created."
fi

# 4. Deploy Cloud Function
echo -e "\n[4/4] Deploying Cloud Function 'process_labeling_export'..."
# Navigate to function directory
cd functions/process_labeling_export

gcloud functions deploy process_labeling_export \
    --gen2 \
    --runtime=python39 \
    --region=${REGION} \
    --source=. \
    --entry-point=process_labeling_export \
    --trigger-bucket=${BUCKET_NAME} \
    --set-env-vars PROJECT_ID=${PROJECT_ID},DATASET_ID=${DATASET_ID} \
    --quiet

echo -e "\n✅ Deployment Complete!"
echo "   - Upload JSONL export files to: gs://${BUCKET_NAME}/"
echo "   - Check logs: gcloud functions logs read process_labeling_export"
