# Data Governance Policy

## 1. Data Classification & Handling
All data used in the Cardio AI project is classified according to the following levels:

*   **Level 1: Public**: Aggregated metrics, published research. No restrictions.
*   **Level 2: Internal**: De-identified ECG signals, model weights, anonymized metadata. Accessible to engineering team.
*   **Level 3: Restricted (PHI)**: Raw patient records containing names, MRNs, or dates. **Strictly prohibited** in this repository and standard training pipelines. Must remain in secure, HIPAA-compliant storage (e.g., specific GCS buckets with limited IAM).

## 2. Data Lineage & Versioning
*   **Raw Data**: Immutable. Stored in GCS with object versioning enabled.
*   **Processed Data**: Versioned using Vertex AI Managed Datasets. Every training job must reference a specific dataset version ID.
*   **Traceability**: All models must link back to the specific dataset version used for training via Vertex AI Experiments metadata.

## 3. Access Control (RBAC)
*   **Data Scientists**: Read-only access to Level 2 data. Write access to Feature Store.
*   **Clinical Reviewers**: Read-only access to validation sets and inference results.
*   **Service Accounts**: Minimal privilege. Training jobs access data via specific service accounts, not user credentials.

## 4. Data Retention & Deletion
*   **Feedback Data**: Retained for 5 years to support longitudinal model improvement.
*   **Temporary Artifacts**: Scratch data in GCS (e.g., staging files) deleted after 30 days via Lifecycle Policies.
*   **Right to be Forgotten**: Upon patient request, all data associated with a specific Patient ID must be purged from BigQuery and GCS within 72 hours. Models trained on this data are not immediately retracted but must exclude this data in the next retraining cycle.
