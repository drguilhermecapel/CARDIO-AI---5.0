# Model Governance Policy

## 1. Model Lifecycle
The lifecycle of a model consists of four stages:

1.  **Development (Experimental)**:
    *   Tracked in Vertex AI Experiments.
    *   No strict performance thresholds.
    *   Code in feature branches.

2.  **Staging (Candidate)**:
    *   Model exported to Vertex AI Model Registry.
    *   Must pass **Automated Evaluation Pipeline** (Unit tests, Integration tests).
    *   Must pass **Bias Analysis**.

3.  **Production (Active)**:
    *   Deployed to Vertex AI Endpoint.
    *   Subject to **Clinical Acceptance Review**.
    *   Monitored for drift and performance degradation.

4.  **Archived (Retired)**:
    *   Removed from serving.
    *   Artifacts retained in GCS for audit purposes for 2 years.

## 2. Versioning Strategy
Models follow Semantic Versioning (`vX.Y.Z`):
*   **X (Major)**: Change in architecture or input data schema (requires client update).
*   **Y (Minor)**: Retraining with new data or hyperparameter tuning (performance change).
*   **Z (Patch)**: Hotfix for serving infrastructure or post-processing logic.

## 3. Reproducibility
Every production model must be reproducible. The following artifacts are required:
*   **Code**: Git commit hash.
*   **Data**: Dataset version URI.
*   **Environment**: Docker container digest.
*   **Hyperparameters**: Config file or Vertex AI params.

## 4. Retirement Policy
A model is retired when:
*   A new version significantly outperforms it (statistically significant improvement).
*   Drift detection indicates unrecoverable degradation.
*   Underlying data schema changes.
