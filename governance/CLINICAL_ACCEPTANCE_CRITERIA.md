# Clinical Acceptance Criteria

Before a model version is promoted to **Production**, it must meet the following strict criteria.

## 1. Performance Thresholds (Test Set)

| Pathology | Minimum Sensitivity (Recall) | Minimum Specificity | Target AUC |
| :--- | :--- | :--- | :--- |
| **Myocardial Infarction (MI)** | **90%** (Critical) | 85% | > 0.92 |
| **Atrial Fibrillation (AFIB)** | **95%** (Urgent) | 90% | > 0.96 |
| **PVC** | 80% | 85% | > 0.88 |
| **Normal** | 90% | 80% | > 0.90 |

*Note: Failure to meet Critical thresholds results in automatic rejection.*

## 2. Fairness & Bias
*   **Maximum Disparity**: The difference in F1-score between any two demographic groups (e.g., Male vs. Female, <65 vs >65) must not exceed **0.05 (5%)**.
*   **Representation**: Test set must include at least 100 samples from each protected group.

## 3. Calibration
*   **Brier Score**: Must be < 0.15 for all classes.
*   **Reliability Curve**: Visual inspection must show alignment within 10% of the diagonal.

## 4. Clinical Validation
*   **Blind Review**: A random sample of 50 disagreements (Model vs. Ground Truth) must be reviewed by a senior cardiologist.
*   **Safety Check**: Zero "Critical Misses" (e.g., classifying a massive STEMI as Normal) allowed in the validation set.

## 5. Operational Metrics
*   **Latency**: 99th percentile (P99) inference time < 500ms.
*   **Throughput**: Capable of handling 100 requests/second.
