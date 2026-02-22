# Risk Management Matrix (ISO 14971 / IEC 62304)

**Device Class:** C (Critical - Life Support/Diagnostic)
**Date:** 2026-02-22

## Risk Acceptance Criteria
- **Unacceptable:** Severity 4-5 & Probability > 2
- **ALARP (As Low As Reasonably Practicable):** Severity 3-4 & Probability 1-2
- **Acceptable:** Severity 1-2

## Risk Register

| ID | Hazard Category | Hazard Description | Cause | Initial Risk (S x P) | Mitigation Strategy | Residual Risk (S x P) |
|---|---|---|---|---|---|---|
| **R01** | Clinical | **Missed STEMI (False Negative)** | Algorithm fails to detect subtle ST elevation. | 5 x 3 = 15 (High) | 1. Ensemble voting (ViT + Rule-based).<br>2. Lower threshold for "Indeterminate" alert.<br>3. Human review required for all negatives in ER. | 5 x 1 = 5 (Med) |
| **R02** | Clinical | **False Positive AFib** | Noise interpreted as irregular rhythm. | 3 x 4 = 12 (High) | 1. Signal Quality Index (SQI) check.<br>2. P-wave subtraction analysis.<br>3. Context-aware filtering (previous history). | 3 x 2 = 6 (Low) |
| **R03** | Clinical | **Incorrect QT Measurement** | T-wave end misdetection (flat T-wave). | 4 x 3 = 12 (High) | 1. Multi-lead T-wave delineation.<br>2. Use median beat analysis.<br>3. Report confidence interval for QTc. | 4 x 1 = 4 (Low) |
| **R04** | Technical | **Lead Reversal** | Electrodes placed incorrectly (LA/RA swap). | 3 x 3 = 9 (Med) | 1. Automated Lead Reversal Detection (Axis check).<br>2. Alert user to re-acquire. | 3 x 1 = 3 (Low) |
| **R05** | Technical | **Latency > 5s** | Server overload or network latency. | 2 x 3 = 6 (Med) | 1. Edge computing fallback.<br>2. Auto-scaling K8s cluster.<br>3. Async processing for non-stat cases. | 2 x 1 = 2 (Low) |
| **R06** | Cybersecurity | **PHI Data Leak** | Unencrypted storage or transmission. | 5 x 2 = 10 (High) | 1. AES-256 at rest, TLS 1.3 in transit.<br>2. De-identification before cloud upload.<br>3. Regular Pen-testing. | 5 x 1 = 5 (Med) |
| **R07** | Cybersecurity | **Adversarial Attack** | Malicious noise injected to alter diagnosis. | 4 x 1 = 4 (Low) | 1. Input sanitization.<br>2. Adversarial training.<br>3. Digital signature on raw data. | 4 x 1 = 4 (Low) |
| **R08** | Usability | **Alert Fatigue** | Too many low-confidence alerts. | 2 x 5 = 10 (High) | 1. Configurable alert thresholds.<br>2. "Smart Alerts" grouping similar findings. | 2 x 2 = 4 (Low) |
| **R09** | Data | **Training Data Bias** | Model trained mostly on male/caucasian data. | 4 x 4 = 16 (High) | 1. Stratified validation (Sex, Age, Ethnicity).<br>2. Bias correction loss functions.<br>3. Continuous monitoring of sub-group performance. | 4 x 2 = 8 (Med) |
| **R10** | System | **Database Corruption** | Hardware failure or software bug. | 4 x 2 = 8 (Med) | 1. ACID compliant DB (Postgres).<br>2. Point-in-time recovery backups.<br>3. Immutable audit logs. | 4 x 1 = 4 (Low) |
| **R11** | Clinical | **Pacemaker Spike Misinterpretation** | Spikes seen as QRS or noise. | 3 x 3 = 9 (Med) | 1. High-frequency spike detection module.<br>2. "Paced Rhythm" mode. | 3 x 1 = 3 (Low) |
| **R12** | Clinical | **Pediatric Misdiagnosis** | Adult criteria applied to children. | 4 x 3 = 12 (High) | 1. Age input mandatory.<br>2. Pediatric-specific reference ranges (Davignon). | 4 x 1 = 4 (Low) |
| **R13** | Technical | **Downtime during update** | Service unavailable during critical care. | 4 x 2 = 8 (Med) | 1. Blue/Green deployment.<br>2. Canary releases.<br>3. Offline mode in app. | 4 x 1 = 4 (Low) |
| **R14** | Regulatory | **Audit Trail Gaps** | Actions not logged for compliance. | 3 x 2 = 6 (Med) | 1. Blockchain-like immutable log.<br>2. Centralized logging (ELK stack). | 3 x 1 = 3 (Low) |
| **R15** | Clinical | **Hyperkalemia Missed** | Peaked T-waves ignored. | 5 x 2 = 10 (High) | 1. Specific T-wave slope analysis.<br>2. Electrolyte imbalance module. | 5 x 1 = 5 (Med) |
| **R16** | Technical | **Mobile App Crash** | Memory leak or unhandled exception. | 2 x 3 = 6 (Med) | 1. Robust error handling.<br>2. Crashlytics monitoring.<br>3. Automated UI testing. | 2 x 1 = 2 (Low) |
| **R17** | Clinical | **Brugada Pattern Missed** | Subtle Type 2 pattern ignored. | 4 x 2 = 8 (Med) | 1. Specialized Brugada detector (V1/V2 high leads).<br>2. Provocation test protocol support. | 4 x 1 = 4 (Low) |
| **R18** | System | **API Rate Limit Exceeded** | Hospital floods server during mass casualty. | 3 x 2 = 6 (Med) | 1. Priority queueing for "STAT" ECGs.<br>2. Dynamic rate limiting. | 3 x 1 = 3 (Low) |
| **R19** | Data | **Label Noise in Training** | Ground truth errors in training set. | 3 x 4 = 12 (High) | 1. Multi-cardiologist consensus for Gold Standard.<br>2. Label smoothing. | 3 x 2 = 6 (Low) |
| **R20** | Technical | **Integration Failure (EHR)** | HL7 message format mismatch. | 2 x 4 = 8 (Med) | 1. Mirth Connect integration engine.<br>2. Strict FHIR validation. | 2 x 1 = 2 (Low) |

## Severity Scale
1. Negligible
2. Minor
3. Serious
4. Critical
5. Catastrophic

## Probability Scale
1. Improbable (< 10^-6)
2. Remote (10^-6 - 10^-4)
3. Occasional (10^-4 - 10^-2)
4. Probable (10^-2 - 10^-1)
5. Frequent (> 10^-1)
