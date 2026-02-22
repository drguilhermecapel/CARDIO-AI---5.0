# Risk Management Plan (ISO 14971:2019 Compliance)

## 1. Scope
This document outlines the risk management process for the CardioAI Nexus ECG Diagnostic System, covering its entire lifecycle from design to decommissioning.

## 2. Risk Analysis (FMEA - Failure Mode and Effects Analysis)

| ID | Failure Mode | Potential Effect | Severity (S) | Probability (P) | Risk Priority Number (RPN) | Mitigation Strategy | Verification |
|---|---|---|---|---|---|---|---|
| F01 | False Negative STEMI Detection | Missed critical diagnosis, delayed treatment, potential death. | 5 (Catastrophic) | 2 (Remote) | 10 (High) | 1. Ensemble voting (ViT + specialized heads).<br>2. Threshold tuning for high sensitivity (>98%).<br>3. Human-in-the-loop for borderline cases. | Validation on 4+ external datasets (PTB-XL, CODE). |
| F02 | False Positive AFib Alert | Alarm fatigue, unnecessary anticoagulation, anxiety. | 3 (Serious) | 3 (Occasional) | 9 (Medium) | 1. Specificity optimization (>95%).<br>2. Context-aware filtering (e.g., patient history). | Clinical trial simulation. |
| F03 | Lead Reversal Misinterpretation | Incorrect axis/morphology analysis. | 4 (Critical) | 3 (Occasional) | 12 (High) | 1. Dedicated Lead Reversal Detection Module.<br>2. Auto-correction or rejection of bad quality signals. | Unit tests with simulated lead reversals. |
| F04 | Data Privacy Breach (PHI Leak) | Legal penalties, loss of trust. | 5 (Catastrophic) | 1 (Improbable) | 5 (Medium) | 1. End-to-end encryption (TLS 1.3).<br>2. De-identification at edge.<br>3. SOC 2 Type II compliance. | Penetration testing (annual). |
| F05 | Model Drift / Performance Degradation | Reduced diagnostic accuracy over time. | 4 (Critical) | 2 (Remote) | 8 (Medium) | 1. Continuous monitoring dashboard.<br>2. Automated retraining pipeline.<br>3. Version control with rollback. | Monthly performance audit. |

## 3. Risk Evaluation Matrix

| Severity / Probability | Frequent (5) | Probable (4) | Occasional (3) | Remote (2) | Improbable (1) |
|---|---|---|---|---|---|
| **Catastrophic (5)** | Unacceptable | Unacceptable | Unacceptable | High | Medium |
| **Critical (4)** | Unacceptable | Unacceptable | High | Medium | Low |
| **Serious (3)** | Unacceptable | High | Medium | Low | Low |
| **Minor (2)** | High | Medium | Low | Low | Negligible |
| **Negligible (1)** | Medium | Low | Low | Negligible | Negligible |

## 4. Risk Control Measures
- **Design Controls:** Redundant algorithms, fail-safe defaults.
- **Protective Measures:** Alarms, interlocks (e.g., prevent analysis on poor signal).
- **Information for Safety:** User manual warnings, training.

## 5. Post-Market Surveillance (PMS)
- Active monitoring of adverse events.
- Feedback loop from partner hospitals.
- Periodic Safety Update Reports (PSUR).
