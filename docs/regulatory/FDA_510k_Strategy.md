# FDA 510(k) Premarket Notification Strategy
**Device Name:** CardioAI Nexus
**Regulation Number:** 21 CFR 870.2340 (Electrocardiograph)
**Product Code:** DPS (Electrocardiograph), DXH (Automated ECG Analysis)
**Class:** II

## 1. Predicate Device Selection
To demonstrate Substantial Equivalence (SE), we select the following predicates:
*   **Primary Predicate:** GE MUSE v9 (K183296) - Automated ECG Analysis.
*   **Secondary Predicate:** Philips IntelliSpace ECG (K123456).

## 2. Intended Use
CardioAI Nexus is a software-only device intended to analyze 12-lead ECG data to detect rhythms, intervals, and morphological abnormalities. It is intended for use by qualified medical professionals as a diagnostic aid. It is **not** intended to be the sole means of diagnosis.

## 3. Indications for Use
The device provides analysis for:
*   Adult patients (>18 years).
*   Detection of Sinus Rhythm, Atrial Fibrillation, Atrial Flutter, Ventricular Tachycardia, AV Blocks.
*   Detection of ST-segment changes indicative of ischemia (STEMI/NSTEMI).
*   Measurement of PR, QRS, QT, QTc intervals.

## 4. Software Documentation (Guidance for Industry)
**Level of Concern:** Moderate (Failure could lead to minor injury or delayed treatment).

### 4.1 Deliverables
*   **SRS:** See `docs/Software_Requirements_Spec.md`.
*   **SDS:** Software Design Specification (Architecture, Algorithms).
*   **Risk Management:** ISO 14971 Hazard Analysis.
*   **V&V:** Validation Reports (Sensitivity/Specificity >95% vs Gold Standard).
*   **Cybersecurity:** Threat Modeling, SBOM, Penetration Testing Report.

## 5. Performance Testing (Bench)
*   **ECG Database Testing:** Testing against MIT-BIH, AHA, CSE, and PTB-XL databases.
*   **Metrics:**
    *   QRS Detection Sensitivity: >99%
    *   AFib Sensitivity: >96%
    *   STEMI Sensitivity: >95% (as per user requirement)
*   **Clinical Study:** A retrospective study comparing CardioAI output vs. Consensus of 3 Cardiologists on 1,000 diverse ECGs.

## 6. Labeling
*   User Manual must state: "Computerized interpretation is a valuable tool but must be overread by a physician."
*   Contraindications: Not for pediatric use (unless validated). Not for critical care monitoring (alarms).

---
**Submission Timeline:**
1.  Pre-Submission Meeting (Q-Sub): Month 1
2.  Design Freeze: Month 3
3.  V&V Completion: Month 5
4.  510(k) Submission: Month 6
5.  FDA Review (90 days): Month 9
