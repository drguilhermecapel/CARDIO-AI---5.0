# NMPA (China) Regulatory Strategy
**Device Name:** CardioAI Nexus
**Classification:** Class II or III (Depending on specific claims; Automated diagnostic software is often Class III if it provides definitive diagnosis, or Class II if it's an aid. We target Class II as a diagnostic aid).
**Regulatory Body:** National Medical Products Administration (NMPA)

## 1. Local Registration Requirements
*   **Legal Agent:** A China-based legal agent is required for foreign manufacturers.
*   **Local Testing (Type Testing):** Software must undergo testing at an NMPA-certified testing center (e.g., CMTC).
*   **Chinese Standards (GB/YY):** Must comply with relevant Chinese standards (e.g., GB/T 25000.51 for software engineering).

## 2. Software Registration Guidelines
NMPA has specific guidelines for AI medical software:
*   **"Guiding Principles for Registration Review of Artificial Intelligence Medical Device Software"**
*   **Data Quality:** Detailed documentation of the training, validation, and testing datasets.
    *   Data sources, collection methods, inclusion/exclusion criteria.
    *   Annotation rules and quality control (e.g., consensus of multiple senior doctors).
    *   Dataset distribution (age, gender, disease prevalence, equipment models).
*   **Algorithm Design:** Architecture, hyperparameters, loss functions.
*   **Algorithm Verification & Validation:**
    *   Performance metrics (Sensitivity, Specificity, ROC, AUC).
    *   Stress testing, adversarial testing, robustness against noise.
*   **Explainability:** How the algorithm reaches its conclusions (e.g., highlighting critical ECG segments).

## 3. Clinical Evaluation
*   **Clinical Trial Exemption:** If the device is substantially equivalent to a device on the exemption list, a trial may be avoided.
*   **Clinical Trial (if required):** Must be conducted in China, following China GCP. Multi-center study comparing AI performance against gold standard (expert consensus or angiography for ischemia).
*   **Real-World Data (RWD):** NMPA is increasingly accepting RWD, but it requires strict methodological rigor.

## 4. Cybersecurity and Data Privacy
*   **"Guiding Principles for Cybersecurity Registration Review of Medical Devices"**
*   Must comply with China's Data Security Law (DSL) and Personal Information Protection Law (PIPL).
*   **Data Localization:** Patient data collected in China must generally be stored in China. Cross-border data transfer requires security assessments.

## 5. Submission Dossier (eRPS)
*   Application Form.
*   Product Technical Requirement (PTR) - crucial document defining testable specs.
*   Type Test Report.
*   Clinical Evaluation Report (CER) or Clinical Trial Report.
*   Software Description Report.
*   Cybersecurity Description Report.
*   Labeling and IFU (in Simplified Chinese).
*   Proof of Home Country Approval (e.g., FDA 510k or CE Mark) is generally required for imported devices.

---
**Action Plan:**
1.  Appoint Chinese Legal Agent.
2.  Draft Product Technical Requirement (PTR).
3.  Conduct Type Testing at NMPA center.
4.  Determine Clinical Evaluation Route (Trial vs. Exemption/CER).
5.  Prepare AI-specific documentation (Data & Algorithm reports).
6.  Submit via eRPS.
