# ANVISA RDC 665/2022 - Good Manufacturing Practices (GMP) Strategy
**Device Name:** CardioAI Nexus
**Classification:** Risk Class II (Medium Risk)
**Regulation:** RDC 665/2022 (formerly RDC 16/2013) & RDC 751/2022 (Registration)

## 1. Quality Management System (QMS)
To comply with RDC 665, the following QMS processes must be established:

### 1.1 Design Controls (Controle de Projeto)
*   **Design Input:** SRS (`docs/Software_Requirements_Spec.md`) serves as the primary input.
*   **Design Output:** Source code, architecture diagrams, and build artifacts.
*   **Design Review:** Formal reviews at key milestones (Alpha, Beta, RC).
*   **Design Verification:** Unit tests, integration tests, and static analysis.
*   **Design Validation:** Clinical validation study (see `docs/Clinical_Validation_Plan.md`).
*   **Design Transfer:** Release management process (CI/CD pipeline).

### 1.2 Risk Management (Gerenciamento de Risco)
*   Must follow **ISO 14971:2019**.
*   **Hazard Analysis:** Identify potential hazards (e.g., misdiagnosis, delayed treatment).
*   **Risk Control:** Implement mitigations (e.g., "Human in the loop" requirement, confidence scores).
*   **Benefit-Risk Analysis:** Demonstrate that clinical benefits outweigh residual risks.

### 1.3 Software Lifecycle (Ciclo de Vida de Software)
*   Must follow **IEC 62304**.
*   **Configuration Management:** Git-based version control with strict branching strategy.
*   **Problem Resolution:** Issue tracking (Jira/GitHub Issues) for bugs and anomalies.

## 2. Technical File (Dossiê Técnico)
Required for RDC 751/2022 submission:
1.  **Device Description:** Intended use, principles of operation (AI/Deep Learning).
2.  **Labeling:** User manual in Portuguese (PT-BR).
3.  **Clinical Evaluation:** Literature review + Validation study results.
4.  **Risk Management File:** Risk Management Plan and Report.
5.  **Software Validation Report:** Summary of V&V activities.

## 3. LGPD Compliance (Data Privacy)
*   **Data Minimization:** Only process necessary patient data.
*   **Anonymization:** De-identify data for training/improvement.
*   **Consent:** Explicit consent required for data usage if not anonymized.
*   **Security:** Encryption at rest and in transit (AES-256, TLS 1.3).

## 4. Post-Market Surveillance (Tecnovigilância)
*   **Complaint Handling:** Process for receiving and investigating user complaints.
*   **Adverse Event Reporting:** Mandatory reporting to ANVISA (Notivisa) for serious adverse events.
*   **Field Safety Corrective Actions (FSCA):** Procedures for recalls or software patches.

---
**Action Plan:**
1.  Translate User Manual and Labeling to Portuguese.
2.  Appoint a Brazil Registration Holder (BRH) if the manufacturer is foreign.
3.  Conduct internal audit against RDC 665 checklist.
4.  Submit technical dossier via Solicita system.
