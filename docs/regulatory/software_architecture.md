# Software Architecture Specification (IEC 62304 Class C)

## 1. System Overview
CardioAI Nexus is a cloud-native, microservices-based platform for ECG analysis.

## 2. Architectural Components

### 2.1. Frontend (Client)
- **Technology:** React, TypeScript, Tailwind CSS.
- **Responsibility:** Image capture, UI/UX, Report visualization.
- **Safety:** Input validation, secure communication (HTTPS).

### 2.2. API Gateway (Ingress)
- **Technology:** Nginx / Traefik.
- **Responsibility:** Load balancing, SSL termination, Rate limiting.

### 2.3. Inference Engine (Backend)
- **Technology:** Python (FastAPI), PyTorch/TensorFlow.
- **Model:** Ensemble of Vision Transformers (ViT) + CNNs.
- **Responsibility:** Preprocessing, Inference, Post-processing logic.

### 2.4. Data Storage
- **Operational DB:** PostgreSQL (Patient metadata, Reports).
- **Audit Log:** Immutable Ledger (Blockchain-like or Append-only DB).
- **Object Storage:** S3-compatible (Raw ECG images, Anonymized).

## 3. Data Flow
1. **Acquisition:** User uploads ECG image via Frontend.
2. **Transmission:** Image sent to API Gateway (TLS 1.3).
3. **Preprocessing:** Backend normalizes image (1024x512), checks quality.
4. **Inference:** ViT Ensemble processes image -> JSON result.
5. **Logic Check:** Rule-based guardrails (e.g., "If HR > 100, cannot be Sinus Bradycardia").
6. **Reporting:** JSON result converted to HL7/FHIR.
7. **Storage:** Result stored, Audit Log updated.

## 4. Safety Classification (IEC 62304)
- **Class C:** Failure could result in death or serious injury (e.g., missed STEMI).
- **Mitigation:** Rigorous testing (Unit, Integration, System), Code Reviews, Static Analysis.

## 5. SOUP (Software of Unknown Provenance)
- **PyTorch:** Validated via extensive community use and internal regression testing.
- **FastAPI:** Validated via internal testing.
- **PostgreSQL:** Industry standard, validated.
