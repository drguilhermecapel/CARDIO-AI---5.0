# Security & Compliance Checklist

**Target System:** Cardio AI Endpoint
**Date:** `YYYY-MM-DD`

## 1. Access Control (IAM)
- [ ] **Service Accounts**: Endpoint runs as a dedicated Service Account (not default Compute Engine SA).
- [ ] **Least Privilege**: Service Account has only `BigQuery Data Editor` and `Storage Object Viewer` roles.
- [ ] **Public Access**: Endpoint is **NOT** accessible to the public internet (Internal Load Balancer or VPC Service Controls).

## 2. Data Protection
- [ ] **Encryption in Transit**: All API traffic uses HTTPS/TLS 1.2+.
- [ ] **Encryption at Rest**: GCS buckets and BigQuery datasets use Customer-Managed Encryption Keys (CMEK) or Google-managed defaults.
- [ ] **De-identification**: No PHI (names, MRNs) is logged to application logs (stdout/stderr).

## 3. Vulnerability Management
- [ ] **Container Scanning**: Docker image scanned for vulnerabilities (Artifact Registry Analysis).
    - [ ] No Critical/High vulnerabilities found.
- [ ] **Dependency Audit**: `npm audit` / `pip audit` run successfully.

## 4. Monitoring & Logging
- [ ] **Audit Logs**: Cloud Audit Logs enabled for Admin Activity and Data Access.
- [ ] **Application Logs**: Structured JSON logging implemented.
- [ ] **Alerting**: Alerts configured for:
    - [ ] High Latency (>1s)
    - [ ] High Error Rate (>1%)
    - [ ] Prediction Drift (>0.1)

## 5. Regulatory
- [ ] **Model Card**: Generated and reviewed.
- [ ] **Bias Report**: Generated and reviewed.
- [ ] **Human-in-the-Loop**: Feedback mechanism verified operational.

**Verified By:** ________________________________ (Security Engineer)
