# Cybersecurity Assessment (FDA Guidance / MDCG 2019-16)

## 1. Threat Modeling (STRIDE)
- **Spoofing:** Mitigated by mTLS and strong authentication (OAuth2/OIDC).
- **Tampering:** Mitigated by digital signatures on reports and immutable audit logs.
- **Repudiation:** Mitigated by comprehensive logging of all user actions.
- **Information Disclosure:** Mitigated by encryption at rest (AES-256) and in transit (TLS 1.3).
- **Denial of Service:** Mitigated by rate limiting, WAF, and auto-scaling.
- **Elevation of Privilege:** Mitigated by RBAC (Role-Based Access Control) and Principle of Least Privilege.

## 2. Vulnerability Management
- **Scanning:** Automated daily scans (Snyk, Dependabot) for dependencies.
- **Penetration Testing:** Annual third-party pentest.
- **Patch Management:** Critical patches applied within 48 hours.

## 3. Data Privacy (HIPAA / GDPR)
- **De-identification:** All PHI removed/hashed before processing in cloud (if applicable).
- **Data Residency:** Data stored in region-specific compliant zones (e.g., AWS Frankfurt for GDPR).
- **Access Control:** Strict "Need to Know" basis.

## 4. Incident Response Plan
1. **Detection:** SIEM alerts (Splunk/Datadog).
2. **Containment:** Isolate affected systems.
3. **Eradication:** Remove threat, patch vulnerability.
4. **Recovery:** Restore from secure backups.
5. **Lessons Learned:** Post-mortem analysis.

## 5. SBOM (Software Bill of Materials)
- Maintained and updated with every release (CycloneDX format).
