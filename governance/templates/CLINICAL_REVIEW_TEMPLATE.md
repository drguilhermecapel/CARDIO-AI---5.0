# Clinical Model Review Template

**Model Version:** `vX.Y.Z`
**Date:** `YYYY-MM-DD`
**Reviewer:** `Dr. [Name]`, Chief of Cardiology / Lead Data Scientist

## 1. Performance Summary
- [ ] **Sensitivity/Specificity Check**: Does the model meet the defined thresholds for all pathologies?
    - MI Sensitivity: ____% (Target: >90%)
    - AFIB Sensitivity: ____% (Target: >95%)
- [ ] **Bias Check**: Is the max F1 disparity < 5%?
    - Max Disparity observed: ____% (Group: __________)

## 2. Safety & Error Analysis
- [ ] **Critical Miss Analysis**: Were there any STEMI cases classified as Normal?
    - [ ] Yes (Reject immediately)
    - [ ] No
- [ ] **False Positive Review**: Are false positives manageable? (High FP rate leads to alarm fatigue).
    - Comment: ________________________________________________________________

## 3. Explainability Review
- [ ] **Saliency Maps**: Do the heatmaps align with clinical features (e.g., highlighting ST-segment in MI)?
    - [ ] Yes, consistently.
    - [ ] Mostly, with some noise.
    - [ ] No, model appears to look at artifacts.

## 4. Deployment Recommendation
- [ ] **APPROVE** for Production Deployment.
- [ ] **APPROVE** for Shadow Mode (Silent evaluation).
- [ ] **REJECT** (Requires retraining).

**Signature:** __________________________________________________
