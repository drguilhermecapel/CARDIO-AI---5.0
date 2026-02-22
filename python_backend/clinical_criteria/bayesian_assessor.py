import numpy as np
from typing import Dict, List, Any, Tuple

class BayesianRiskAssessor:
    """
    Bayesian Clinical Risk Assessor integrating TIMI, HEART, GRACE scores
    and Pre/Post-test probability analysis for ACS/CAD.
    """
    
    def __init__(self):
        # Diamond-Forrester Pre-test Probabilities of CAD (Age, Sex, Symptom)
        # Format: (AgeMin, AgeMax, Sex, SymptomType) -> Prob
        # SymptomType: 0=Non-anginal, 1=Atypical, 2=Typical
        self.df_table = {
            # Male
            (30, 39, 'Male', 'Non-anginal'): 0.05,
            (30, 39, 'Male', 'Atypical'): 0.22,
            (30, 39, 'Male', 'Typical'): 0.70,
            (40, 49, 'Male', 'Non-anginal'): 0.14,
            (40, 49, 'Male', 'Atypical'): 0.46,
            (40, 49, 'Male', 'Typical'): 0.87,
            (50, 59, 'Male', 'Non-anginal'): 0.21,
            (50, 59, 'Male', 'Atypical'): 0.59,
            (50, 59, 'Male', 'Typical'): 0.93,
            (60, 69, 'Male', 'Non-anginal'): 0.28,
            (60, 69, 'Male', 'Atypical'): 0.67,
            (60, 69, 'Male', 'Typical'): 0.94,
            # Female
            (30, 39, 'Female', 'Non-anginal'): 0.01,
            (30, 39, 'Female', 'Atypical'): 0.04,
            (30, 39, 'Female', 'Typical'): 0.26,
            (40, 49, 'Female', 'Non-anginal'): 0.03,
            (40, 49, 'Female', 'Atypical'): 0.13,
            (40, 49, 'Female', 'Typical'): 0.55,
            (50, 59, 'Female', 'Non-anginal'): 0.08,
            (50, 59, 'Female', 'Atypical'): 0.32,
            (50, 59, 'Female', 'Typical'): 0.79,
            (60, 69, 'Female', 'Non-anginal'): 0.19,
            (60, 69, 'Female', 'Atypical'): 0.54,
            (60, 69, 'Female', 'Typical'): 0.91,
        }

        # Likelihood Ratios for ECG Findings (Approximate)
        self.ecg_lrs = {
            "STEMI": 100.0, # Pathognomonic
            "NSTEMI": 10.0, # Significant STD/T-inv
            "LBBB": 5.0, # New LBBB
            "Ischemia": 2.0, # Minor changes
            "Normal": 0.14, # Normal ECG rules out MI well but not perfectly
            "Nonspecific": 0.5 # Doesn't help much
        }

    def _get_symptom_type(self, symptoms: List[str]) -> str:
        if "typical_angina" in symptoms: return "Typical"
        if "atypical_angina" in symptoms: return "Atypical"
        if "chest_pain" in symptoms: return "Non-anginal" # Conservative default for generic CP
        return "Asymptomatic"

    def estimate_pre_test_prob(self, age: int, sex: str, symptoms: List[str]) -> float:
        """
        Estimates pre-test probability of CAD using Diamond-Forrester.
        """
        s_type = self._get_symptom_type(symptoms)
        if s_type == "Asymptomatic": return 0.01
        
        # Find bucket
        age_bucket = None
        if 30 <= age <= 39: age_bucket = (30, 39)
        elif 40 <= age <= 49: age_bucket = (40, 49)
        elif 50 <= age <= 59: age_bucket = (50, 59)
        elif 60 <= age <= 69: age_bucket = (60, 69)
        elif age >= 70: age_bucket = (60, 69) # Cap at max bucket
        else: return 0.01 # Very young
        
        key = (age_bucket[0], age_bucket[1], sex, s_type)
        return self.df_table.get(key, 0.1)

    def calculate_post_test_prob(self, pre_prob: float, ecg_diagnosis: str) -> float:
        """
        Bayesian Update: Post-Test Prob = (Pre-Odds * LR) / (1 + Pre-Odds * LR)
        """
        if pre_prob >= 1.0: return 1.0
        if pre_prob <= 0.0: return 0.0
        
        lr = self.ecg_lrs.get(ecg_diagnosis, 1.0)
        
        # Convert to Odds
        pre_odds = pre_prob / (1 - pre_prob)
        
        # Update Odds
        post_odds = pre_odds * lr
        
        # Convert back to Prob
        post_prob = post_odds / (1 + post_odds)
        return float(post_prob)

    def calculate_timi(self, data: Dict[str, Any], ecg_st_dev: bool) -> Dict[str, Any]:
        """
        TIMI Risk Score for UA/NSTEMI.
        """
        score = 0
        age = data.get('age', 0)
        rf = data.get('risk_factors', [])
        history = data.get('history', [])
        symptoms = data.get('symptoms', [])
        labs = data.get('labs', {})
        
        # 1. Age >= 65
        if age >= 65: score += 1
        # 2. >= 3 Risk Factors
        if len(rf) >= 3: score += 1
        # 3. Known CAD (stenosis >= 50%)
        if 'cad' in history or 'prior_mi' in history or 'pci' in history: score += 1
        # 4. ASA use in last 7 days
        if 'asa_use' in history: score += 1
        # 5. Severe angina (>= 2 episodes in 24h)
        if 'severe_angina' in symptoms: score += 1
        # 6. EKG ST changes >= 0.5mm
        if ecg_st_dev: score += 1
        # 7. Positive Cardiac Marker
        if labs.get('troponin_positive', False): score += 1
        
        risk_map = {
            0: 4.7, 1: 4.7, 2: 8.3, 3: 13.2, 4: 19.9, 5: 26.2, 6: 40.9, 7: 40.9
        }
        
        return {
            "score": score,
            "risk_mace_14d": risk_map.get(score, 40.9),
            "risk_level": "High" if score >= 5 else ("Medium" if score >= 3 else "Low")
        }

    def calculate_heart(self, data: Dict[str, Any], ecg_score: int) -> Dict[str, Any]:
        """
        HEART Score.
        ecg_score: 0 (Normal), 1 (Nonspecific), 2 (Significant STD)
        """
        score = 0
        age = data.get('age', 0)
        rf = data.get('risk_factors', [])
        history = data.get('history', [])
        labs = data.get('labs', {})
        
        # History (Subjective, passed in data or inferred)
        # Assuming 'history_score' provided, else heuristic
        h_score = data.get('history_score', 1) # Default Moderate
        score += h_score
        
        # ECG
        score += ecg_score
        
        # Age
        if age >= 65: score += 2
        elif age >= 45: score += 1
        
        # Risk Factors
        # >=3 or History of Atherosclerotic Disease = 2
        has_athero = 'prior_mi' in history or 'stroke' in history or 'pad' in history
        if len(rf) >= 3 or has_athero: score += 2
        elif len(rf) >= 1: score += 1
        
        # Troponin
        trop = labs.get('troponin_ratio', 0) # Ratio to upper limit
        if trop >= 3: score += 2
        elif trop > 1: score += 1
        
        risk_map = {
            0: 1.7, 1: 1.7, 2: 1.7, 3: 1.7, # Low (0-3)
            4: 16.6, 5: 16.6, 6: 16.6, # Moderate (4-6)
            7: 50.0, 8: 50.0, 9: 50.0, 10: 50.0 # High (7-10)
        }
        
        risk_cat = "Low"
        if score >= 7: risk_cat = "High"
        elif score >= 4: risk_cat = "Moderate"
        
        return {
            "score": score,
            "risk_mace_6w": risk_map.get(score, 50.0),
            "risk_level": risk_cat
        }

    def calculate_grace(self, data: Dict[str, Any], ecg_st_dev: bool) -> Dict[str, Any]:
        """
        Simplified GRACE Score (In-hospital mortality).
        """
        score = 0
        age = data.get('age', 0)
        vitals = data.get('vitals', {})
        labs = data.get('labs', {})
        history = data.get('history', [])
        
        # Age
        if age < 30: score += 0
        elif age < 40: score += 18
        elif age < 50: score += 36
        elif age < 60: score += 55
        elif age < 70: score += 73
        elif age < 80: score += 91
        else: score += 100
        
        # Heart Rate
        hr = vitals.get('hr', 80)
        if hr < 50: score += 0
        elif hr < 70: score += 3
        elif hr < 90: score += 9
        elif hr < 110: score += 15
        elif hr < 150: score += 24
        else: score += 43
        
        # SBP
        sbp = vitals.get('sbp', 120)
        if sbp < 80: score += 58
        elif sbp < 100: score += 53
        elif sbp < 120: score += 43
        elif sbp < 140: score += 34
        elif sbp < 160: score += 24
        elif sbp < 200: score += 10
        else: score += 0
        
        # Creatinine
        creat = labs.get('creatinine', 1.0)
        if creat < 0.4: score += 1
        elif creat < 0.8: score += 4
        elif creat < 1.2: score += 7
        elif creat < 1.6: score += 10
        elif creat < 2.0: score += 13
        elif creat < 4.0: score += 21
        else: score += 28
        
        # Killip Class (1-4)
        killip = vitals.get('killip', 1)
        if killip == 2: score += 20
        elif killip == 3: score += 39
        elif killip == 4: score += 59
        
        # Cardiac Arrest at Admission
        if vitals.get('cardiac_arrest', False): score += 39
        
        # Elevated Enzymes
        if labs.get('troponin_positive', False): score += 14
        
        # ST Deviation
        if ecg_st_dev: score += 28
        
        # Mortality Probability (Sigmoid approx for GRACE)
        # P = 1 / (1 + exp(-(score - offset)/scale)) - simplified lookup
        mortality = 0.0
        if score < 100: mortality = 1.0
        elif score < 150: mortality = 5.0
        elif score < 200: mortality = 20.0
        else: mortality = 50.0
        
        return {
            "score": score,
            "mortality_risk_inhosp": mortality,
            "risk_level": "High" if score > 140 else "Low/Med"
        }

    def assess_patient(self, patient_data: Dict[str, Any], ecg_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main assessment method.
        """
        # Extract ECG details
        ecg_diag = ecg_analysis.get('diagnosis', 'Normal')
        ecg_st_dev = ecg_analysis.get('has_st_deviation', False)
        
        # Map ECG diagnosis to HEART ECG score
        heart_ecg_score = 0
        if ecg_st_dev or ecg_diag in ["STEMI", "NSTEMI", "LBBB"]:
            heart_ecg_score = 2
        elif ecg_diag != "Normal":
            heart_ecg_score = 1
            
        # 1. Pre-test Prob
        pre_prob = self.estimate_pre_test_prob(
            patient_data.get('age', 50),
            patient_data.get('sex', 'Male'),
            patient_data.get('symptoms', [])
        )
        
        # 2. Post-test Prob
        # Map diagnosis to generic category for LR lookup if needed
        lookup_diag = ecg_diag
        if "STEMI" in ecg_diag: lookup_diag = "STEMI"
        elif "Normal" in ecg_diag: lookup_diag = "Normal"
        elif ecg_st_dev: lookup_diag = "NSTEMI"
        
        post_prob = self.calculate_post_test_prob(pre_prob, lookup_diag)
        
        # 3. Scores
        timi = self.calculate_timi(patient_data, ecg_st_dev)
        heart = self.calculate_heart(patient_data, heart_ecg_score)
        grace = self.calculate_grace(patient_data, ecg_st_dev)
        
        return {
            "bayesian_analysis": {
                "pre_test_prob_cad": round(pre_prob, 3),
                "post_test_prob_cad": round(post_prob, 3),
                "ecg_likelihood_ratio": self.ecg_lrs.get(lookup_diag, 1.0)
            },
            "risk_scores": {
                "TIMI": timi,
                "HEART": heart,
                "GRACE": grace
            },
            "clinical_summary": f"Patient has {pre_prob:.1%} pre-test probability. ECG findings adjust this to {post_prob:.1%}."
        }
