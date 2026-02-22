# Clinical Code Mappings (ICD-10, SNOMED-CT) and Translations

DIAGNOSIS_MAP = {
    "Normal Sinus Rhythm": {
        "icd10": "R00.0", # Tachycardia unspecified? No, usually just Normal. R94.31 is abnormal. 
        # Z01.810 is encounter for exam. Let's use R00.0 as placeholder or specific code if available.
        # Actually, Normal often doesn't map to a disease code.
        "snomed": "102542009", # Normal electrocardiogram
        "translations": {
            "en": "Normal Sinus Rhythm",
            "pt": "Ritmo Sinusal Normal",
            "es": "Ritmo Sinusal Normal"
        },
        "recommendations": {
            "en": ["Routine follow-up."],
            "pt": ["Seguimento de rotina."],
            "es": ["Seguimiento de rutina."]
        },
        "severity": "Low"
    },
    "Atrial Fibrillation": {
        "icd10": "I48.91",
        "snomed": "49436004",
        "translations": {
            "en": "Atrial Fibrillation",
            "pt": "Fibrilação Atrial",
            "es": "Fibrilación Auricular"
        },
        "recommendations": {
            "en": ["Evaluate anticoagulation (CHA2DS2-VASc).", "Rate vs Rhythm control."],
            "pt": ["Avaliar anticoagulação (CHA2DS2-VASc).", "Controle de frequência vs ritmo."],
            "es": ["Evaluar anticoagulación (CHA2DS2-VASc).", "Control de frecuencia vs ritmo."]
        },
        "severity": "Medium"
    },
    "STEMI": {
        "icd10": "I21.3", # ST elevation (STEMI) myocardial infarction of unspecified site
        "snomed": "401303003",
        "translations": {
            "en": "ST-Elevation Myocardial Infarction (STEMI)",
            "pt": "Infarto Agudo do Miocárdio com Supra de ST (IAMCSST)",
            "es": "Infarto Agudo de Miocardio con Elevación del ST"
        },
        "recommendations": {
            "en": ["ACTIVATE STEMI PROTOCOL.", "Immediate reperfusion therapy (PCI or Fibrinolysis)."],
            "pt": ["ATIVAR PROTOCOLO DE IAM.", "Terapia de reperfusão imediata (Angioplastia ou Fibrinólise)."],
            "es": ["ACTIVAR PROTOCOLO IAM.", "Terapia de reperfusión inmediata."]
        },
        "severity": "High"
    },
    "NSTEMI": {
        "icd10": "I21.4", # Non-ST elevation (NSTEMI) myocardial infarction
        "snomed": "401314000",
        "translations": {
            "en": "Non-ST Elevation Myocardial Infarction (NSTEMI)",
            "pt": "Infarto Agudo do Miocárdio sem Supra de ST (IAMSSST)",
            "es": "Infarto Agudo de Miocardio sin Elevación del ST"
        },
        "recommendations": {
            "en": ["Admit to CCU.", "Serial Troponins.", "Risk Stratification (TIMI/GRACE)."],
            "pt": ["Admissão em UTI Coronariana.", "Troponinas seriadas.", "Estratificação de risco (TIMI/GRACE)."],
            "es": ["Ingreso a UCC.", "Troponinas seriadas.", "Estratificación de riesgo."]
        },
        "severity": "High"
    },
    "LBBB": {
        "icd10": "I44.7",
        "snomed": "164909002",
        "translations": {
            "en": "Left Bundle Branch Block",
            "pt": "Bloqueio de Ramo Esquerdo (BRE)",
            "es": "Bloqueo de Rama Izquierda"
        },
        "recommendations": {
            "en": ["If new onset, rule out ischemia.", "Evaluate for heart failure."],
            "pt": ["Se novo, descartar isquemia.", "Avaliar insuficiência cardíaca."],
            "es": ["Si es nuevo, descartar isquemia.", "Evaluar insuficiencia cardíaca."]
        },
        "severity": "Medium"
    },
    "RBBB": {
        "icd10": "I45.1",
        "snomed": "59118001",
        "translations": {
            "en": "Right Bundle Branch Block",
            "pt": "Bloqueio de Ramo Direito (BRD)",
            "es": "Bloqueo de Rama Derecha"
        },
        "recommendations": {
            "en": ["Usually benign in isolation.", "Rule out RV strain if acute."],
            "pt": ["Geralmente benigno isoladamente.", "Descartar sobrecarga de VD se agudo."],
            "es": ["Generalmente benigno.", "Descartar sobrecarga VD."]
        },
        "severity": "Low"
    }
}
