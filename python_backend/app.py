from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import torch
import numpy as np
from PIL import Image
import io
from model_architecture import ECGViT

app = FastAPI(title="ECG Hospital-Grade AI", version="1.0.0")

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = {
    'arrhythmia': 12,
    'ischemia': 8,
    'structural': 5,
    'conduction': 6
}

# --- Load Model ---
model = ECGViT(NUM_CLASSES).to(DEVICE)
# model.load_state_dict(torch.load("weights/best_model.pth"))
model.eval()

# --- Schemas ---
class Diagnosis(BaseModel):
    condition: string
    probability: float
    severity: string

class AnalysisResponse(BaseModel):
    diagnóstico_principal: str
    confiança_principal: float
    diagnósticos_diferenciais: List[Diagnosis]
    regiões_críticas: Dict[str, List[int]] # Heatmap coords
    qualidade_sinal: float
    alertas: List[str]
    tempo_processamento: float

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((1024, 512))
    # Normalize to [0, 1] and standard ImageNet mean/std
    img_array = np.array(img) / 255.0
    img_array = (img_array - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return torch.tensor(img_array).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

@app.post("/predict", response_model=AnalysisResponse)
async def predict_ecg(file: UploadFile = File(...)):
    import time
    start_time = time.time()
    
    try:
        contents = await file.read()
        input_tensor = preprocess_image(contents)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            
        # --- Post-Processing (Mock Logic for Reference) ---
        # In production, apply Softmax, Thresholding, and Mapping to labels
        
        # Generate GradCAM (Placeholder)
        heatmap = {"V1": [100, 200, 150, 250], "II": [50, 100, 80, 120]}
        
        process_time = time.time() - start_time
        
        return {
            "diagnóstico_principal": "Sinus Rhythm",
            "confiança_principal": 0.98,
            "diagnósticos_diferenciais": [],
            "regiões_críticas": heatmap,
            "qualidade_sinal": 9.5,
            "alertas": [],
            "tempo_processamento": process_time
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "online", "gpu": torch.cuda.is_available()}
