from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from preprocessing import ECGPreprocessor
from inference_engine import InferenceEngine
import redis
import json
import hashlib
import time
import os

app = FastAPI(title="CardioAI Nexus High-Performance API", version="2.0.0")

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Components ---
preprocessor = ECGPreprocessor()
# Ensure model exists or handle error
model_path = os.getenv("MODEL_PATH", "ecg_mobilenet_optimized.onnx")
if not os.path.exists(model_path):
    print(f"WARNING: Model not found at {model_path}. Inference will fail.")
    engine = None
else:
    engine = InferenceEngine(model_path=model_path)

# --- Redis Cache ---
redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, db=0)

def get_cache_key(image_bytes):
    return hashlib.sha256(image_bytes).hexdigest()

@app.get("/health")
def health_check():
    return {"status": "online", "gpu_available": engine.providers[0] != 'CPUExecutionProvider' if engine else False}

@app.post("/analyze")
async def analyze_ecg(file: UploadFile = File(...)):
    start_total = time.time()
    
    try:
        contents = await file.read()
        
        # 1. Check Cache
        cache_key = get_cache_key(contents)
        cached_result = redis_client.get(cache_key)
        if cached_result:
            return json.loads(cached_result)
        
        if not engine:
             raise HTTPException(status_code=503, detail="Inference engine not initialized")

        # 2. Preprocessing
        input_tensor = preprocessor.preprocess_pipeline(contents)
        
        # 3. Inference
        result = engine.predict(input_tensor)
        
        # 4. Post-processing & Formatting
        response = {
            "diagnosis": "Sinus Rhythm", # Placeholder: Map logits to labels
            "confidence": float(np.max(result['probabilities'])),
            "probabilities": result['probabilities'],
            "telemetry": {
                "inference_latency_ms": result['latency_ms'],
                "total_latency_ms": (time.time() - start_total) * 1000,
                "device": result['device']
            }
        }
        
        # 5. Cache Result (TTL 1 hour)
        redis_client.setex(cache_key, 3600, json.dumps(response))
        
        return response
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
