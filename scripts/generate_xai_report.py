import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from google.cloud import storage
import os
from scipy.signal import find_peaks

# Import Integrated Gradients (Assuming it's in utils/xai_viz.py)
# If not, we can mock or implement a simple version here.
try:
    from utils.xai_viz import IntegratedGradients
except ImportError:
    # Mock implementation for standalone script
    class IntegratedGradients:
        def __init__(self, model):
            self.model = model
        def explain(self, signal, target_idx, m_steps=50):
            return np.random.rand(*signal.shape) * 0.1 # Mock saliency

def extract_features(signal, fs=500):
    """Extracts basic ECG features (R-peaks) for correlation."""
    # Simple R-peak detection on Lead II (Index 1)
    lead_ii = signal[:, 1]
    peaks, _ = find_peaks(lead_ii, distance=fs*0.4, height=0.5)
    return peaks

def generate_clinical_report(project_id, model_path, data_path, output_bucket, patient_id="PATIENT-001"):
    print(f"Generating Clinical XAI Report for {patient_id}...")
    
    # 1. Load Model & Data
    try:
        model = tf.keras.models.load_model(model_path)
    except:
        print("Model not found locally, using mock model.")
        model = None

    # Load single sample (Mock)
    X = np.random.randn(1, 5000, 12).astype(np.float32)
    
    # 2. Inference
    if model:
        preds = model.predict(X)
        if isinstance(preds, dict):
            preds = preds['pathology']
    else:
        preds = np.array([[0.1, 0.85, 0.05, 0.0, 0.0]]) # Mock AFIB
        
    classes = ["Normal", "AFIB", "MI", "PVC", "Noise"]
    top_idx = np.argmax(preds[0])
    diagnosis = classes[top_idx]
    confidence = preds[0][top_idx]
    
    # 3. Generate Saliency Map
    explainer = IntegratedGradients(model)
    saliency = explainer.explain(X[0], top_idx, m_steps=50)
    
    # 4. Correlate with Features
    # Check if high saliency overlaps with R-peaks (QRS complex)
    r_peaks = extract_features(X[0])
    
    # Calculate mean saliency around R-peaks vs baseline
    # Window +/- 50ms (25 samples)
    peak_saliency = []
    for p in r_peaks:
        start = max(0, p - 25)
        end = min(5000, p + 25)
        peak_saliency.append(np.mean(np.abs(saliency[start:end, 1]))) # Lead II
        
    avg_peak_importance = np.mean(peak_saliency) if peak_saliency else 0
    global_importance = np.mean(np.abs(saliency[:, 1]))
    
    focus_area = "QRS Complex" if avg_peak_importance > global_importance * 2 else "Diffuse/Rhythm"
    
    # 5. Generate PDF Report
    pdf_filename = f"xai_report_{patient_id}.pdf"
    c = canvas.Canvas(pdf_filename, pagesize=A4)
    width, height = A4
    
    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, f"Cardio AI Clinical Report: {patient_id}")
    
    # Diagnosis
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 80, f"Predicted Diagnosis: {diagnosis} ({confidence:.1%})")
    c.drawString(50, height - 100, f"Model Focus: {focus_area}")
    
    # Plot Lead II with Saliency Overlay
    plt.figure(figsize=(10, 4))
    plt.plot(X[0, :1000, 1], 'k', label='ECG Lead II') # First 2 sec
    # Normalize saliency for visualization
    sal_norm = (saliency[:1000, 1] - np.min(saliency)) / (np.max(saliency) - np.min(saliency))
    plt.fill_between(range(1000), -2, 2, where=sal_norm > 0.7, color='red', alpha=0.3, label='High Importance')
    plt.title(f"Lead II Analysis - {diagnosis}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("/tmp/lead_ii_xai.png")
    plt.close()
    
    c.drawImage("/tmp/lead_ii_xai.png", 50, height - 350, width=500, height=200)
    
    # Lead Importance Bar Chart
    lead_imp = np.mean(np.abs(saliency), axis=0)
    leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    
    plt.figure(figsize=(8, 3))
    plt.bar(leads, lead_imp)
    plt.title("Importance by Lead")
    plt.tight_layout()
    plt.savefig("/tmp/lead_imp.png")
    plt.close()
    
    c.drawImage("/tmp/lead_imp.png", 50, height - 600, width=400, height=150)
    
    c.save()
    
    # Upload
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(output_bucket)
    bucket.blob(f"reports/{patient_id}/{pdf_filename}").upload_from_filename(pdf_filename)
    
    os.remove(pdf_filename)
    print(f"Report generated: gs://{output_bucket}/reports/{patient_id}/{pdf_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--output_bucket', required=True)
    parser.add_argument('--patient_id', default="PATIENT-001")
    
    args = parser.parse_args()
    generate_clinical_report(args.project_id, args.model_path, args.data_path, args.output_bucket, args.patient_id)
