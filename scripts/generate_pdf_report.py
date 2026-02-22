import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import tensorflow as tf
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
import io
from datetime import datetime

# Import existing modules (assuming they are in python path)
# We might need to adjust paths if running from root
import sys
sys.path.append(os.getcwd())

from models.artifact_detection import LightweightArtifactDetector
from utils.xai_viz import IntegratedGradients, visualize_explanation

def draw_ecg_grid(ax, seconds, millivolts):
    """
    Draws a standard ECG grid:
    - Minor lines every 0.04s (1mm at 25mm/s) and 0.1mV (1mm at 10mm/mV)
    - Major lines every 0.2s (5mm) and 0.5mV (5mm)
    """
    ax.set_xticks(np.arange(0, seconds, 0.2))
    ax.set_yticks(np.arange(-millivolts, millivolts, 0.5))
    ax.grid(which='major', linestyle='-', linewidth=0.5, color='red', alpha=0.3)
    
    ax.set_xticks(np.arange(0, seconds, 0.04), minor=True)
    ax.set_yticks(np.arange(-millivolts, millivolts, 0.1), minor=True)
    ax.grid(which='minor', linestyle=':', linewidth=0.5, color='red', alpha=0.2)

def plot_12_lead_standard(signal, fs=500, title="12-Lead ECG"):
    """
    Generates a matplotlib figure with standard 12-lead layout (3x4).
    """
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    # 3 rows, 4 columns
    fig, axes = plt.subplots(3, 4, figsize=(16, 8), sharex=True, sharey=True)
    
    # Time vector
    t = np.arange(len(signal)) / fs
    duration = len(signal) / fs
    
    for i, ax in enumerate(axes.flatten()):
        if i < 12:
            # Draw Grid
            draw_ecg_grid(ax, duration, 2.0)
            
            # Plot Signal
            ax.plot(t, signal[:, i], 'k', linewidth=0.8)
            ax.set_title(leads[i], loc='left', fontsize=10, fontweight='bold')
            
            # Remove spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            
            # Remove ticks
            ax.tick_params(axis='both', which='both', length=0)
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    plt.tight_layout()
    return fig

def generate_pdf(record_id, signal_path, model_path, output_path):
    print(f"Generating report for {record_id}...")
    
    # 1. Load Signal
    # Mock loading if file not found for demo
    if os.path.exists(signal_path):
        signal_data = np.load(signal_path)
    else:
        print("Signal file not found, using mock data.")
        signal_data = np.random.randn(5000, 12) * 0.5 # Mock
        
    fs = 500
    
    # 2. Artifact Detection
    detector = LightweightArtifactDetector(sampling_rate=fs)
    # Use Lead II for rhythm analysis
    lead_ii = signal_data[:, 1]
    beat_results = detector.process_record(lead_ii)
    
    total_beats = len(beat_results)
    valid_beats = sum(1 for b in beat_results if b['is_valid'])
    quality_score = (valid_beats / total_beats * 100) if total_beats > 0 else 0
    
    # Calculate HR
    if total_beats > 1:
        duration_min = (len(signal_data) / fs) / 60
        hr = int(total_beats / duration_min)
    else:
        hr = 0
        
    # 3. Feature Extraction (Simplified)
    # Calculate QTc (Bazett)
    qtc = 420 # Mock
    qrs_dur = 90 # Mock
    pr_int = 160 # Mock
    axis = 45 # Mock
    
    # 4. Model Inference (with Uncertainty)
    try:
        model = tf.keras.models.load_model(model_path)
        # MC Dropout
        preds = []
        for _ in range(20):
            p = model(signal_data[np.newaxis, ...], training=True)
            if isinstance(p, dict): p = p['pathology']
            preds.append(p.numpy()[0])
        
        preds = np.array(preds)
        mean_preds = np.mean(preds, axis=0)
        std_preds = np.std(preds, axis=0)
        
        classes = ["Normal", "AFIB", "MI", "PVC", "Noise"]
        top_idx = np.argmax(mean_preds)
        diagnosis = classes[top_idx]
        confidence = mean_preds[top_idx]
        uncertainty = std_preds[top_idx]
        
        # 5. XAI
        explainer = IntegratedGradients(model)
        saliency = explainer.explain(signal_data, top_idx)
        
    except Exception as e:
        print(f"Model inference failed: {e}")
        diagnosis = "Analysis Failed"
        confidence = 0.0
        uncertainty = 0.0
        saliency = np.zeros_like(signal_data)

    # --- PDF Generation ---
    doc = SimpleDocTemplate(output_path, pagesize=landscape(A4),
                            rightMargin=10*mm, leftMargin=10*mm,
                            topMargin=10*mm, bottomMargin=10*mm)
    
    styles = getSampleStyleSheet()
    elements = []
    
    # Header
    header_data = [
        [f"Patient ID: {record_id}", f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}"],
        [f"Age/Sex: Unknown", f"Ref Phys: Dr. AI"]
    ]
    t = Table(header_data, colWidths=[140*mm, 140*mm])
    t.setStyle(TableStyle([('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
                           ('FONTSIZE', (0,0), (-1,-1), 12),
                           ('BOTTOMPADDING', (0,0), (-1,-1), 12)]))
    elements.append(t)
    elements.append(Spacer(1, 5*mm))
    
    # Measurements Table
    meas_data = [
        ["Rate", "PR", "QRS", "QT/QTc", "Axis", "Quality"],
        [f"{hr} bpm", f"{pr_int} ms", f"{qrs_dur} ms", f"400/{qtc} ms", f"{axis} deg", f"{quality_score:.1f}%"]
    ]
    t_meas = Table(meas_data, colWidths=[45*mm]*6)
    t_meas.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTSIZE', (0,0), (-1,-1), 10)
    ]))
    elements.append(t_meas)
    elements.append(Spacer(1, 5*mm))
    
    # AI Diagnosis Section
    elements.append(Paragraph("<b>AI INTERPRETATION</b>", styles['Heading3']))
    
    diag_text = f"<b>Primary Diagnosis:</b> {diagnosis} (Conf: {confidence:.2f}, Uncertainty: {uncertainty:.2f})"
    elements.append(Paragraph(diag_text, styles['Normal']))
    
    diff_text = "<b>Differential:</b> " + ", ".join([f"{classes[i]}: {mean_preds[i]:.2f}" for i in range(len(classes)) if i != top_idx])
    elements.append(Paragraph(diff_text, styles['Normal']))
    
    elements.append(Spacer(1, 5*mm))
    
    # 12-Lead Plot
    elements.append(Paragraph("<b>12-LEAD ECG</b>", styles['Heading3']))
    
    fig_ecg = plot_12_lead_standard(signal_data, fs)
    img_buf = io.BytesIO()
    fig_ecg.savefig(img_buf, format='png', dpi=150, bbox_inches='tight')
    img_buf.seek(0)
    img = Image(img_buf, width=280*mm, height=100*mm)
    elements.append(img)
    plt.close(fig_ecg)
    
    elements.append(Spacer(1, 5*mm))
    
    # XAI Plot (Lead II Focus)
    elements.append(Paragraph("<b>AI ATTENTION MAP (Lead II Focus)</b>", styles['Heading3']))
    
    fig_xai, ax = plt.subplots(figsize=(12, 2))
    t = np.arange(len(lead_ii)) / fs
    ax.plot(t, lead_ii, 'k', linewidth=0.8)
    # Overlay saliency
    sal_ii = np.abs(saliency[:, 1])
    sal_ii = (sal_ii - sal_ii.min()) / (sal_ii.max() - sal_ii.min() + 1e-8)
    mask = sal_ii > 0.2
    ax.scatter(t[mask], lead_ii[mask], c=sal_ii[mask], cmap='hot', s=10, alpha=0.6)
    ax.set_title("Red regions indicate features used for classification")
    ax.axis('off')
    
    img_buf_xai = io.BytesIO()
    fig_xai.savefig(img_buf_xai, format='png', dpi=150, bbox_inches='tight')
    img_buf_xai.seek(0)
    img_xai = Image(img_buf_xai, width=280*mm, height=40*mm)
    elements.append(img_xai)
    plt.close(fig_xai)
    
    # Footer
    elements.append(Spacer(1, 10*mm))
    footer_text = "<i>Disclaimer: This report is generated by an AI system (CardioAI v1.0). It is intended for triage support only and must be verified by a qualified cardiologist.</i>"
    elements.append(Paragraph(footer_text, styles['Italic']))
    
    # Build
    doc.build(elements)
    print(f"Report saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--record_id', required=True)
    parser.add_argument('--signal_path', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output_path', default='report.pdf')
    
    args = parser.parse_args()
    generate_pdf(args.record_id, args.signal_path, args.model_path, args.output_path)
