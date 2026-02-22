import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

class IntegratedGradients:
    """
    Local implementation of Integrated Gradients for batch reporting.
    """
    def __init__(self, model):
        self.model = model

    def interpolate_images(self, baseline, image, alphas):
        alphas_x = alphas[:, tf.newaxis, tf.newaxis]
        baseline_x = tf.expand_dims(baseline, axis=0)
        input_x = tf.expand_dims(image, axis=0)
        delta = input_x - baseline_x
        images = baseline_x +  alphas_x * delta
        return images

    def compute_gradients(self, images, target_class_idx):
        with tf.GradientTape() as tape:
            tape.watch(images)
            # Predict
            # Handle multi-output model (pathology head)
            preds = self.model(images)
            if isinstance(preds, dict):
                probs = preds['pathology'][:, target_class_idx]
            else:
                probs = preds[:, target_class_idx]
                
        return tape.gradient(probs, images)

    def explain(self, image, target_class_idx, m_steps=50, baseline=None):
        # image shape: (5000, 12)
        if baseline is None:
            baseline = tf.zeros_like(image)

        # 1. Generate alphas
        alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)

        # 2. Interpolate path
        interpolated_images = self.interpolate_images(baseline, image, alphas)

        # 3. Compute Gradients
        # (Batch compute if memory allows, else loop)
        grads = self.compute_gradients(interpolated_images, target_class_idx)

        # 4. Integral approximation (Trapezoidal rule)
        # Average gradients
        avg_grads = tf.reduce_mean(grads[:-1] + grads[1:], axis=0) / 2.0

        # 5. Scale by (Input - Baseline)
        integrated_grads = (image - baseline) * avg_grads

        return integrated_grads.numpy()

def correlate_with_features(signal_data, saliency_map, fs=500):
    """
    Checks if high saliency regions overlap with QRS complexes.
    """
    # Use Lead II (index 1) for rhythm
    lead_idx = 1
    lead_sig = signal_data[:, lead_idx]
    lead_saliency = np.abs(saliency_map[:, lead_idx])
    
    # Detect QRS
    # Simple squared energy detector
    sos = signal.butter(3, [5, 15], 'bandpass', fs=fs, output='sos')
    filtered = signal.sosfiltfilt(sos, lead_sig)
    energy = filtered ** 2
    peaks, _ = signal.find_peaks(energy, height=np.mean(energy)*3, distance=int(0.2*fs))
    
    # Define QRS regions (Peak +/- 50ms)
    qrs_mask = np.zeros_like(lead_sig, dtype=bool)
    window = int(0.05 * fs)
    for p in peaks:
        start = max(0, p - window)
        end = min(len(lead_sig), p + window)
        qrs_mask[start:end] = True
        
    # Calculate Saliency in QRS vs Non-QRS
    total_saliency = np.sum(lead_saliency)
    qrs_saliency = np.sum(lead_saliency[qrs_mask])
    
    ratio = qrs_saliency / total_saliency if total_saliency > 0 else 0
    
    return {
        "qrs_peaks": len(peaks),
        "saliency_in_qrs_ratio": ratio,
        "interpretation": "Focus on QRS" if ratio > 0.5 else "Focus on T-wave/Noise"
    }

def visualize_explanation(signal_data, saliency_map, title="ECG Explanation"):
    """
    Plots 12-lead ECG with saliency heatmap overlay.
    """
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    fig, axes = plt.subplots(12, 1, figsize=(15, 20), sharex=True)
    
    # Normalize saliency for visualization (0-1)
    saliency_norm = np.abs(saliency_map)
    saliency_norm = (saliency_norm - saliency_norm.min()) / (saliency_norm.max() - saliency_norm.min() + 1e-8)
    
    for i, ax in enumerate(axes):
        # Plot Signal
        ax.plot(signal_data[:, i], 'k', linewidth=0.8, label='Signal')
        
        # Overlay Saliency (as colored span or scatter)
        # We use scatter for variable intensity
        t = np.arange(len(signal_data))
        # Only plot high saliency points to reduce clutter
        mask = saliency_norm[:, i] > 0.1
        if mask.sum() > 0:
            ax.scatter(t[mask], signal_data[mask, i], c=saliency_norm[mask, i], cmap='hot', s=10, alpha=0.5, label='Importance')
        
        ax.set_ylabel(leads[i], rotation=0, labelpad=20)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        if i < 11:
            ax.set_xticks([])
            
    axes[0].set_title(title)
    axes[-1].set_xlabel("Time (samples)")
    
    plt.tight_layout()
    return fig
