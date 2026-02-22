import argparse
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from google.cloud import storage
from utils.xai_viz import IntegratedGradients, visualize_explanation, correlate_with_features

def generate_report(model_path, data_path, output_bucket, run_id):
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # Initialize Explainer
    explainer = IntegratedGradients(model)
    
    # Load Data (Mock)
    print("Loading sample data...")
    # X shape (Batch, 5000, 12)
    X = np.random.randn(5, 5000, 12).astype(np.float32)
    
    client = storage.Client()
    bucket = client.bucket(output_bucket)
    
    classes = ["Normal", "AFIB", "MI", "PVC", "Noise"]
    
    for i in range(len(X)):
        signal = X[i]
        
        # Predict
        preds = model.predict(signal[np.newaxis, ...])
        if isinstance(preds, dict):
            probs = preds['pathology'][0]
        else:
            probs = preds[0]
            
        top_class = np.argmax(probs)
        class_name = classes[top_class]
        
        print(f"Sample {i}: Predicted {class_name} ({probs[top_class]:.2f})")
        
        # Explain
        saliency = explainer.explain(signal, top_class)
        
        # Correlate
        features = correlate_with_features(signal, saliency)
        print(f"  -> XAI Analysis: {features}")
        
        # Visualize
        fig = visualize_explanation(
            signal, saliency, 
            title=f"Sample {i}: {class_name} (Conf: {probs[top_class]:.2f})\nFocus: {features['interpretation']}"
        )
        
        # Save
        local_path = f"/tmp/xai_sample_{i}.png"
        fig.savefig(local_path)
        plt.close(fig)
        
        # Upload
        blob = bucket.blob(f"xai_reports/{run_id}/sample_{i}_{class_name}.png")
        blob.upload_from_filename(local_path)
        
    print(f"XAI Reports uploaded to gs://{output_bucket}/xai_reports/{run_id}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--data_path', default='mock')
    parser.add_argument('--output_bucket', required=True)
    parser.add_argument('--run_id', required=True)
    
    args = parser.parse_args()
    generate_report(args.model_path, args.data_path, args.output_bucket, args.run_id)
