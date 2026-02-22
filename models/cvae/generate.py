import tensorflow as tf
import numpy as np
import argparse
import os
from google.cloud import storage

def generate(decoder_path, output_bucket, class_label, num_samples=10):
    """
    Generates synthetic ECGs for a specific class.
    """
    print(f"Loading decoder from {decoder_path}...")
    decoder = tf.keras.models.load_model(decoder_path)
    
    latent_dim = decoder.input_shape[0][1] # [z_input, label_input]
    num_classes = decoder.input_shape[1][1]
    
    # Sample Z
    z = np.random.normal(size=(num_samples, latent_dim))
    
    # Create Label Vector
    # Assuming class_label is an integer index
    labels = np.zeros((num_samples, num_classes))
    labels[:, class_label] = 1.0
    
    print(f"Generating {num_samples} samples for class {class_label}...")
    generated_signals = decoder.predict([z, labels])
    
    # Save to GCS Staging
    client = storage.Client()
    bucket = client.bucket(output_bucket)
    
    timestamp = tf.timestamp().numpy()
    
    for i in range(num_samples):
        sig = generated_signals[i]
        filename = f"staging/synthetic_class_{class_label}_{i}_{timestamp}.npy"
        
        # Save locally then upload
        local_path = f"/tmp/synthetic_{i}.npy"
        np.save(local_path, sig)
        
        blob = bucket.blob(filename)
        blob.upload_from_filename(local_path)
        
        # Add metadata for review
        blob.metadata = {'status': 'PENDING_REVIEW', 'class': str(class_label)}
        blob.patch()
        
    print(f"Uploaded {num_samples} samples to gs://{output_bucket}/staging/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--decoder_path', required=True)
    parser.add_argument('--output_bucket', required=True)
    parser.add_argument('--class_label', type=int, required=True)
    parser.add_argument('--num_samples', type=int, default=10)
    
    args = parser.parse_args()
    generate(args.decoder_path, args.output_bucket, args.class_label, args.num_samples)
