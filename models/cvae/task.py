import argparse
import os
import tensorflow as tf
import numpy as np
from model import CVAE

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--job-dir', type=str, required=True, help='GCS location to write checkpoints and export models')
    return parser.parse_args()

def load_data():
    # MOCK DATA LOADER
    # In production, read from BigQuery or TFRecords in GCS
    print("Loading mock data...")
    num_samples = 1000
    signal_len = 5000
    channels = 12
    num_classes = 5
    
    X = np.random.randn(num_samples, signal_len, channels).astype(np.float32)
    # One-hot labels
    y_indices = np.random.randint(0, num_classes, size=num_samples)
    y = tf.one_hot(y_indices, num_classes)
    
    dataset = tf.data.Dataset.from_tensor_slices(((X, y),)) # Tuple input for model.fit
    dataset = dataset.shuffle(1000).batch(32)
    return dataset

def main():
    args = get_args()
    
    print(f"Training CVAE for {args.epochs} epochs...")
    
    dataset = load_data()
    
    cvae = CVAE(signal_len=5000, channels=12, num_classes=5)
    cvae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate))
    
    cvae.fit(dataset, epochs=args.epochs)
    
    # Save Model
    # We save the decoder separately for generation
    print(f"Saving model to {args.job_dir}...")
    
    # Save Decoder
    decoder_path = os.path.join(args.job_dir, 'decoder')
    cvae.decoder.save(decoder_path)
    
    # Save Encoder (optional)
    encoder_path = os.path.join(args.job_dir, 'encoder')
    cvae.encoder.save(encoder_path)
    
    print("Training complete.")

if __name__ == "__main__":
    main()
