import tensorflow as tf
import numpy as np
from models.augmentations import augment_ecg

def create_dataset(num_samples=100, length=5000, channels=12):
    """
    Creates a dummy tf.data.Dataset for demonstration.
    In production, this would read from TFRecords or BigQuery.
    """
    # Generate random synthetic ECGs
    X = np.random.randn(num_samples, length, channels).astype(np.float32)
    y = np.random.randint(0, 2, size=(num_samples,)).astype(np.int32)
    
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    return dataset

def train_pipeline():
    print("Setting up Training Pipeline with Augmentations...")
    
    # 1. Load Data
    ds = create_dataset()
    
    # 2. Shuffle and Batch
    ds = ds.shuffle(buffer_size=100)
    
    # 3. Apply Augmentations (ONLY on Training Data)
    # num_parallel_calls=tf.data.AUTOTUNE ensures parallel execution
    ds = ds.map(augment_ecg, num_parallel_calls=tf.data.AUTOTUNE)
    
    # 4. Batch and Prefetch
    ds = ds.batch(32)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    print("Pipeline ready. Iterating through one batch...")
    
    for batch_x, batch_y in ds.take(1):
        print(f"Batch Shape: {batch_x.shape}")
        print(f"Labels Shape: {batch_y.shape}")
        print("Augmentations applied successfully.")

if __name__ == "__main__":
    train_pipeline()
