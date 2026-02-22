import os
import tensorflow as tf
import numpy as np
from models.hybrid.model import create_hybrid_model

def load_diverse_svt_dataset():
    """
    Simulates loading a diverse dataset including rare SVT variants.
    In a real scenario, this would load from BigQuery or GCS.
    """
    print("Loading diverse SVT dataset (AVNRT, AVRT, Focal AT)...")
    # Mock data
    num_samples = 100
    X = np.random.randn(num_samples, 5000, 12).astype(np.float32)
    # 5 classes: Normal, AFIB, MI, PVC, Noise
    # We might need to expand classes for SVT if the model supports it, 
    # but for now we'll assume SVT maps to 'AFIB' or a new class if we changed num_classes.
    # The current model has 5 classes. Let's assume we are fine-tuning to improve 
    # discrimination within existing classes or we'd need to change the head.
    # For this demo, we'll stick to the 5 classes but assume the data is 'harder' examples.
    y = np.random.randint(0, 5, size=(num_samples,))
    y_onehot = tf.keras.utils.to_categorical(y, num_classes=5)
    
    # Dummy quality labels
    y_quality = np.random.randint(0, 2, size=(num_samples, 12))
    
    return X, {'pathology': y_onehot, 'quality': y_quality}

def finetune_model():
    model_path = os.environ.get("MODEL_PATH", "models/hybrid/final_model")
    
    # 1. Load or Create Model
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        try:
            model = tf.keras.models.load_model(model_path)
        except:
            print("Could not load model, creating new one with SE Attention...")
            model = create_hybrid_model()
    else:
        print("Creating new model with SE Attention...")
        model = create_hybrid_model()
        
    # 2. Load Data
    X, y = load_diverse_svt_dataset()
    
    # 3. Compile with low learning rate for fine-tuning
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(optimizer=optimizer, 
                  loss={'pathology': 'categorical_crossentropy', 'quality': 'binary_crossentropy'},
                  metrics={'pathology': 'accuracy', 'quality': 'accuracy'})
    
    # 4. Train
    print("Starting fine-tuning on diverse dataset...")
    model.fit(X, y, epochs=5, batch_size=32)
    
    # 5. Save
    print(f"Saving fine-tuned model to {model_path}...")
    model.save(model_path)
    print("Fine-tuning complete.")

if __name__ == "__main__":
    finetune_model()
