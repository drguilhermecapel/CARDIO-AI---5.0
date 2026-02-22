import os
import tensorflow as tf
import numpy as np

def create_dummy_model(output_path):
    print(f"Creating dummy model at {output_path}...")
    
    # Input: (5000, 12)
    inputs = tf.keras.Input(shape=(5000, 12), name='ecg_input')
    
    # Simple Conv1D with Attention (SE Block simulation)
    x = tf.keras.layers.Conv1D(32, 5, activation='relu')(inputs)
    
    # SE Block
    filters = 32
    se = tf.keras.layers.GlobalAveragePooling1D()(x)
    se = tf.keras.layers.Reshape((1, filters))(se)
    se = tf.keras.layers.Dense(filters // 2, activation='relu')(se)
    se = tf.keras.layers.Dense(filters, activation='sigmoid')(se)
    x = tf.keras.layers.Multiply()([x, se])
    
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.5)(x) # For MC Dropout
    
    # Output: 5 classes (Normal, AFIB, MI, PVC, Noise)
    outputs = tf.keras.layers.Dense(5, activation='softmax', name='pathology')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs={'pathology': outputs})
    
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    
    # Save
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
        
    model.save(output_path)
    print("Model saved.")

if __name__ == "__main__":
    model_path = os.environ.get("MODEL_PATH", "models/hybrid/final_model")
    create_dummy_model(model_path)
