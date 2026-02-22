import tensorflow as tf
from tensorflow.keras import layers, models, Input
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger("StackingEnsemble")

class AdvancedStackingEnsemble:
    """
    Multi-modal Stacking Ensemble for ECG Classification.
    Combines CNN (Visual), LSTM (Temporal), and Transformer (Attention) 
    with a Gradient Boosting Meta-Learner.
    """
    
    def __init__(self, num_classes: int = 5, input_shape: Tuple[int, int] = (5000, 12)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.base_models = {}
        # Meta-learner: Gradient Boosting
        self.meta_learner = GradientBoostingClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=3,
            random_state=42
        )
        self.is_fitted = False

    def build_cnn(self) -> models.Model:
        """CNN for local pattern recognition (morphology)."""
        inputs = Input(shape=self.input_shape)
        x = layers.Conv1D(32, 7, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        
        x = layers.Conv1D(64, 5, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        
        x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
        x = layers.GlobalAveragePooling1D()(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        return models.Model(inputs, outputs, name="CNN_ResNet")

    def build_lstm_attention(self) -> models.Model:
        """Bi-LSTM + Attention for temporal sequence analysis."""
        inputs = Input(shape=self.input_shape)
        # Downsample to reduce sequence length (5000 -> 500)
        x = layers.Conv1D(32, 10, strides=10, activation='relu')(inputs)
        
        # Bidirectional LSTM
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
        
        # Self-Attention
        query = layers.Dense(64)(x)
        value = layers.Dense(64)(x)
        # (Batch, Steps, Features)
        attention = layers.Attention()([query, value])
        
        x = layers.GlobalAveragePooling1D()(attention)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        return models.Model(inputs, outputs, name="LSTM_Attn")

    def build_transformer(self) -> models.Model:
        """Vision Transformer (ViT) style encoder for complex correlations."""
        inputs = Input(shape=self.input_shape)
        # Patch projection (Conv1D as embedding)
        x = layers.Conv1D(64, 20, strides=20)(inputs) # (Batch, 250, 64)
        
        # Positional encoding would go here (omitted for brevity)
        
        # Transformer Block 1
        x1 = layers.MultiHeadAttention(num_heads=2, key_dim=64)(x, x)
        x = layers.Add()([x, x1])
        x = layers.LayerNormalization()(x)
        
        x2 = layers.Dense(64, activation='relu')(x)
        x = layers.Add()([x, x2])
        x = layers.LayerNormalization()(x)
        
        x = layers.GlobalAveragePooling1D()(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        return models.Model(inputs, outputs, name="Transformer")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 5, batch_size: int = 32):
        """
        Trains the ensemble using a Hold-out Blending strategy.
        """
        logger.info("Initializing base models...")
        self.base_models['cnn'] = self.build_cnn()
        self.base_models['lstm'] = self.build_lstm_attention()
        self.base_models['transformer'] = self.build_transformer()
        
        for name, model in self.base_models.items():
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            
        # Split Data for Stacking (Level 0 / Level 1)
        # 70% for Base Models, 30% for Meta Learner
        split_idx = int(len(X_train) * 0.7)
        X_l0, X_l1 = X_train[:split_idx], X_train[split_idx:]
        y_l0, y_l1 = y_train[:split_idx], y_train[split_idx:]
        
        # 1. Train Base Models
        logger.info("Training Base Models (Level 0)...")
        for name, model in self.base_models.items():
            logger.info(f"Training {name}...")
            model.fit(X_l0, y_l0, epochs=epochs, batch_size=batch_size, verbose=1)
            
        # 2. Generate Meta-Features
        logger.info("Generating Meta-Features (Level 1)...")
        l1_preds = []
        for name, model in self.base_models.items():
            preds = model.predict(X_l1, verbose=0)
            l1_preds.append(preds)
            
        # Concatenate predictions: (N_samples, N_models * N_classes)
        meta_X = np.hstack(l1_preds)
        meta_y = np.argmax(y_l1, axis=1) # Convert one-hot to labels for sklearn
        
        # 3. Train Meta-Learner
        logger.info("Training Meta-Learner (Gradient Boosting)...")
        self.meta_learner.fit(meta_X, meta_y)
        
        self.is_fitted = True
        logger.info("Ensemble Training Complete.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Returns class probabilities from the meta-learner.
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
            
        # 1. Base Predictions
        base_preds = []
        for name, model in self.base_models.items():
            base_preds.append(model.predict(X, verbose=0))
            
        # 2. Meta Features
        meta_X = np.hstack(base_preds)
        
        # 3. Meta Prediction
        final_probs = self.meta_learner.predict_proba(meta_X)
        return final_probs

# Example Usage
if __name__ == "__main__":
    # Mock Data
    N = 100
    X = np.random.randn(N, 5000, 12).astype(np.float32)
    y = np.random.randint(0, 5, N)
    y_onehot = np.eye(5)[y]
    
    ensemble = AdvancedStackingEnsemble()
    ensemble.fit(X, y_onehot, epochs=1)
    
    preds = ensemble.predict(X[:5])
    print("Predictions shape:", preds.shape)
    print("Sample prediction:", preds[0])
