import tensorflow as tf
from tensorflow.keras import layers, models

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def se_block(input_tensor, ratio=16):
    """Squeeze-and-Excitation Block"""
    channel_axis = -1
    filters = input_tensor.shape[channel_axis]
    se_shape = (1, filters)

    se = layers.GlobalAveragePooling1D()(input_tensor)
    se = layers.Reshape(se_shape)(se)
    se = layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = layers.Multiply()([input_tensor, se])
    return x

def create_hybrid_model(input_shape=(5000, 12), num_classes=5):
    inputs = layers.Input(shape=input_shape)
    
    # --- CNN Branch (Local Features) with SE Attention ---
    # ResNet-like blocks for local feature extraction
    x = layers.Conv1D(64, 7, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = se_block(x) # Add SE Attention
    x = layers.MaxPooling1D(2)(x) # 2500
    
    x = layers.Conv1D(128, 5, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = se_block(x) # Add SE Attention
    x = layers.MaxPooling1D(2)(x) # 1250
    
    x = layers.Conv1D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = se_block(x) # Add SE Attention
    x = layers.MaxPooling1D(2)(x) # 625
    
    cnn_features = x
    
    # --- Transformer Branch (Global Context) ---
    # Treat the CNN output as a sequence of "patches"
    # Shape: (Batch, 625, 256) -> Sequence length 625, embedding dim 256
    
    projection_dim = 256
    num_heads = 4
    transformer_layers = 2
    
    # Add positional encoding
    # We can use the PatchEncoder logic, but since it's already a sequence from CNN, 
    # we just add positional embeddings to the existing vectors.
    seq_len = cnn_features.shape[1]
    positions = tf.range(start=0, limit=seq_len, delta=1)
    pos_embedding = layers.Embedding(input_dim=seq_len, output_dim=projection_dim)(positions)
    
    transformer_input = cnn_features + pos_embedding
    
    for _ in range(transformer_layers):
        # Layer Norm 1
        x1 = layers.LayerNormalization(epsilon=1e-6)(transformer_input)
        # Multi-Head Attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip Connection 1
        x2 = layers.Add()([attention_output, transformer_input])
        
        # Layer Norm 2
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP
        x3 = layers.Dense(projection_dim * 2, activation="relu")(x3)
        x3 = layers.Dense(projection_dim)(x3)
        # Skip Connection 2
        transformer_input = layers.Add()([x3, x2])
        
    transformer_features = layers.GlobalAveragePooling1D()(transformer_input)
    
    # --- Multi-Task Heads ---
    
    # Head 1: Pathology Classification (Multi-Label)
    # Using Sigmoid because a patient can have multiple conditions
    p = layers.Dense(128, activation='relu')(transformer_features)
    p = layers.Dropout(0.5)(p)
    pathology_output = layers.Dense(num_classes, activation='sigmoid', name='pathology')(p)
    
    # Head 2: Lead Quality Assessment (Per Lead)
    # Predicts probability of each lead being "Good Quality"
    # Input shape has 12 channels. We want to map features back to 12 quality scores.
    # We can use a separate dense branch.
    q = layers.Dense(64, activation='relu')(transformer_features)
    quality_output = layers.Dense(12, activation='sigmoid', name='quality')(q)
    
    model = models.Model(inputs=inputs, outputs={'pathology': pathology_output, 'quality': quality_output}, name="Hybrid_MultiTask")
    return model
