import tensorflow as tf

def baseline_wander(x, amplitude=0.3, fs=500):
    """
    Adds a low-frequency sinusoidal baseline wander.
    Simulates patient breathing or movement.
    """
    # x shape: (length, channels)
    length = tf.shape(x)[0]
    t = tf.range(length, dtype=tf.float32) / float(fs)
    
    # Random frequency between 0.05Hz and 0.5Hz
    freq = tf.random.uniform([], 0.05, 0.5)
    # Random phase
    phase = tf.random.uniform([], 0, 2 * 3.14159)
    
    wander = amplitude * tf.math.sin(2 * 3.14159 * freq * t + phase)
    wander = tf.expand_dims(wander, -1) # Broadcast to channels
    
    return x + wander

def gaussian_noise(x, stddev=0.02):
    """
    Adds random Gaussian noise (sensor noise).
    """
    noise = tf.random.normal(tf.shape(x), stddev=stddev, dtype=x.dtype)
    return x + noise

def lead_dropout(x, prob=0.1):
    """
    Randomly sets some leads (channels) to zero.
    Simulates electrode disconnection.
    """
    # x shape: (length, channels)
    channels = tf.shape(x)[1]
    
    # Generate mask: 1 = keep, 0 = drop
    # We want to ensure we don't drop ALL leads, but for simplicity:
    mask = tf.random.uniform((channels,), dtype=tf.float32) > prob
    mask = tf.cast(mask, x.dtype)
    
    # If all are dropped (rare), keep original
    # logic: if sum(mask) == 0, return x, else x * mask
    # TF conditional is a bit heavy, let's just allow it or ensure prob is low.
    
    return x * mask

def time_stretch(x, rate=0.15):
    """
    Simulates Heart Rate Variation by stretching/compressing the signal.
    Uses linear interpolation.
    """
    # x shape: (length, channels)
    original_len = tf.shape(x)[0]
    channels = tf.shape(x)[1]
    
    # Random factor: 1.0 +/- rate
    factor = 1.0 + tf.random.uniform([], -rate, rate)
    new_len = tf.cast(tf.cast(original_len, tf.float32) * factor, tf.int32)
    
    # Resize requires 3D (batch, height, width) or 4D. 
    # We treat (length, channels) as (length, channels, 1) image? 
    # tf.image.resize expects [batch, height, width, channels]
    # Let's map: length -> height, 1 -> width, channels -> channels
    x_expanded = tf.expand_dims(x, 0) # (1, length, channels)
    
    # Resize
    x_resized = tf.image.resize(x_expanded, [new_len, channels])
    x_resized = tf.squeeze(x_resized, 0)
    
    # Crop or Pad back to original length to maintain tensor shape for batching
    if new_len > original_len:
        # Crop center
        start = (new_len - original_len) // 2
        x_final = x_resized[start:start+original_len, :]
    else:
        # Pad center
        pad_total = original_len - new_len
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        x_final = tf.pad(x_resized, [[pad_left, pad_right], [0, 0]])
        
    return x_final

def augment_ecg(x, label):
    """
    Applies a pipeline of augmentations stochastically.
    Input:
        x: ECG signal tensor (length, channels)
        label: Target label
    Output:
        Augmented x, label
    """
    # 1. Baseline Wander (Probability 50%)
    if tf.random.uniform([]) < 0.5:
        x = baseline_wander(x)
        
    # 2. Gaussian Noise (Probability 40%)
    if tf.random.uniform([]) < 0.4:
        x = gaussian_noise(x)
        
    # 3. Lead Dropout (Probability 20%)
    if tf.random.uniform([]) < 0.2:
        x = lead_dropout(x)
        
    # 4. Time Stretch / HR Variation (Probability 30%)
    if tf.random.uniform([]) < 0.3:
        x = time_stretch(x)
        
    return x, label
