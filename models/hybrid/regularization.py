import tensorflow as tf

def mixup(ds_one, ds_two, alpha=0.2):
    """
    Applies Mixup augmentation to a pair of datasets.
    ds_one, ds_two: Datasets yielding (x, y) or (x, y, w)
    """
    # Unpack
    x1 = ds_one[0]
    y1 = ds_one[1]
    w1 = ds_one[2] if len(ds_one) > 2 else None
    
    x2 = ds_two[0]
    y2 = ds_two[1]
    w2 = ds_two[2] if len(ds_two) > 2 else None
    
    # Sample lambda from Beta distribution
    # TF doesn't have direct Beta sampling, use Gamma
    # Beta(a, b) = Gamma(a) / (Gamma(a) + Gamma(b))
    dist = tf.random.gamma(shape=[], alpha=alpha)
    dist_b = tf.random.gamma(shape=[], alpha=alpha)
    lam = dist / (dist + dist_b)
    
    # Mix Inputs
    x = lam * x1 + (1 - lam) * x2
    
    # Mix Labels (Pathology only, Quality is usually per-sample)
    # y is a dict {'pathology': ..., 'quality': ...}
    y_path_1 = y1['pathology']
    y_path_2 = y2['pathology']
    
    y_path = lam * y_path_1 + (1 - lam) * y_path_2
    
    # For quality, we can mix or just take one. Let's mix.
    y_qual_1 = y1['quality']
    y_qual_2 = y2['quality']
    y_qual = lam * y_qual_1 + (1 - lam) * y_qual_2
    
    y_mixed = {'pathology': y_path, 'quality': y_qual}
    
    if w1 is not None and w2 is not None:
        w = lam * w1 + (1 - lam) * w2
        return (x, y_mixed, w)
    
    return (x, y_mixed)

def focal_loss(gamma=2.0, alpha=0.25):
    """
    Binary Focal Loss for imbalanced datasets.
    FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
    """
    def focal_loss_fixed(y_true, y_pred):
        # Clip to prevent NaN
        epsilon = 1.e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Calculate p_t
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        
        # Calculate alpha_t
        alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        
        # Calculate cross entropy
        cross_entropy = -tf.math.log(p_t)
        
        # Calculate Focal Loss
        loss = alpha_t * tf.pow(1 - p_t, gamma) * cross_entropy
        
        return tf.reduce_mean(loss)
        
    return focal_loss_fixed

def label_smoothing_loss(smoothing=0.1):
    """
    Binary Crossentropy with Label Smoothing.
    """
    def loss(y_true, y_pred):
        return tf.keras.losses.binary_crossentropy(y_true, y_pred, label_smoothing=smoothing)
    return loss
