import argparse
import os
import tensorflow as tf
import numpy as np
from google.cloud import aiplatform
from model import create_hybrid_model

from regularization import mixup, focal_loss, label_smoothing_loss

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--model_dir', type=str, required=True, help='GCS location to export model')
    parser.add_argument('--experiment_name', type=str, default='ecg-hybrid-experiment')
    parser.add_argument('--run_name', type=str, default='run-1')
    
    # Regularization Flags
    parser.add_argument('--use_mixup', action='store_true', help='Enable Mixup augmentation')
    parser.add_argument('--mixup_alpha', type=float, default=0.2)
    parser.add_argument('--loss_type', type=str, default='weighted_bce', choices=['weighted_bce', 'focal', 'label_smoothing'])
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    
    # Fairness Flags
    parser.add_argument('--group_weights', type=str, default=None, help='JSON string of weights per group (e.g. {"Male": 1.0, "Female": 2.0})')
    
    return parser.parse_args()

def load_data(batch_size=32, use_mixup=False, mixup_alpha=0.2, group_weights=None):
    # MOCK DATA LOADER
    print("Loading mock data...")
    num_samples = 1000
    signal_len = 5000
    channels = 12
    num_classes = 5
    
    X = np.random.randn(num_samples, signal_len, channels).astype(np.float32)
    y_pathology = np.random.randint(0, 2, size=(num_samples, num_classes)).astype(np.float32)
    y_quality = np.random.randint(0, 2, size=(num_samples, channels)).astype(np.float32)
    
    # Mock Demographics for Fairness
    sex = np.random.choice(['Male', 'Female'], num_samples)
    
    # Calculate Sample Weights
    sample_weights = np.ones(num_samples, dtype=np.float32)
    if group_weights:
        print(f"Applying fairness weights: {group_weights}")
        weights_dict = json.loads(group_weights)
        for s in ['Male', 'Female']:
            if s in weights_dict:
                # In real app, match patient_id to demographics
                mask = (sex == s)
                sample_weights[mask] = weights_dict[s]
    
    ds = tf.data.Dataset.from_tensor_slices((X, {'pathology': y_pathology, 'quality': y_quality}, sample_weights))
    ds = ds.shuffle(1000)
    
    if use_mixup:
        # Create two copies to mix
        ds_one = ds.shuffle(1000)
        ds_two = ds.shuffle(1000)
        ds = tf.data.Dataset.zip((ds_one, ds_two))
        ds = ds.map(lambda d1, d2: mixup(d1, d2, alpha=mixup_alpha), num_parallel_calls=tf.data.AUTOTUNE)
    
    ds = ds.batch(batch_size)
    return ds

def main():
    args = get_args()
    
    # Initialize Vertex AI Experiment
    aiplatform.init(experiment=args.experiment_name)
    aiplatform.start_run(args.run_name)
    
    # Log Parameters
    params = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "architecture": "Hybrid Multi-Task",
        "loss_type": args.loss_type,
        "use_mixup": args.use_mixup,
        "fairness_weights": args.group_weights
    }
    aiplatform.log_params(params)
    
    print(f"Training Hybrid Multi-Task Model for {args.epochs} epochs...")
    print(f"Regularization: Mixup={args.use_mixup}, Loss={args.loss_type}")
    
    dataset = load_data(
        batch_size=args.batch_size, 
        use_mixup=args.use_mixup, 
        mixup_alpha=args.mixup_alpha,
        group_weights=args.group_weights
    )
    
    # Configure Loss Function
    if args.loss_type == 'focal':
        pathology_loss = focal_loss(gamma=args.focal_gamma)
    elif args.loss_type == 'label_smoothing':
        pathology_loss = label_smoothing_loss(smoothing=args.label_smoothing)
    else:
        # Default Weighted BCE
        pathology_weights = [1.0, 2.0, 5.0, 1.0, 0.5]
        pathology_loss = get_weighted_loss(pathology_weights)
    
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = create_hybrid_model(input_shape=(5000, 12), num_classes=5)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss={
                'pathology': pathology_loss,
                'quality': 'binary_crossentropy'
            },
            loss_weights={
                'pathology': 1.0, 
                'quality': 0.3
            },
            metrics={
                'pathology': ['AUC', 'Precision', 'Recall'],
                'quality': ['accuracy']
            }
        )
    
    # Callbacks
    tensorboard_dir = os.path.join(args.model_dir, 'logs')
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir, histogram_freq=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.model_dir, 'checkpoints/cp-{epoch:04d}.ckpt'),
            save_weights_only=True,
            verbose=1
        )
    ]
    
    # Custom Callback for Vertex AI Logging (Aggregated)
    class VertexLogger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            # Log key metrics
            aiplatform.log_metrics({
                "loss": logs['loss'],
                "pathology_loss": logs['pathology_loss'],
                "pathology_auc": logs['pathology_auc'],
                "quality_accuracy": logs['quality_accuracy']
            })
            
    callbacks.append(VertexLogger())
    
    history = model.fit(
        dataset, 
        epochs=args.epochs,
        callbacks=callbacks
    )
    
    # Save Final Model
    print(f"Saving model to {args.model_dir}...")
    model.save(args.model_dir)
    
    aiplatform.end_run()
    print("Training complete.")

if __name__ == "__main__":
    main()
