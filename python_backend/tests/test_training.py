import pytest
import os
import numpy as np
from training.train_pipeline import TrainingPipeline
from training.data_loader import ECGDataLoader

def test_data_loader_synthetic():
    loader = ECGDataLoader()
    X, y, meta = loader.generate_synthetic_dataset(n_samples=10)
    
    assert X.shape == (10, 5000, 12)
    assert y.shape[0] == 10
    assert len(meta) == 10
    assert 'ethnicity' in meta.columns

def test_training_pipeline_execution(tmpdir):
    # Use temp dir for output
    out_dir = str(tmpdir.mkdir("models"))
    pipeline = TrainingPipeline(output_dir=out_dir)
    
    # Run with minimal data/epochs
    metrics = pipeline.run(n_samples=20, epochs=1)
    
    assert os.path.exists(os.path.join(out_dir, "cnn_model.h5"))
    assert os.path.exists(os.path.join(out_dir, "validation_report.md"))
    assert os.path.exists(os.path.join(out_dir, "confusion_matrix.png"))
    assert len(metrics) > 0
