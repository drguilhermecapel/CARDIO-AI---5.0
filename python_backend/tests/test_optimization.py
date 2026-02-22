import pytest
import numpy as np
from optimization.inference_engine import OptimizedECGInference

def test_inference_latency():
    # Initialize without quantization for speed in test
    engine = OptimizedECGInference(quantize=False, use_gpu=False)
    
    # Generate random ECG
    ecg = np.random.randn(5000, 12).astype(np.float32)
    
    # First run (cold start)
    res1 = engine.predict_single(ecg)
    assert 'prediction' in res1
    
    # Second run (cached)
    res2 = engine.predict_single(ecg)
    assert res2['cached'] is True
    assert res2['latency_ms'] < res1['latency_ms'] # Cache should be faster

def test_batch_processing():
    engine = OptimizedECGInference(quantize=False, use_gpu=False)
    batch = [np.random.randn(5000, 12).astype(np.float32) for _ in range(5)]
    
    res = engine.predict_batch(batch)
    assert len(res['predictions']) == 5
    assert res['avg_latency_ms'] > 0

def test_benchmark_report():
    engine = OptimizedECGInference(quantize=False, use_gpu=False)
    report = engine.benchmark(10) # Short benchmark
    
    assert report['p99_ms'] > 0
    assert report['throughput_req_sec'] > 0
