import time
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor

class Benchmark:
    """
    Performance Benchmark for CardioAI Nexus.
    """
    def __init__(self):
        self.latencies = []
        self.errors = 0
        self.lock = threading.Lock()

    def mock_inference(self, data):
        """Simulate inference time (including network + compute)."""
        # Simulate distribution: mostly fast (50ms), some slow (200ms)
        delay = np.random.gamma(shape=2.0, scale=0.05) # Mean 100ms
        time.sleep(delay)
        return True

    def run_stress_test(self, total_requests=1000, concurrency=50):
        print(f"Starting Stress Test: {total_requests} requests, {concurrency} threads.")
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(self._worker) for _ in range(total_requests)]
            for future in futures:
                future.result()
                
        total_time = time.time() - start_time
        self.generate_report(total_time, total_requests)

    def _worker(self):
        start = time.time()
        try:
            self.mock_inference(None)
            latency = (time.time() - start) * 1000
            with self.lock:
                self.latencies.append(latency)
        except Exception:
            with self.lock:
                self.errors += 1

    def generate_report(self, total_time, total_requests):
        latencies = np.array(self.latencies)
        throughput = total_requests / total_time
        
        print("\n# Performance Benchmark Report")
        print(f"Total Requests: {total_requests}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Throughput: {throughput:.2f} req/sec")
        print(f"Errors: {self.errors}")
        print("\n## Latency Metrics (ms)")
        print(f"Mean: {np.mean(latencies):.2f}")
        print(f"P50 (Median): {np.percentile(latencies, 50):.2f}")
        print(f"P95: {np.percentile(latencies, 95):.2f}")
        print(f"P99: {np.percentile(latencies, 99):.2f}")
        print(f"Max: {np.max(latencies):.2f}")
        
        if np.percentile(latencies, 99) > 2000:
            print("\n[FAIL] P99 Latency > 2000ms")
        else:
            print("\n[PASS] P99 Latency < 2000ms")

if __name__ == "__main__":
    benchmark = Benchmark()
    benchmark.run_stress_test(total_requests=1000, concurrency=20)
