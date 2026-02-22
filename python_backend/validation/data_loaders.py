import os
import numpy as np
import pandas as pd
import wfdb
import ast
from typing import Tuple, List, Dict, Any

class ECGDatasetLoader:
    """
    Base class for ECG Dataset Loaders.
    """
    def load_data(self, limit: int = None) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
        """
        Returns:
            X: Signal data (N, Leads, Length)
            Y: Labels (N, Classes) or (N,)
            Meta: Metadata DataFrame
        """
        raise NotImplementedError

class PTBXLLoader(ECGDatasetLoader):
    """
    Loader for PTB-XL Dataset (PhysioNet).
    """
    def __init__(self, data_path: str, sampling_rate: int = 500):
        self.data_path = data_path
        self.sampling_rate = sampling_rate
        
    def load_data(self, limit: int = None) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
        # Load CSV
        csv_path = os.path.join(self.data_path, 'ptbxl_database.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"PTB-XL database file not found at {csv_path}")
            
        df = pd.read_csv(csv_path, index_col='ecg_id')
        df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
        
        # Filter for 12-lead (usually all are 12-lead in PTB-XL)
        
        if limit:
            df = df.head(limit)
            
        data = wfdb.io.dl_database('ptb-xl', self.data_path) # This downloads if not present? No, dl_database is different.
        # We assume data is present or we use a generator
        
        # Load signals
        X = self._load_raw_data(df)
        
        return X, df

    def _load_raw_data(self, df):
        if self.sampling_rate == 100:
            data = [wfdb.rdsamp(os.path.join(self.data_path, f)) for f in df.filename_lr]
        else:
            data = [wfdb.rdsamp(os.path.join(self.data_path, f)) for f in df.filename_hr]
            
        data = np.array([signal for signal, meta in data])
        return data

class MockPTBXLLoader(ECGDatasetLoader):
    """
    Generates synthetic data mimicking PTB-XL structure for testing/validation
    when real data is not available.
    """
    def __init__(self, n_samples: int = 100):
        self.n_samples = n_samples
        
    def load_data(self, limit: int = None) -> Tuple[np.ndarray, pd.DataFrame]:
        n = limit if limit else self.n_samples
        
        # Synthetic Signals (12 leads, 5000 samples)
        X = np.random.randn(n, 5000, 12) # PTB-XL is (Samples, Length, Leads) usually, we need to transpose later
        
        # Synthetic Metadata
        data = {
            'ecg_id': range(1, n+1),
            'age': np.random.randint(20, 90, n),
            'sex': np.random.choice([0, 1], n), # 0: M, 1: F
            'scp_codes': [{'NORM': 100} if i % 2 == 0 else {'IMI': 100} for i in range(n)], # Alternating Normal/Inferior MI
            'strat_fold': np.random.randint(1, 11, n)
        }
        df = pd.DataFrame(data).set_index('ecg_id')
        
        return X, df

class MITBIHLoader(ECGDatasetLoader):
    """
    Loader for MIT-BIH Arrhythmia Database.
    Note: This is 2-lead, so it requires adaptation for 12-lead models (e.g. padding or specific model).
    """
    def __init__(self, data_path: str):
        self.data_path = data_path
        
    def load_data(self, limit: int = None):
        # Implementation for MIT-BIH loading
        # ...
        pass
