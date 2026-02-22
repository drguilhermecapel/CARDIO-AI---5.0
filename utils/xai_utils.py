import numpy as np

class SignalInverter:
    """
    Helper class to map coordinates from a processed/resampled signal 
    back to the original raw signal.
    Useful for XAI (Explainable AI) to highlight original regions.
    """
    
    def __init__(self, transformation_metadata):
        """
        Args:
            transformation_metadata (dict): Record from 'signal_transformations' table.
                                            Must contain 'original_fs', 'target_fs'.
        """
        self.meta = transformation_metadata
        self.scale_factor = self.meta.get('scale_factor', 1.0)
        
        if self.scale_factor == 0:
            # Fallback if scale_factor not explicitly stored
            orig = self.meta.get('original_fs', 500)
            target = self.meta.get('target_fs', 500)
            self.scale_factor = target / orig if orig > 0 else 1.0

    def map_index(self, processed_idx):
        """
        Maps a single index from processed signal to original signal.
        """
        return int(processed_idx / self.scale_factor)

    def map_range(self, start_idx, end_idx):
        """
        Maps a range (start, end) from processed to original.
        """
        return (self.map_index(start_idx), self.map_index(end_idx))

    def map_saliency_map(self, saliency_array, original_length):
        """
        Resizes a saliency map (array of importance scores) back to original length.
        """
        from scipy.ndimage import zoom
        
        current_len = len(saliency_array)
        zoom_factor = original_length / current_len
        
        # Linear interpolation
        return zoom(saliency_array, zoom_factor, order=1)

# Example Usage
if __name__ == "__main__":
    # Mock metadata from BigQuery
    meta = {
        "original_fs": 360,
        "target_fs": 500,
        "scale_factor": 1.3888
    }
    
    inverter = SignalInverter(meta)
    
    # If model says index 250 (in 500Hz signal) is important:
    orig_idx = inverter.map_index(250)
    print(f"Processed Index 250 -> Original Index {orig_idx} (Expected ~180)")
