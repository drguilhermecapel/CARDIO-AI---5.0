import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger("SSM_CPC")

class S4Layer(nn.Module):
    """
    Simplified Structured State Space Sequence (S4) Layer.
    Implements the continuous-time state space model:
        x'(t) = A x(t) + B u(t)
        y(t)  = C x(t) + D u(t)
    Discretized using Zero-Order Hold (ZOH) or Bilinear transform.
    
    This implementation uses a diagonal approximation (S4D) and a 
    Python-loop recurrence for compatibility (slow but functional without CUDA kernels).
    """
    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=False):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.transposed = transposed
        
        # Parameters
        # A: Diagonal initialization (Real part < 0 for stability)
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(d_model, d_state)))
        self.A_imag = nn.Parameter(torch.pi * torch.arange(d_state).float().repeat(d_model, 1))
        
        # B: Input projection
        self.B = nn.Parameter(torch.randn(d_model, d_state))
        
        # C: Output projection (Complex)
        self.C = nn.Parameter(torch.randn(d_model, d_state, 2)) 
        
        # D: Skip connection
        self.D = nn.Parameter(torch.randn(d_model))
        
        # Step size (log_dt)
        self.log_dt = nn.Parameter(torch.log(torch.tensor(0.01)))
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, u):
        """
        Args:
            u: Input tensor. Shape (Batch, Length, Dim) if transposed=False.
        """
        if self.transposed:
            u = u.transpose(1, 2) # (B, D, L) -> (B, L, D) for processing
            
        B_batch, L, D = u.shape
        
        # Materialize parameters
        dt = torch.exp(self.log_dt)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag # (D, N)
        
        # Discretize (Euler for simplicity in this demo implementation)
        # x_t = (1 + A*dt) * x_{t-1} + (B*dt) * u_t
        # This is a naive recurrence. Real S4 uses FFT convolution.
        
        # Precompute
        A_discrete = 1 + A * dt # (D, N)
        B_discrete = self.B.complex() * dt.unsqueeze(-1) # (D, N)
        
        # Initialize state
        h = torch.zeros(B_batch, D, self.d_state, device=u.device, dtype=torch.cfloat)
        ys = []
        
        C_complex = torch.view_as_complex(self.C) # (D, N)
        
        # Recurrence Scan
        for t in range(L):
            u_t = u[:, t, :].unsqueeze(-1) # (B, D, 1)
            
            # Update state: h = A_bar * h + B_bar * u
            # Broadcasting: (D, N) * (B, D, N) -> (B, D, N)
            h = (A_discrete.unsqueeze(0) * h) + (B_discrete.unsqueeze(0) * u_t)
            
            # Output: y = C * h
            # Sum over state dimension N
            y_t = torch.sum(h * C_complex.unsqueeze(0), dim=-1).real # (B, D)
            ys.append(y_t)
            
        y_stack = torch.stack(ys, dim=1) # (B, L, D)
        
        # Add D skip connection
        y_stack = y_stack + (self.D.unsqueeze(0).unsqueeze(0) * u)
        
        # Norm & Dropout
        out = self.norm(self.dropout(y_stack))
        
        if self.transposed:
            out = out.transpose(1, 2)
            
        return out

class CPCModel(nn.Module):
    """
    Contrastive Predictive Coding (CPC) Model.
    Uses an SSM (S4) as the autoregressive context network to capture long-term dependencies.
    """
    def __init__(self, input_channels=12, encoder_dim=64, context_dim=128, prediction_steps=12):
        super().__init__()
        self.prediction_steps = prediction_steps
        self.context_dim = context_dim
        
        # 1. Encoder (g_enc): Raw ECG -> Latent Z
        # Downsamples the signal
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, encoder_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(encoder_dim),
            nn.ReLU(),
            nn.Conv1d(encoder_dim, encoder_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(encoder_dim),
            nn.ReLU(),
            nn.Conv1d(encoder_dim, context_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(context_dim),
            nn.ReLU()
        )
        
        # 2. Autoregressor (g_ar): Latent Z -> Context C
        # S4 Layer for long-range context
        self.autoregressor = S4Layer(d_model=context_dim, d_state=64, transposed=False)
        
        # 3. Predictors (W_k): Context C_t -> Predicted Z_{t+k}
        # Linear mappings for future steps
        self.predictors = nn.ModuleList([
            nn.Linear(context_dim, context_dim) for _ in range(prediction_steps)
        ])

    def forward(self, x):
        """
        Args:
            x: (Batch, Channels, Length)
        Returns:
            z: Latent sequence (Batch, Length_enc, Dim)
            c: Context sequence (Batch, Length_enc, Dim)
        """
        # Encode
        z = self.encoder(x) # (B, Dim, L_enc)
        z = z.transpose(1, 2) # (B, L_enc, Dim)
        
        # Autoregress
        c = self.autoregressor(z) # (B, L_enc, Dim)
        
        return z, c

    def compute_loss(self, z, c):
        """
        Computes InfoNCE Loss.
        """
        batch_size, length, dim = z.shape
        total_loss = 0
        steps_averaged = 0
        
        # Sample random time steps to compute loss on (to save memory/compute)
        # Or compute on all valid steps
        # Here we sample 10 random valid start points
        valid_length = length - self.prediction_steps
        if valid_length <= 0:
            return torch.tensor(0.0, requires_grad=True)
            
        t_samples = torch.randint(0, valid_length, (min(10, valid_length),))
        
        for k in range(1, self.prediction_steps + 1):
            W_k = self.predictors[k-1]
            
            for t in t_samples:
                # Context at t
                c_t = c[:, t, :] # (B, D)
                
                # True latent at t+k
                z_tk = z[:, t+k, :] # (B, D)
                
                # Predicted latent
                z_pred = W_k(c_t) # (B, D)
                
                # InfoNCE: Logits = z_pred @ z_tk.T
                # Positive samples are on diagonal
                # Negatives are other batch samples
                logits = torch.mm(z_pred, z_tk.t()) # (B, B)
                
                labels = torch.arange(batch_size, device=z.device)
                loss = F.cross_entropy(logits, labels)
                
                total_loss += loss
                steps_averaged += 1
                
        return total_loss / max(1, steps_averaged)

# Example Usage
if __name__ == "__main__":
    # Mock Data
    x = torch.randn(4, 12, 1000) # (Batch, Channels, Length)
    model = CPCModel(input_channels=12, context_dim=32, prediction_steps=4)
    
    z, c = model(x)
    loss = model.compute_loss(z, c)
    
    print(f"Z shape: {z.shape}")
    print(f"C shape: {c.shape}")
    print(f"InfoNCE Loss: {loss.item()}")
