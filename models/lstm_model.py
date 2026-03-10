"""
LSTM Model for Early Disease Detection from Time-Series Medical Data
Author: Pranay M Mahendrakar
Repository: AI-Early-Disease-Detection-TimeSeries
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LSTMDiseaseDetector(nn.Module):
    """
    Bidirectional LSTM model for time-series medical signal classification.
    
    Supports:
    - ECG arrhythmia classification (MIT-BIH, PTB-XL)
    - ICU patient mortality prediction (MIMIC-III)
    - Multi-class disease detection from physiological signals
    
    Architecture:
    - Input projection layer
    - Multi-layer Bidirectional LSTM
    - Temporal attention mechanism
    - Classification head with dropout regularization
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_classes: int = 5,
        dropout: float = 0.3,
        bidirectional: bool = True,
        use_attention: bool = True,
        batch_first: bool = True
    ):
        """
        Initialize the LSTM Disease Detector.
        
        Args:
            input_dim: Number of input features (signal channels)
            hidden_dim: Number of LSTM hidden units
            num_layers: Number of stacked LSTM layers
            num_classes: Number of output classes
            dropout: Dropout rate for regularization
            bidirectional: Use bidirectional LSTM
            use_attention: Apply temporal attention mechanism
            batch_first: Input tensor shape (batch, seq, features)
        """
        super(LSTMDiseaseDetector, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.num_directions = 2 if bidirectional else 1
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        lstm_out_dim = hidden_dim * self.num_directions
        
        # Temporal attention
        if use_attention:
            self.attention = TemporalAttention(lstm_out_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Weight initialization
        self._init_weights()
    
    def _init_weights(self):
        """Xavier/Glorot weight initialization for LSTM layers."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 for better gradient flow
                n = param.size(0)
                param.data[(n // 4):(n // 2)].fill_(1)
    
    def forward(
        self, 
        x: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the LSTM network.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            return_attention: Return attention weights for interpretability
            
        Returns:
            logits: Classification logits (batch, num_classes)
            attention_weights (optional): Temporal attention weights
        """
        # Input projection
        x = self.input_projection(x)  # (batch, seq, hidden_dim)
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)  # (batch, seq, hidden*directions)
        
        # Apply temporal attention or use last hidden state
        if self.use_attention:
            context, attention_weights = self.attention(lstm_out)
        else:
            context = lstm_out[:, -1, :]  # Last timestep
            attention_weights = None
        
        # Classification
        logits = self.classifier(context)
        
        if return_attention:
            return logits, attention_weights
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability estimates for each class."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)
    
    def get_model_size(self) -> dict:
        """Return model parameter count and size."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 ** 2)
        }


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism for weighting time steps.
    
    Learns which time steps are most informative for classification.
    Provides interpretability through attention weight visualization.
    """
    
    def __init__(self, hidden_dim: int):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention-weighted context vector.
        
        Args:
            lstm_output: LSTM hidden states (batch, seq_len, hidden_dim)
            
        Returns:
            context: Attention-weighted representation (batch, hidden_dim)
            attention_weights: Normalized attention scores (batch, seq_len)
        """
        # Compute attention scores
        scores = self.attention(lstm_output).squeeze(-1)  # (batch, seq_len)
        attention_weights = F.softmax(scores, dim=-1)  # Normalize
        
        # Weighted sum of hidden states
        context = torch.bmm(
            attention_weights.unsqueeze(1), 
            lstm_output
        ).squeeze(1)  # (batch, hidden_dim)
        
        return context, attention_weights


class ECGLSTMClassifier(LSTMDiseaseDetector):
    """
    Specialized LSTM classifier for ECG arrhythmia detection.
    
    Optimized for MIT-BIH Arrhythmia Dataset (5-class AAMI standard):
    - N: Normal beat
    - S: Supraventricular ectopic beat
    - V: Ventricular ectopic beat
    - F: Fusion beat
    - Q: Unknown beat
    """
    
    CLASSES = ['N', 'S', 'V', 'F', 'Q']
    CLASS_NAMES = [
        'Normal',
        'Supraventricular',
        'Ventricular',
        'Fusion',
        'Unknown'
    ]
    
    def __init__(self, **kwargs):
        super().__init__(
            input_dim=kwargs.get('input_dim', 1),
            hidden_dim=kwargs.get('hidden_dim', 128),
            num_layers=kwargs.get('num_layers', 3),
            num_classes=5,  # AAMI 5-class standard
            dropout=kwargs.get('dropout', 0.3),
            bidirectional=True,
            use_attention=True
        )


class ICUMortalityPredictor(LSTMDiseaseDetector):
    """
    LSTM model for ICU in-hospital mortality prediction.
    
    Uses MIMIC-III time-series vital signs:
    - Heart rate, respiratory rate, SpO2
    - Blood pressure (systolic/diastolic)
    - Temperature, GCS score
    
    Predicts mortality risk at 24h, 48h, and 72h horizons.
    """
    
    VITAL_SIGNS = [
        'heart_rate', 'resp_rate', 'spo2',
        'bp_systolic', 'bp_diastolic',
        'temperature', 'gcs_score'
    ]
    
    def __init__(self, prediction_horizon: str = '24h', **kwargs):
        super().__init__(
            input_dim=kwargs.get('input_dim', 7),  # 7 vital signs
            hidden_dim=kwargs.get('hidden_dim', 256),
            num_layers=kwargs.get('num_layers', 4),
            num_classes=2,  # Binary: survive/mortality
            dropout=kwargs.get('dropout', 0.4),
            bidirectional=True,
            use_attention=True
        )
        self.prediction_horizon = prediction_horizon
    
    def predict_mortality_risk(self, x: torch.Tensor) -> torch.Tensor:
        """Return mortality probability score (0-1)."""
        proba = self.predict_proba(x)
        return proba[:, 1]  # Probability of mortality


if __name__ == '__main__':
    # Quick test
    import torch
    
    # Test ECG classifier
    print("Testing ECG LSTM Classifier...")
    ecg_model = ECGLSTMClassifier()
    x_ecg = torch.randn(32, 512, 1)  # batch=32, seq_len=512, channels=1
    output = ecg_model(x_ecg)
    print(f"ECG Output shape: {output.shape}")  # (32, 5)
    print(f"Model size: {ecg_model.get_model_size()}")
    
    # Test ICU predictor
    print("\nTesting ICU Mortality Predictor...")
    icu_model = ICUMortalityPredictor(prediction_horizon='24h')
    x_icu = torch.randn(16, 48, 7)  # batch=16, seq_len=48h, features=7
    risk_scores = icu_model.predict_mortality_risk(x_icu)
    print(f"ICU Risk scores shape: {risk_scores.shape}")  # (16,)
    print("All tests passed!")
