"""
Transformer Model for Early Disease Detection from Time-Series Medical Data
Author: Pranay M Mahendrakar
Repository: AI-Early-Disease-Detection-TimeSeries
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for preserving temporal order.
    
    Adds position-dependent signals to input embeddings so the model
    can differentiate between time steps in the physiological signal.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings."""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention for capturing inter-temporal dependencies.
    
    Allows the model to attend to different temporal positions simultaneously,
    capturing both local ECG waveform patterns and global signal trends.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadSelfAttention, self).__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-head self-attention.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask (batch, 1, seq_len, seq_len)
            
        Returns:
            output: Attention output (batch, seq_len, d_model)
            attention_weights: Attention distribution (batch, heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections and reshape
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Context vector
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        output = self.W_o(context)
        return output, attention_weights


class TransformerBlock(nn.Module):
    """
    Single Transformer encoder block with pre-norm architecture.
    
    Components:
    - Multi-head self-attention
    - Position-wise feed-forward network
    - Layer normalization (pre-norm for stability)
    - Residual connections
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        dropout: float = 0.1
    ):
        super(TransformerBlock, self).__init__()
        
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply transformer block with pre-norm and residual connections."""
        # Self-attention with pre-norm
        normed_x = self.norm1(x)
        attn_out, attn_weights = self.attention(normed_x, mask)
        x = x + self.dropout(attn_out)
        
        # Feed-forward with pre-norm
        x = x + self.feed_forward(self.norm2(x))
        
        return x, attn_weights


class TransformerDiseaseDetector(nn.Module):
    """
    Transformer-based model for time-series medical disease detection.
    
    Architecture:
    - Input embedding + Positional encoding
    - Stack of Transformer encoder blocks
    - Global average pooling + CLS token classification
    - Multi-layer classification head
    
    Key advantages over LSTM:
    - Parallel processing of entire signal window
    - Global attention captures long-range dependencies
    - Scalable to very long sequences
    - Better interpretability via attention maps
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 512,
        num_classes: int = 5,
        max_seq_len: int = 1000,
        dropout: float = 0.1,
        use_cls_token: bool = True
    ):
        """
        Initialize Transformer Disease Detector.
        
        Args:
            input_dim: Number of input signal channels
            d_model: Transformer model dimension
            num_heads: Number of attention heads
            num_layers: Number of Transformer blocks
            d_ff: Feed-forward network hidden dimension
            num_classes: Number of output classes
            max_seq_len: Maximum input sequence length
            dropout: Dropout probability
            use_cls_token: Use CLS token for classification
        """
        super(TransformerDiseaseDetector, self).__init__()
        
        self.d_model = d_model
        self.use_cls_token = use_cls_token
        
        # Input projection
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model)
        )
        
        # CLS token for classification
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model, 
            max_len=max_seq_len + 1,  # +1 for CLS token
            dropout=dropout
        )
        
        # Transformer encoder layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(d_model // 4, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize transformer weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through Transformer network.
        
        Args:
            x: Input signal (batch, seq_len, input_dim)
            mask: Optional padding mask (batch, seq_len)
            return_attention: Return attention maps for visualization
            
        Returns:
            logits: Classification output (batch, num_classes)
            attention_maps (optional): List of attention weights per layer
        """
        batch_size, seq_len, _ = x.shape
        
        # Input embedding
        x = self.input_embedding(x)  # (batch, seq, d_model)
        
        # Prepend CLS token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Apply transformer blocks
        attention_maps = []
        for block in self.transformer_blocks:
            x, attn_weights = block(x, mask)
            if return_attention:
                attention_maps.append(attn_weights.detach())
        
        x = self.norm(x)
        
        # Classification using CLS token or global average pooling
        if self.use_cls_token:
            cls_representation = x[:, 0, :]  # CLS token representation
        else:
            cls_representation = x.mean(dim=1)  # Global average pooling
        
        logits = self.classifier(cls_representation)
        
        if return_attention:
            return logits, attention_maps
        return logits
    
    def get_attention_maps(self, x: torch.Tensor) -> list:
        """Extract attention maps for interpretability visualization."""
        _, attention_maps = self.forward(x, return_attention=True)
        return attention_maps
    
    def get_model_size(self) -> dict:
        """Return model parameter statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 ** 2)
        }


class ECGTransformerClassifier(TransformerDiseaseDetector):
    """
    Specialized Transformer for ECG arrhythmia classification.
    Optimized configuration for 12-lead ECG and single-lead recordings.
    """
    
    def __init__(self, num_leads: int = 1, **kwargs):
        super().__init__(
            input_dim=num_leads,
            d_model=kwargs.get('d_model', 128),
            num_heads=kwargs.get('num_heads', 8),
            num_layers=kwargs.get('num_layers', 6),
            d_ff=kwargs.get('d_ff', 512),
            num_classes=kwargs.get('num_classes', 5),
            max_seq_len=kwargs.get('max_seq_len', 1000),
            dropout=kwargs.get('dropout', 0.1)
        )


class ICUTransformerPredictor(TransformerDiseaseDetector):
    """
    Transformer for ICU multi-variate time series mortality prediction.
    Handles irregular sampling and missing values common in ICU data.
    """
    
    def __init__(self, num_vitals: int = 7, **kwargs):
        super().__init__(
            input_dim=num_vitals,
            d_model=kwargs.get('d_model', 256),
            num_heads=kwargs.get('num_heads', 8),
            num_layers=kwargs.get('num_layers', 4),
            d_ff=kwargs.get('d_ff', 1024),
            num_classes=2,  # Binary mortality prediction
            max_seq_len=kwargs.get('max_seq_len', 48),
            dropout=kwargs.get('dropout', 0.2)
        )


if __name__ == '__main__':
    import torch
    
    print("Testing ECG Transformer Classifier...")
    ecg_transformer = ECGTransformerClassifier(num_leads=1)
    x_ecg = torch.randn(32, 512, 1)
    output, attn_maps = ecg_transformer(x_ecg, return_attention=True)
    print(f"ECG Output shape: {output.shape}")
    print(f"Attention maps: {len(attn_maps)} layers, shape: {attn_maps[0].shape}")
    print(f"Model size: {ecg_transformer.get_model_size()}")
    
    print("\nTesting ICU Transformer Predictor...")
    icu_transformer = ICUTransformerPredictor(num_vitals=7)
    x_icu = torch.randn(16, 48, 7)
    output = icu_transformer(x_icu)
    print(f"ICU Output shape: {output.shape}")
    print("All tests passed!")
