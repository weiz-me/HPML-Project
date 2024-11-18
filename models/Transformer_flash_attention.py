import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn.flash_attn_interface import flash_attn_unpadded_func  # Import Flash Attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_model, n_heads, max_len, causal=False):
        super().__init__()

        # Assume d_v = d_k
        self.d_k = d_k
        self.n_heads = n_heads

        self.key = nn.Linear(d_model, d_k * n_heads)
        self.query = nn.Linear(d_model, d_k * n_heads)
        self.value = nn.Linear(d_model, d_k * n_heads)

        # Final linear layer
        self.fc = nn.Linear(d_k * n_heads, d_model)

        # Causal mask (for decoder self-attention)
        self.causal = causal
        if causal:
            cm = torch.tril(torch.ones(max_len, max_len))
            self.register_buffer(
                "causal_mask",
                cm.view(1, 1, max_len, max_len)
            )

    def forward(self, q, k, v, pad_mask=None):
        """
        Forward pass with Flash Attention.
        """
        # Linear projection to get q, k, v
        q = self.query(q)  # Shape: (N, T, h*d_k)
        k = self.key(k)    # Shape: (N, T, h*d_k)
        v = self.value(v)  # Shape: (N, T, h*d_k)

        N, T_q, _ = q.size()
        _, T_k, _ = k.size()

        # Reshape to prepare for multi-head attention
        q = q.view(N, T_q, self.n_heads, self.d_k).transpose(1, 2)  # Shape: (N, h, T, d_k)
        k = k.view(N, T_k, self.n_heads, self.d_k).transpose(1, 2)  # Shape: (N, h, T, d_k)
        v = v.view(N, T_k, self.n_heads, self.d_k).transpose(1, 2)  # Shape: (N, h, T, d_k)

        # Flash Attention requires padding masks (if any)
        if pad_mask is not None:
            pad_mask = pad_mask[:, None, None, :]  # Expand dims for heads and sequences

        # Apply causal mask if needed
        if self.causal:
            causal_mask = self.causal_mask[:, :, :T_q, :T_k]
            if pad_mask is not None:
                mask = pad_mask & causal_mask
            else:
                mask = causal_mask
        else:
            mask = pad_mask

        # Flash Attention computation
        # Note: Flash Attention combines softmax and attention-weighted sum
        attn_output = flash_attn_unpadded_func(q, k, v, mask=mask)

        # Reshape back to original format
        attn_output = attn_output.transpose(1, 2).contiguous().view(N, T_q, self.n_heads * self.d_k)

        # Final projection
        return self.fc(attn_output)

# EncoderBlock, DecoderBlock, Encoder, Decoder, and Transformer remain unchanged

class EncoderBlock(nn.Module):
    def __init__(self, d_k, d_model, n_heads, max_len, dropout_prob=0.1):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_k, d_model, n_heads, max_len, causal=False)
        self.ann = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout_prob),
        )
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x, pad_mask=None):
        x = self.ln1(x + self.mha(x, x, x, pad_mask))
        x = self.ln2(x + self.ann(x))
        x = self.dropout(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_k, d_model, n_heads, max_len, dropout_prob=0.1):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.mha1 = MultiHeadAttention(d_k, d_model, n_heads, max_len, causal=True)
        self.mha2 = MultiHeadAttention(d_k, d_model, n_heads, max_len, causal=False)
        self.ann = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout_prob),
        )
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, enc_output, dec_input, enc_mask=None, dec_mask=None):
        # Self-attention on decoder input
        x = self.ln1(
            dec_input + self.mha1(dec_input, dec_input, dec_input, dec_mask)
        )

        # Multi-head attention including encoder output
        x = self.ln2(x + self.mha2(x, enc_output, enc_output, enc_mask))

        x = self.ln3(x + self.ann(x))
        x = self.dropout(x)
        return x

# PositionalEncoding, Encoder, Decoder, and Transformer classes remain unchanged.
