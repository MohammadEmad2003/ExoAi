"""
TabKANet: Tabular Data Modelling with Kolmogorov-Arnold Network + Transformer
Extracted from cosmicanalystsexoai.ipynb
"""

import torch
import torch.nn as nn
import math


class KANLayer(nn.Module):
    """
    Vectorized KAN-style module:
    - Inputs: x (B, n_inputs)
    - Internal "inner" linear combinations: u_q = x @ W[:, q] + b[q]  (q=0..K-1)
    - Apply per-q outer univariate mapping phi_q (shared small MLP applied vectorized)
    - Learnable mixing matrix A (K x n_out_features) to produce per-output-feature embeddings.
    - Output: (B, n_out, d_model) where n_out typically = n_inputs (one embedding per numeric feature)
    """
    def __init__(self, n_inputs, n_out, K_inner=16, d_model=64, phi_hidden=64):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_out = n_out
        self.K = K_inner
        self.d_model = d_model
        # inner linear weights W: shape (n_inputs, K)
        self.W = nn.Parameter(torch.randn(n_inputs, K_inner) * (1.0 / math.sqrt(n_inputs)))
        self.b = nn.Parameter(torch.zeros(K_inner))
        # phi: small MLP applied to scalar u_q -> vector d_model, vectorized below
        self.phi = nn.Sequential(
            nn.Linear(1, phi_hidden),
            nn.GELU(),
            nn.Linear(phi_hidden, d_model)
        )
        # mixing matrix A: (K, n_out) -> how each inner channel contributes to each output feature
        self.A = nn.Parameter(torch.randn(K_inner, n_out) * (1.0 / math.sqrt(K_inner)))
        # optional layernorm on output vector dims
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, n_inputs)
        B = x.shape[0]
        device = x.device
        # compute u = x @ W + b -> (B, K)
        u = x @ self.W.to(device) + self.b.to(device)  # (B, K)
        # apply phi to each scalar of u vectorized:
        u_flat = u.reshape(-1, 1)                      # (B*K, 1)
        phi_out = self.phi(u_flat)                     # (B*K, d_model)
        phi_out = phi_out.view(B, self.K, self.d_model) # (B, K, d_model)
        # mix into per-output-feature embeddings using A: (K, n_out)
        # out = einsum 'bkd,kp->bpd' -> (B, n_out, d_model)
        out = torch.einsum('bkd,kp->bpd', phi_out, self.A.to(device))
        out = self.ln(out)
        return out  # (B, n_out, d_model)


class TabKANet(nn.Module):
    """
    TabKANet: Tabular Data Modelling with Kolmogorov-Arnold Network + Transformer
    """
    def __init__(self, n_num, n_cat, cat_card_list, d_model=64, K_inner=16, 
                 trans_heads=4, trans_depth=3, mlp_hidden=128, n_classes=3, dropout=0.1):
        super().__init__()
        self.n_num = n_num
        self.n_cat = n_cat
        self.d_model = d_model
        self.n_classes = n_classes
        # KAN: map numeric features -> (B, n_num, d_model)
        # Here set n_out = n_num so we output per-numeric-feature embedding
        self.kan = KANLayer(n_inputs=n_num if n_num>0 else 1, n_out=n_num if n_num>0 else 1, 
                           K_inner=K_inner, d_model=d_model, phi_hidden=max(32,d_model))
        # categorical embeddings
        self.cat_embs = nn.ModuleList([nn.Embedding(card, d_model, padding_idx=card-1) 
                                      for card in cat_card_list]) if n_cat>0 else nn.ModuleList()
        # learned feature positional embedding for all features (num + cat)
        self.n_features = (n_num if n_num>0 else 0) + n_cat
        self.feat_emb = nn.Embedding(self.n_features + 1, d_model)
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1,1,d_model) * 0.02)
        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, trans_heads, 
                                                 dim_feedforward=mlp_hidden, 
                                                 dropout=dropout, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=trans_depth)
        self.head_ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x_num, x_cat):
        device = x_num.device if x_num is not None and x_num.numel() else next(self.parameters()).device
        # numeric KAN tokens (B, n_num, d_model)
        if self.n_num > 0:
            kan_out = self.kan(x_num)  # (B, n_num, d_model)
        else:
            kan_out = torch.zeros(x_cat.shape[0], 0, self.d_model, device=device)

        # categorical embeddings -> (B, n_cat, d_model)
        if self.n_cat > 0:
            cat_tokens = []
            for i, emb in enumerate(self.cat_embs):
                idxs = x_cat[:, i]
                pad = emb.padding_idx
                pad_t = torch.tensor(pad, dtype=idxs.dtype, device=device)
                idxs = torch.where(idxs < 0, pad_t, idxs).to(device)
                cat_tokens.append(emb(idxs))
            cat_tokens = torch.stack(cat_tokens, dim=1)  # (B, n_cat, d_model)
        else:
            cat_tokens = torch.zeros(x_num.shape[0], 0, self.d_model, device=device)

        # concatenate tokens per-feature in fixed order: numerics first, then cat
        tokens = torch.cat([kan_out, cat_tokens], dim=1)  # (B, n_features, d_model)
        B = tokens.shape[0]
        # add learned feature embeddings
        feat_idx = torch.arange(tokens.size(1), device=device).unsqueeze(0)  # (1, n_features)
        tokens = tokens + self.feat_emb(feat_idx)

        # prepend CLS
        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, tokens], dim=1)   # (B, n_features+1, d_model)
        seq = seq.permute(1,0,2)                # (seq_len, B, d_model)
        out = self.transformer(seq)             # (seq_len, B, d_model)
        cls_out = out[0]                        # (B, d_model)
        logits = self.head(self.head_ln(cls_out))
        return logits


def get_model(**config):
    """
    Factory function to create TabKANet model
    """
    return TabKANet(**config)
