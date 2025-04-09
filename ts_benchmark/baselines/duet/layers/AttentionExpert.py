import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionExpert(nn.Module):
    def __init__(self, d_model, num_heads):
        super(AttentionExpert, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm(x + self.dropout(ffn_output))
        return x
