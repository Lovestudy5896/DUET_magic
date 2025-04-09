import torch
import torch.nn as nn
class Shared_extractor(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.d_model
        self.channels = configs.enc_in
        self.enc_in = 1 if configs.CI else configs.enc_in

        self.ffn = nn.Sequential(
            nn.Linear(self.enc_in, configs.d_ff),
            nn.ReLU() if configs.activation == "relu" else nn.GELU(),
            nn.Linear(configs.d_ff, self.enc_in)
        )

    def encoder(self, x):
        residual = x  # [B, L, C]
        x = self.ffn(x)
        return x + residual

    def forecast(self, x_enc):
        return self.encoder(x_enc)

    def forward(self, x_enc):
        if x_enc.shape[0] == 0:
            return torch.empty((0, self.pred_len, self.enc_in)).to(x_enc.device)
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]
