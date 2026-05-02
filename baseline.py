import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMBaseline(nn.Module):
    """参数量与 CSM v2 相近的 baseline"""
    def __init__(self, obs_dim=56, n_actions=6, hidden=96):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.lstm = nn.LSTMCell(hidden, hidden)
        self.policy = nn.Linear(hidden, n_actions)
        self.value = nn.Linear(hidden, 1)
        self.hidden_size = hidden
        self.reset_hidden()
    
    def reset_hidden(self):
        self.h = torch.zeros(1, self.hidden_size)
        self.c = torch.zeros(1, self.hidden_size)
    
    def forward(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32) if not isinstance(obs, torch.Tensor) else obs
        x = self.encoder(obs_t).unsqueeze(0)
        self.h, self.c = self.lstm(x, (self.h, self.c))
        feat = self.h.squeeze(0)
        return {
            'action_probs': F.softmax(self.policy(feat), dim=-1),
            'value': self.value(feat),
        }