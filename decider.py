"""The decision model.

A small transformer encoder turns the query into one vector; a learned
"pointer head" (one embedding row per option) scores the options directly:

    scores = query @ option_matrix.T / sqrt(d)

No language-modeling head, no autoregressive decoding, no parsing the model's
words back into a decision. One forward pass, one softmax, done. This is the
pattern behind this week's decider-model releases, shrunk to toy scale.
"""

import math

import torch
import torch.nn as nn


class TinyDecider(nn.Module):
    def __init__(self, vocab_size, n_options, d=128, n_heads=4,
                 n_layers=2, d_ff=256, max_len=32, dropout=0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        # this is the part that actually matters: learned option vectors,
        # the thing a normal LM head would be replaced with
        self.options = nn.Parameter(torch.randn(n_options, d) * 0.02)
        self.drop = nn.Dropout(dropout)
        self.scale = math.sqrt(d)
        self.max_len = max_len

    def encode(self, ids, mask):
        pos = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
        h = self.drop(self.tok(ids) + self.pos(pos))
        h = self.enc(h, src_key_padding_mask=(mask == 0))
        # masked mean pool -> one vector per query
        m = mask.unsqueeze(-1).float()
        return (h * m).sum(1) / m.sum(1).clamp(min=1)

    def forward(self, ids, mask, temperature=1.0):
        q = self.encode(ids, mask)
        return (q @ self.options.t()) / (self.scale * temperature)

    @torch.no_grad()
    def decide(self, ids, mask, intents, temperature=1.0, threshold=0.6):
        """Returns (best_intent, probs_dict, escalate_bool)."""
        probs = self(ids, mask, temperature).softmax(-1)[0]
        best = int(probs.argmax())
        return (intents[best],
                {i: float(p) for i, p in zip(intents, probs)},
                float(probs[best]) < threshold)
