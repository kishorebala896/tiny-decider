"""The generative baseline: same problem, solved the "normal" way.

A small decoder-only transformer that must *generate* the intent name
("tech support", "billing", ...) token by token. This is what happens when
you route with a prompted LLM: autoregressive decoding, then you parse the
text back into a decision and pray it was one of your options.

Included to show why the decider pattern exists, not because it's good.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import LABEL_TEXT, encode, tokenize


class GenRouter(nn.Module):
    def __init__(self, vocab_size, d=64, n_heads=4, n_layers=2,
                 d_ff=128, max_len=40, dropout=0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerDecoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True)
        # decoder-only: the "memory" is just the encoded query prepended.
        # simpler to read as a causal transformer over [query | label]
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True)
        self.trunk = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(d, vocab_size, bias=False)  # the LM head the decider throws away
        self.drop = nn.Dropout(dropout)
        self.scale = math.sqrt(d)
        self.max_len = max_len

    def forward(self, ids, mask):
        # ids = [query tokens | label tokens], causal over the whole thing.
        # good enough for a toy baseline.
        pos = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
        h = self.drop(self.tok(ids) + self.pos(pos))
        causal = torch.triu(torch.ones(ids.size(1), ids.size(1),
                                       device=ids.device, dtype=torch.bool), 1)
        pad = mask == 0
        h = self.trunk(h, mask=causal,
                       src_key_padding_mask=pad)
        return self.head(h)

    @torch.no_grad()
    def route(self, query_ids, query_mask, vocab, intents, max_new=6):
        """Greedy-decode the intent name. Returns (pred_intent, valid, n_steps).

        Stops at <eos>. Anything else — rambling past the label, emitting a
        word that isn't an option — counts as invalid. That fragility is the
        point of the comparison."""
        inv = {i: w for w, i in vocab.items()}
        bos = vocab["<bos>"]
        eos = vocab["<eos>"]
        ids = query_ids.clone()
        mask = query_mask.clone()
        steps = 0
        for _ in range(max_new):
            causal = torch.triu(torch.ones(ids.size(1), ids.size(1),
                                           device=ids.device, dtype=torch.bool), 1)
            pos = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
            h = self.tok(ids) + self.pos(pos)
            h = self.trunk(h, mask=causal, src_key_padding_mask=(mask == 0))
            nxt = int(self.head(h[:, -1]).argmax(-1))
            steps += 1
            if nxt == eos:
                break
            ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], 1)
            mask = torch.cat([mask, torch.ones(1, 1, dtype=torch.long,
                                               device=mask.device)], 1)
        words = [inv.get(int(t), "?") for t in ids[0][query_ids.size(1):]]
        text = " ".join(w for w in words
                        if w not in ("<pad>", "<bos>", "<eos>"))
        valid = text in [LABEL_TEXT[i] for i in intents]
        pred = next((i for i in intents if LABEL_TEXT[i] == text), None)
        return pred, valid, steps
