"""Decider vs. generative router, head to head.

Trains the small generative baseline briefly, then compares on a held-out set:
accuracy, invalid-output rate, and latency. The point of the comparison is
the one the decider-model vendors are making this week: when the job is
"pick one of N options", generating text is the slow, fragile way to do it.

    python compare.py --ckpt decider.pt --gen-steps 1200
"""

import argparse
import random
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import INTENTS, LABEL_TEXT, encode, make_example, tokenize
from decider import TinyDecider
from gen_baseline import GenRouter


def load_decider(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = TinyDecider(**ckpt["config"])
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, ckpt


def train_gen(vocab, steps, seed=7):
    """Teach the generator to emit intent names, teacher-forced.

    Uses its own extended vocab (adds <eos> at the end, so the decider's
    indices are untouched) and trains on [BOS] label-words [EOS]."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    gen_vocab = dict(vocab)
    gen_vocab.setdefault("<eos>", len(gen_vocab))
    bos, eos = gen_vocab["<bos>"], gen_vocab["<eos>"]
    model = GenRouter(len(gen_vocab))
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    model.train()
    for _ in tqdm(range(steps), desc="training generative baseline"):
        texts, labels = zip(*[make_example(rng) for _ in range(64)])
        q_ids, q_masks = zip(*[encode(t, gen_vocab, 24) for t in texts])
        lab = [[bos] + [gen_vocab[t] for t in tokenize(LABEL_TEXT[l])] + [eos]
               for l in labels]
        L = max(len(x) for x in lab)          # longest [BOS, w1..wk, EOS]
        ids, masks, tgts = [], [], []
        for qi, qm, lb in zip(q_ids, q_masks, lab):
            w = lb[1:]                        # label words + EOS to predict
            pad = L - len(w)
            ids.append(qi + lb + [0] * pad)
            masks.append(qm + [1] * len(lb) + [0] * pad)
            # logits at 23..23+L predict tokens at 24..24+L
            tgts.append([-100] * 24 + w + [-100] * pad)
        ids = torch.tensor(ids)
        logits = model(ids, torch.tensor(masks))
        loss = F.cross_entropy(logits[:, 23:23 + L].reshape(-1, logits.size(-1)),
                               torch.tensor(tgts)[:, 24:24 + L].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    print(f"  generative baseline final loss: {loss.item():.3f}")
    return model, gen_vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="decider.pt")
    ap.add_argument("--gen-steps", type=int, default=1200)
    ap.add_argument("--n-test", type=int, default=400)
    args = ap.parse_args()

    decider, ckpt = load_decider(args.ckpt)
    vocab, intents, temp = ckpt["vocab"], ckpt["intents"], ckpt["temperature"]
    gen, gen_vocab = train_gen(vocab, args.gen_steps)

    rng = random.Random(2026)
    test = [make_example(rng) for _ in range(args.n_test)]

    # --- decider ---
    t0 = time.perf_counter()
    d_correct = 0
    for text, label in test:
        ids, mask = encode(text, vocab, decider.max_len)
        best, _, _ = decider.decide(torch.tensor([ids]), torch.tensor([mask]),
                                    intents, temp)
        d_correct += best == label
    d_ms = (time.perf_counter() - t0) / len(test) * 1000

    # --- generative baseline ---
    t0 = time.perf_counter()
    g_correct = g_invalid = g_steps = 0
    for text, label in test:
        ids, mask = encode(text, vocab, 24)
        pred, valid, steps = gen.route(torch.tensor([ids]),
                                       torch.tensor([mask]), gen_vocab, intents)
        g_steps += steps
        if not valid:
            g_invalid += 1
        elif pred == label:
            g_correct += 1
    g_ms = (time.perf_counter() - t0) / len(test) * 1000

    n = len(test)
    print("\n==== decider vs generative router (test set, n=%d) ====" % n)
    print(f"{'':28s} {'accuracy':>9s} {'invalid':>8s} {'ms/query':>9s} {'fwd passes':>11s}")
    print(f"{'decider (pointer head)':28s} {d_correct/n:>9.3f} {'n/a':>8s} {d_ms:>9.1f} {'1':>11s}")
    print(f"{'generative baseline':28s} {g_correct/n:>9.3f} {g_invalid/n:>8.3f} {g_ms:>9.1f} "
          f"{g_steps/n:>11.1f}")
    print("\nThe generator has to emit the label one token at a time and sometimes")
    print("emits something that is not a valid option at all. The decider cannot")
    print("do that — its output space *is* the option set.")


if __name__ == "__main__":
    main()
