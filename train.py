"""Train the decider.

Data is synthetic and infinite, so we just stream fresh batches — no epochs.
After training we fit a temperature on a held-out set so the reported
confidences are at least roughly calibrated (a decider that says 0.99 and is
wrong half the time is worse than useless in a routing/escalation setup).
"""

import argparse
import random

import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import INTENTS, build_vocab, encode, make_example
from decider import TinyDecider


def batch(rng, vocab, bs, max_len, distractor_p=0.12, hedge_p=0.15):
    texts, labels = zip(*[make_example(rng, distractor_p, hedge_p) for _ in range(bs)])
    ids, masks = zip(*[encode(t, vocab, max_len) for t in texts])
    return (torch.tensor(ids), torch.tensor(masks),
            torch.tensor([INTENTS.index(l) for l in labels]))


# the calibration/eval stream is deliberately messier than training — closer
# to what the thing would see in the wild, and it keeps temperature fitting
# from collapsing (on near-separable data the best temperature is ~0, which
# is a lie)
EVAL_DISTRACTOR_P, EVAL_HEDGE_P = 0.40, 0.25


@torch.no_grad()
def evaluate(model, vocab, seed, n=1000, max_len=32, temperature=1.0):
    rng = random.Random(seed)
    correct, conf_sum, nll = 0, 0.0, 0.0
    for _ in range(n // 50):
        ids, mask, y = batch(rng, vocab, 50, max_len,
                             EVAL_DISTRACTOR_P, EVAL_HEDGE_P)
        logits = model(ids, mask, temperature)
        probs = logits.softmax(-1)
        pred = logits.argmax(-1)
        correct += (pred == y).sum().item()
        conf_sum += probs.max(-1).values.sum().item()
        nll += F.cross_entropy(logits, y, reduction="sum").item()
    return correct / n, conf_sum / n, nll / n


def fit_temperature(model, vocab, max_len=32):
    """Grid-search a temperature minimizing NLL on held-out data.

    Constrained to T >= 1.0 (soften only, never sharpen): on data the model
    already aces, the unconstrained optimum collapses toward 0, which just
    manufactures confidence. Hacky but works."""
    rng = random.Random(999)
    all_logits, all_y = [], []
    for _ in range(20):
        ids, mask, y = batch(rng, vocab, 50, max_len,
                             EVAL_DISTRACTOR_P, EVAL_HEDGE_P)
        with torch.no_grad():
            all_logits.append(model(ids, mask))
        all_y.append(y)
    logits = torch.cat(all_logits)
    y = torch.cat(all_y)
    best_t, best_nll = 1.0, float("inf")
    for t in [x / 20 for x in range(20, 101)]:
        nll = F.cross_entropy(logits / t, y).item()
        if nll < best_nll:
            best_t, best_nll = t, nll
    return best_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="decider.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    vocab = build_vocab()
    print(f"vocab: {len(vocab)} tokens, {len(INTENTS)} options: {INTENTS}")

    model = TinyDecider(len(vocab), len(INTENTS))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for step in tqdm(range(args.steps), desc="training"):
        ids, mask, y = batch(rng, vocab, args.batch, model.max_len)
        loss = F.cross_entropy(model(ids, mask), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 500 == 0:
            acc, conf, nll = evaluate(model, vocab, seed=1234)
            print(f"  step {step+1}: loss={loss.item():.3f} "
                  f"val_acc={acc:.3f} mean_conf={conf:.3f} val_nll={nll:.3f}")

    temp = fit_temperature(model, vocab)
    acc, conf, nll = evaluate(model, vocab, seed=1234, temperature=temp)
    print(f"temperature={temp:.2f} -> val_acc={acc:.3f} mean_conf={conf:.3f} nll={nll:.3f}")

    torch.save({"state": model.state_dict(), "vocab": vocab,
                "intents": INTENTS, "temperature": temp,
                "config": {"vocab_size": len(vocab), "n_options": len(INTENTS)}},
               args.ckpt)
    print(f"saved {args.ckpt}")


if __name__ == "__main__":
    main()
