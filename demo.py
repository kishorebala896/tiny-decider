"""Try the decider on some real-ish queries.

Shows the scores for every option, the latency of a single forward pass, and
the escalation call when confidence is below threshold. Run it:

    python demo.py --ckpt decider.pt
"""

import argparse
import time

import torch

from data import encode
from decider import TinyDecider

QUERIES = [
    "my credit card was charged twice for the router",
    "the thermostat keeps disconnecting from wifi every ten minutes",
    "do you have a student discount on the headphones",
    "i want to send back the laptop, the screen arrived cracked",
    "i cant log in, the password reset link expired",
    "hi can someone call me about my stuff",          # genuinely unclear
    "my bill is fine but the camera itself is broken",  # distractor: "bill"
    "the sales rep promised a discount but my statement shows full price",
]


def bar(p, width=24):
    return "#" * int(p * width)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="decider.pt")
    ap.add_argument("--threshold", type=float, default=0.6)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = TinyDecider(**ckpt["config"])
    model.load_state_dict(ckpt["state"])
    model.eval()
    vocab, intents, temp = ckpt["vocab"], ckpt["intents"], ckpt["temperature"]
    print(f"loaded {args.ckpt} (temperature={temp:.2f})\n")

    for q in QUERIES:
        ids, mask = encode(q, vocab, model.max_len)
        ids_t = torch.tensor([ids])
        mask_t = torch.tensor([mask])
        t0 = time.perf_counter()
        with torch.no_grad():
            best, probs, escalate = model.decide(
                ids_t, mask_t, intents, temp, args.threshold)
        ms = (time.perf_counter() - t0) * 1000
        print(f"> {q}")
        for intent in sorted(probs, key=probs.get, reverse=True)[:3]:
            print(f"    {intent:12s} {probs[intent]:.2f} {bar(probs[intent])}")
        flag = "  <-- ESCALATE to human" if escalate else ""
        print(f"    -> {best} ({ms:.1f} ms){flag}\n")


if __name__ == "__main__":
    main()
