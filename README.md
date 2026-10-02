# tiny-decider

A decision model built from scratch in PyTorch — no API keys, no pretrained
weights, runs on CPU. It routes support tickets to one of six teams by
*scoring the options directly* in a single forward pass, instead of
generating text and parsing it back into a decision.

## Why this, why now

This week the whole industry piled into the same idea within about 48 hours:

- **Cloudflare Clef / Clef-flash** (Oct 1) — open-weight decision models on a
  Qwen backbone, Apache 2.0, answering yes/no and multiple-choice questions
  with calibrated probabilities in one forward pass.
- **Amazon Strands Decider 2B** (Oct 1) — strips a Qwen base model down to
  its backbone, throws away the language-modeling head entirely, and replaces
  it with a ~1M-parameter "pointer head" that scores the options you hand it.
- **OpenAI's Luna** decisions API (mentioned Sep 30) — same category, closed.

The bet: when the job is "pick one of N options" — routing, triage,
guardrails, tool selection, evals — generating text is the slow, fragile way
to do it. You pay for autoregressive decoding, then you parse the output and
pray it landed on one of your options. A decision model skips all of that.

> The vendor numbers above are motivation, not the lesson. What I wanted was
> to feel the actual mechanism in my hands, at a scale where I can read every
> line.

## What this repo does

Trains a small transformer encoder (~320K params) on synthetic support
tickets, with a learned pointer head: one embedding row per option, scores
computed as `query @ options.T`. Then it compares that against a generative
baseline — a tiny decoder trained to *emit* the team name token by token, the
way you'd do it with a prompted LLM.

```
pip install -r requirements.txt        # CPU torch
python train.py --steps 2500          # a few minutes on CPU
python demo.py                         # route some queries, watch confidence
python compare.py                      # decider vs generative router, head to head
```

`demo.py` also shows the escalation call: when the top score is below
threshold, it flags the ticket for a human instead of guessing. That is the
whole point of calibrated probabilities in a routing setup.

## What surprised me

Two things. First, how little data the pointer head needs to get going —
with an infinite synthetic stream it separates the six teams fast, and most
of the remaining errors are on the deliberately ambiguous tickets, which is
exactly where you want the model to be unsure.

Second, the generative baseline is worse in the specific way the vendors
claim: not just slower (one forward pass per token instead of one total),
but it occasionally emits something that isn't a valid option at all. The
decider *cannot* do that — its output space is the option set. That
structural guarantee, more than the latency, is what sold me on the pattern
for guardrail-type use cases.

The temperature scaling taught me something too. My first attempt let the
grid search go below 1.0, and it immediately collapsed toward 0 — because
when the model already aces the calibration set, the NLL-optimal move is to
sharpen everything into fake certainty. That's manufactured confidence, so I
constrained the search to soften-only (T >= 1.0). It settled at 1.0: on this
toy data the model is nearly separable, so there's nothing to fix. The
escalation threshold in `demo.py` is there for the day the data isn't this
clean — which, with real tickets, is day one.

## Rough edges

- The data is synthetic, so the 95%+ accuracy is a toy number. Real tickets
  are messier in ways templates can't capture (sarcasm, multi-intent
  messages, twelve-turn threads).
- The generative baseline is under-trained relative to the decider on
  purpose (it's a demo, not a bake-off), but the *shape* of the result —
  slower, sometimes invalid — holds regardless.
- Temperature is fit by dumb grid search. It works; a proper implementation
  would use LBFGS on a real validation set.

## TODO

- [ ] Multi-intent tickets: score each option independently (sigmoid head)
      instead of forcing one winner.
- [ ] Try it on a real public dataset (e.g. Banking77) and see how far the
      toy holds up.
- [ ] Measure what happens to calibration under distribution shift —
      that's the part I'd actually worry about in production.

## The one-line version

If your LLM's job is to pick from a menu, stop asking it to write an essay
about the menu. Score the menu.
