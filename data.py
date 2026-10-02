"""Synthetic support-ticket routing data.

Infinite stream of labeled queries for training a decision model. The labels
are crisp (billing vs tech_support etc.) but the queries are messy on purpose:
distractor words from other intents, dropped nouns, hedged phrasing. That is
the whole point — a router earns its keep on the messy ones.
"""

import random
import re

INTENTS = ["billing", "tech_support", "sales", "returns", "account", "unclear"]

LABEL_TEXT = {
    "billing": "billing",
    "tech_support": "tech support",
    "sales": "sales",
    "returns": "returns",
    "account": "account",
    "unclear": "unclear",
}

PRODUCTS = ["router", "laptop", "phone plan", "streaming bundle",
            "smart thermostat", "headphones", "mesh wifi", "security camera"]
AMOUNTS = ["12.99", "49.99", "89.00", "149.99", "5.50", "230.00", "1,200.00"]
CITIES = ["Newark", "Austin", "Seattle", "Chicago", "Boston", "Denver"]
DAYS = ["Monday", "Tuesday", "last week", "yesterday", "this morning"]

# each template has {slots}; true label is the dict key even when the text
# borrows vocabulary from other intents (deliberate — distractors)
TEMPLATES = {
    "billing": [
        "my {card} was charged twice this month",
        "why is my invoice ${amt} higher than usual",
        "i want a refund on my last bill",
        "the technician fixed my {prod} but the charge on my statement looks wrong",
        "cancel my subscription and stop billing me",
        "i was billed for {prod} after i returned it {day}",
        "my autopay took ${amt} and i did not approve it",
        "can you explain the extra fees on my {card} statement",
        "the sales rep promised a discount but my bill shows full price",
        "i need a copy of my invoice from {day}",
        "my payment failed even though my {card} is fine",
        "you charged me a late fee but i paid on time",
        "why did my {prod} plan renew at a higher rate",
        "split this ${amt} charge across two payments please",
        "my bill mentions a device i never bought",
    ],
    "tech_support": [
        "my {prod} keeps disconnecting every few minutes",
        "the {prod} i bought {day} will not turn on",
        "wifi drops whenever i plug in the {prod}",
        "my bill is fine but the {prod} itself is broken",
        "how do i factory reset my {prod}",
        "the {prod} app crashes on startup since {day}",
        "my {prod} shows error code {code} and nothing works",
        "sound cuts out on my {prod} after the update",
        "the technician installed it wrong and now nothing connects",
        "my {prod} overheats and shuts down",
        "can you walk me through setting up the {prod} again",
        "the {prod} pairs with my phone but not my laptop",
        "firmware update bricked my {prod}, what now",
        "intermittent lag on the {prod} during video calls",
        "the {prod} stopped working right after the power outage",
    ],
    "sales": [
        "what is the price of the {prod} with the current promo",
        "do you have a bundle deal for {prod} and {prod2}",
        "i am comparing your {prod} against a competitor, convince me",
        "is there a student discount on the {prod}",
        "when does the {prod} go on sale next",
        "can i get a quote for ten {prod} units for my office",
        "does the {prod} come in other colors",
        "what is the warranty on the {prod}",
        "i want to upgrade from my old {prod}, what do you recommend",
        "do you offer financing for the ${amt} {prod}",
        "the salesperson at your {city} store was helpful, i want to buy now",
        "is the {prod} in stock near {city}",
        "what is the return policy if i buy the {prod} today",
        "any trade-in value for my old {prod} toward a new one",
        "looking for a gift, is the {prod} a good pick for a teenager",
    ],
    "returns": [
        "i want to send back the {prod} i bought {day}",
        "how do i start a return for my {prod}",
        "the {prod} arrived damaged, i need a replacement or refund",
        "can i return the {prod} to the {city} store instead of shipping",
        "my return label never arrived by email",
        "it has been two weeks and my refund for the {prod} is still pending",
        "i opened the box but never used the {prod}, can i still return it",
        "the {prod} is not what the website described, sending it back",
        "return pickup was scheduled for {day} and nobody came",
        "i was told the refund would post in 3 days, it has been 10",
        "can i exchange the {prod} for a different model",
        "the courier lost my return package, what are my options",
        "do i need the original receipt to return the {prod}",
        "i changed my mind about the {prod}, still in the return window",
        "partial refund received but i returned everything, please check",
    ],
    "account": [
        "i cannot log in to my account since {day}",
        "reset my password, the email link expired",
        "change the email address on my account",
        "my account was locked after too many attempts",
        "how do i delete my account and my data",
        "merge my two accounts into one please",
        "update my shipping address for future orders",
        "i never got the verification code by text",
        "someone logged in from {city} and it was not me",
        "turn on two-factor authentication for my account",
        "my profile still shows my old phone number",
        "cancel the account recovery request i made {day}",
        "why does the app say my session expired every hour",
        "add my spouse as an authorized user on the account",
        "export all my order history as a csv",
    ],
    "unclear": [
        "hi i need help with my stuff",
        "can someone call me about my order",
        "hello, are you there",
        "i have a question about the thing i got",
        "it is not working the way i expected",
        "please help, this is urgent",
        "i talked to someone {day} but nothing happened",
        "not happy with my recent experience",
        "can you look into this for me",
        "i need to talk to a human",
        "something is wrong, not sure what",
        "following up on my earlier message",
    ],
}

WORD_RE = re.compile(r"[a-z]+|\d[\d,]*\.?\d*|\$")


def tokenize(text):
    return WORD_RE.findall(text.lower())


def fill(template, rng):
    s = template
    if "{card}" in s:
        s = s.replace("{card}", rng.choice(["card", "credit card", "debit card"]))
    if "{amt}" in s:
        s = s.replace("{amt}", rng.choice(AMOUNTS))
    if "{prod}" in s:
        s = s.replace("{prod}", rng.choice(PRODUCTS))
    if "{prod2}" in s:
        s = s.replace("{prod2}", rng.choice(PRODUCTS))
    if "{city}" in s:
        s = s.replace("{city}", rng.choice(CITIES))
    if "{day}" in s:
        s = s.replace("{day}", rng.choice(DAYS))
    if "{code}" in s:
        s = s.replace("{code}", "E" + str(rng.randint(100, 599)))
    return s


HEDGES = ["um", "hey", "hi", "please", "so", "well", "ok", "thanks"]


def make_example(rng, distractor_p=0.12, hedge_p=0.15):
    """One (text, intent) pair. A fraction get a distractor phrase from another
    intent's vocab — the label stays put. Raise distractor_p for a harder stream."""
    intent = rng.choice(INTENTS)
    text = fill(rng.choice(TEMPLATES[intent]), rng)
    # distractor: borrow a phrase from a *different* intent's vocab, label stays
    if intent != "unclear" and rng.random() < distractor_p:
        other = rng.choice([i for i in INTENTS if i not in (intent, "unclear")])
        distractor = rng.choice(TEMPLATES[other])
        phrase = " ".join(fill(distractor, rng).split()[:4])
        text = text + ", " + phrase
    if rng.random() < hedge_p:  # hedged phrasing
        text = rng.choice(HEDGES) + ", " + text
    return text, intent


def build_vocab(n_samples=20000, seed=0, min_freq=2, max_vocab=4000):
    from collections import Counter
    rng = random.Random(seed)
    counts = Counter()
    for _ in range(n_samples):
        text, _ = make_example(rng)
        counts.update(tokenize(text))
    # label words must survive even if rare
    for intent in INTENTS:
        counts.update(tokenize(LABEL_TEXT[intent]))
    words = ["<pad>", "<unk>", "<bos>"] + [
        w for w, c in counts.most_common(max_vocab) if c >= min_freq
    ]
    # belt and suspenders: the generator must be able to emit every label
    for intent in INTENTS:
        for t in tokenize(LABEL_TEXT[intent]):
            if t not in words:
                words.append(t)
    return {w: i for i, w in enumerate(words)}


def encode(text, vocab, max_len=32):
    ids = [vocab.get(t, vocab["<unk>"]) for t in tokenize(text)][:max_len]
    mask = [1] * len(ids)
    while len(ids) < max_len:
        ids.append(vocab["<pad>"])
        mask.append(0)
    return ids, mask
