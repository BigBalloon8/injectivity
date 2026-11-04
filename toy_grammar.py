"""
Toy probabilistic grammar ("mini English") for training very small language models.

- 40-word vocabulary, singular/plural agreement, relative clauses, prepositional phrases
- The grammar is unambiguous, so the entropy of the sentence distribution equals the
  entropy of the derivations and can be computed exactly. That gives you a hard floor
  for your model's loss.
- Also writes agreement "minimal pairs" for testing whether the model learned
  subject-verb agreement across a relative clause (e.g. "the dogs that chase the cat run/runs").

Usage:
    python toy_grammar.py --n 50000 --seed 0
Outputs:
    train.txt   one sentence per line, tokens separated by spaces, ending with "."
    valid.txt   held-out sentences from the same distribution
    vocab.txt   one token per line
    pairs.jsonl agreement test pairs: {"good": ..., "bad": ..., "has_rc": bool}
"""

import argparse
import json
import math
import random
from collections import Counter

# ---------------------------------------------------------------------------
# Grammar: nonterminal -> list of (probability, right-hand side)
# Anything that is not a key in GRAMMAR is a terminal (a word).
# ---------------------------------------------------------------------------

def uniform(words):
    p = 1.0 / len(words)
    return [(p, [w]) for w in words]

GRAMMAR = {
    "S": [(0.5, ["NP_sg", "VP_sg", "."]),
          (0.5, ["NP_pl", "VP_pl", "."])],

    # Subject noun phrases (number must agree with the main verb)
    "NP_sg": [(0.50, ["Det_sg", "N_sg"]),
              (0.30, ["Det_sg", "Adj", "N_sg"]),
              (0.20, ["Det_sg", "N_sg", "RC_sg"])],
    "NP_pl": [(0.50, ["Det_pl", "N_pl"]),
              (0.30, ["Det_pl", "Adj", "N_pl"]),
              (0.20, ["Det_pl", "N_pl", "RC_pl"])],

    # Object / prepositional-object noun phrases can be either number
    "NP": [(0.5, ["NP_sg"]), (0.5, ["NP_pl"])],

    # Relative clauses: the verb inside agrees with the noun they modify
    "RC_sg": [(1.0, ["that", "VP_sg"])],
    "RC_pl": [(1.0, ["that", "VP_pl"])],

    "VP_sg": [(0.45, ["Vt_sg", "NP"]),
              (0.35, ["Vi_sg"]),
              (0.20, ["Vi_sg", "PP"])],
    "VP_pl": [(0.45, ["Vt_pl", "NP"]),
              (0.35, ["Vi_pl"]),
              (0.20, ["Vi_pl", "PP"])],

    "PP": [(1.0, ["P", "NP"])],

    "Det_sg": [(0.5, ["the"]), (0.3, ["a"]), (0.2, ["every"])],
    "Det_pl": [(0.5, ["the"]), (0.3, ["some"]), (0.2, ["two"])],
    "Adj":    uniform(["small", "big", "happy", "old", "red"]),
    "N_sg":   uniform(["dog", "cat", "bird", "child", "farmer", "fox"]),
    "N_pl":   uniform(["dogs", "cats", "birds", "children", "farmers", "foxes"]),
    "Vt_sg":  uniform(["chases", "sees", "likes", "follows"]),
    "Vt_pl":  uniform(["chase", "see", "like", "follow"]),
    "Vi_sg":  uniform(["sleeps", "runs", "sings"]),
    "Vi_pl":  uniform(["sleep", "run", "sing"]),
    "P":      uniform(["near", "under"]),
}

# Singular <-> plural verb forms, used to build agreement test pairs
VERB_FLIP = {}
for sg, pl in [("chases", "chase"), ("sees", "see"), ("likes", "like"),
               ("follows", "follow"), ("sleeps", "sleep"), ("runs", "run"),
               ("sings", "sing")]:
    VERB_FLIP[sg] = pl
    VERB_FLIP[pl] = sg


def check_grammar():
    for lhs, rules in GRAMMAR.items():
        total = sum(p for p, _ in rules)
        assert abs(total - 1.0) < 1e-9, f"probabilities for {lhs} sum to {total}"


def vocabulary():
    words = set()
    for rules in GRAMMAR.values():
        for _, rhs in rules:
            words.update(s for s in rhs if s not in GRAMMAR)
    return sorted(words)


# ---------------------------------------------------------------------------
# Exact statistics: entropy and expected length, by fixed-point iteration
# (iteration is needed because the grammar is recursive through relative clauses
#  and prepositional phrases)
# ---------------------------------------------------------------------------

def exact_stats(iters=500):
    H = {a: 0.0 for a in GRAMMAR}   # entropy in bits of the derivation from a
    L = {a: 0.0 for a in GRAMMAR}   # expected number of words produced from a
    for _ in range(iters):
        newH, newL = {}, {}
        for a, rules in GRAMMAR.items():
            h = l = 0.0
            for p, rhs in rules:
                h += p * (-math.log2(p) + sum(H.get(s, 0.0) for s in rhs))
                l += p * sum(L[s] if s in GRAMMAR else 1.0 for s in rhs)
            newH[a], newL[a] = h, l
        H, L = newH, newL
    return H["S"], L["S"]


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def expand(sym, rng, out):
    """Expand sym, appending words to out. Returns log2-probability of the derivation."""
    if sym not in GRAMMAR:
        out.append(sym)
        return 0.0
    rules = GRAMMAR[sym]
    r = rng.random()
    acc = 0.0
    for p, rhs in rules:
        acc += p
        if r < acc:
            break
    lp = math.log2(p)
    for s in rhs:
        lp += expand(s, rng, out)
    return lp


def sample_sentence(rng):
    """Returns (tokens, log2 prob, index of main verb, subject has relative clause)."""
    p, rhs = None, None
    r = rng.random()
    acc = 0.0
    for p, rhs in GRAMMAR["S"]:
        acc += p
        if r < acc:
            break
    tokens = []
    lp = math.log2(p)
    lp += expand(rhs[0], rng, tokens)          # subject NP
    verb_idx = len(tokens)
    has_rc = "that" in tokens
    lp += expand(rhs[1], rng, tokens)          # main VP
    tokens.append(".")
    return tokens, lp, verb_idx, has_rc


# ---------------------------------------------------------------------------
# Exact next-token distributions (used by data.py's "grammar" task)
#
# A parser state is a stack of symbols still to be produced. Expanding the top
# until a word is on top, then matching it against the next word, tracks every
# derivation consistent with the prefix. This works because the grammar has no
# left recursion, and it gives p(next word | prefix) exactly.
# ---------------------------------------------------------------------------

_EXPAND_CACHE = {}

def _terminal_tops(stack):
    """All ways to expand stack's top until a word is on top: list of (stack, prob)."""
    if stack in _EXPAND_CACHE:
        return _EXPAND_CACHE[stack]
    out = []
    if stack:
        top = stack[0]
        if top not in GRAMMAR:
            out.append((stack, 1.0))
        else:
            for p, rhs in GRAMMAR[top]:
                for s, q in _terminal_tops(tuple(rhs) + stack[1:]):
                    out.append((s, p * q))
    _EXPAND_CACHE[stack] = out
    return out


def start_states():
    return {("S",): 1.0}


def next_dist(states):
    """Exact distribution over the next word given the parser states."""
    mass = {}
    for stack, p in states.items():
        for s, q in _terminal_tops(stack):
            mass[s[0]] = mass.get(s[0], 0.0) + p * q
    total = sum(mass.values())
    return {w: m / total for w, m in mass.items()}


def advance(states, word):
    """Parser states after reading word (renormalized to avoid underflow)."""
    new = {}
    for stack, p in states.items():
        for s, q in _terminal_tops(stack):
            if s[0] == word:
                new[s[1:]] = new.get(s[1:], 0.0) + p * q
    total = sum(new.values())
    if total == 0:
        raise ValueError(f"{word!r} is impossible here")
    return {s: p / total for s, p in new.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50000, help="training sentences")
    ap.add_argument("--n_valid", type=int, default=5000)
    ap.add_argument("--n_pairs", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    check_grammar()
    rng = random.Random(args.seed)
    vocab = vocabulary()

    H, L = exact_stats()
    print(f"vocabulary size:              {len(vocab)} words")
    print(f"sentence entropy (exact):     {H:.3f} bits")
    print(f"expected tokens per sentence: {L:.3f} (including the final '.')")
    print(f"loss floor per token:         {H / L:.4f} bits = {H / L * math.log(2):.4f} nats")
    print("  (a perfect model predicting every token incl. '.', given the start of the sentence,")
    print("   gets this average cross-entropy; PyTorch's cross_entropy reports nats)")

    def write_corpus(path, n):
        lengths = Counter()
        lps = []
        with open(path, "w") as f:
            for _ in range(n):
                toks, lp, _, _ = sample_sentence(rng)
                f.write(" ".join(toks) + "\n")
                lengths[len(toks)] += 1
                lps.append(-lp)
        return lengths, sum(lps) / len(lps)

    lengths, emp_H = write_corpus("train.txt", args.n)
    write_corpus("valid.txt", args.n_valid)
    print(f"sampled sentence entropy:     {emp_H:.3f} bits (should be close to exact)")
    print(f"max sentence length seen:     {max(lengths)} tokens")

    with open("vocab.txt", "w") as f:
        f.write("\n".join(vocab) + "\n")

    # Agreement test pairs; half of them have a relative clause between subject and verb
    n_rc = n_plain = 0
    with open("pairs.jsonl", "w") as f:
        while n_rc + n_plain < args.n_pairs:
            toks, _, vi, has_rc = sample_sentence(rng)
            if has_rc and n_rc >= args.n_pairs // 2:
                continue
            if not has_rc and n_plain >= args.n_pairs - args.n_pairs // 2:
                continue
            bad = list(toks)
            bad[vi] = VERB_FLIP[bad[vi]]
            f.write(json.dumps({"good": " ".join(toks), "bad": " ".join(bad),
                                "has_rc": has_rc}) + "\n")
            n_rc += has_rc
            n_plain += not has_rc

    print("\nexamples:")
    for _ in range(8):
        print("  " + " ".join(sample_sentence(rng)[0]))
    print("\nwrote train.txt, valid.txt, vocab.txt, pairs.jsonl")


if __name__ == "__main__":
    main()