"""
toy_data.py

Synthetic datasets for stress-testing the MLP of a toy transformer.

All three tasks are MLP-heavy: attention only has to route operands into
position, while the MLP does the actual nonlinear computation / memorization.
(That's the opposite of copying / induction tasks, which are attention-bound
and make poor MLP benchmarks.)

  "modular"  : a OP b (mod p)     the grokking task; MLP learns the arithmetic
  "kv"       : random key -> value   raw MLP key-value memory capacity
  "boolean"  : f(bits) -> {0, 1}     per-token nonlinear computation (e.g. parity)

Plus one language task (needs toy_grammar.py in the same folder):

  "grammar"  : sentence prefix -> next word, from a toy English grammar.
               Unlike the three above this is NOT MLP-isolated: attention has to
               carry information (e.g. the subject's number) across the sentence.

Every task is framed IDENTICALLY for the model so your training loop never
changes between them:

    input  : a sequence of token ids,            shape (seq_len,)
    target : a single token id, read off at the LAST position (meta.target_pos)
    loss   : cross-entropy at that one position over `meta.vocab_size` classes

    logits = model(input_ids)                  # (batch, seq_len, vocab_size)
    pred   = logits[:, meta.target_pos, :]     # (batch, vocab_size)
    loss   = F.cross_entropy(pred, target)

`meta` also hands you exactly the numbers you need to configure the model:
vocab_size (embedding + unembedding) and seq_len (context length).

Pass batch_size=None for full-batch gradient descent (the usual choice for
grokking); pass an int for minibatches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class DatasetMeta:
    task: str
    vocab_size: int   # size of the token embedding / unembedding table
    seq_len: int      # number of input tokens per example
    target_pos: int   # position to read the prediction from (== seq_len - 1)
    n_train: int
    n_test: int
    extra: dict       # task-specific info (p, op, n_keys, fn, ...)


def set_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)


def _make_loaders(x, y, train_frac, batch_size, seed):
    """Shuffle, split into train/test, and wrap in DataLoaders.

    batch_size=None -> one batch == the whole training split (full-batch GD).
    The test split is always evaluated in a single batch.
    Returns (train_loader, test_loader_or_None, n_train, n_test).
    """
    x = x.long()
    y = y.long()
    n = x.shape[0]

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_train = int(round(train_frac * n))
    tr_idx, te_idx = perm[:n_train], perm[n_train:]

    train_ds = TensorDataset(x[tr_idx], y[tr_idx])
    bs = n_train if batch_size is None else batch_size
    train_loader = DataLoader(train_ds, batch_size=max(bs, 1), shuffle=True, num_workers=4)

    test_loader = None
    if len(te_idx) > 0:
        test_ds = TensorDataset(x[te_idx], y[te_idx])
        test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False, num_workers=4)

    return train_loader, test_loader, n_train, n - n_train


# --------------------------------------------------------------------------- #
# Task 1: modular arithmetic  (the grokking benchmark)
# --------------------------------------------------------------------------- #
_OPS = {
    "add":     lambda a, b, p: (a + b) % p,
    "sub":     lambda a, b, p: (a - b) % p,
    "mul":     lambda a, b, p: (a * b) % p,
    "squares": lambda a, b, p: (a * a + b * b) % p,   # a^2 + b^2 mod p
}


def make_modular(p=113, op="add", train_frac=0.3, batch_size=None, seed=0):
    """All p*p pairs (a, b). Sequence is [a, b, '='], predict the result at '='.

    Vocab is {0..p-1} for the numbers plus one extra id (= p) for the '=' token,
    so vocab_size = p + 1. The result is always < p, so it lives inside the vocab.
    train_frac < 1 is the whole point here: you train on a fraction and watch
    test accuracy grok the rule. p=113 gives 12769 examples.
    """
    if op not in _OPS:
        raise ValueError(f"unknown op {op!r}; choose one of {list(_OPS)}")
    set_seed(seed)

    a = torch.arange(p).repeat_interleave(p)   # 0,0,...,1,1,... pair index // p
    b = torch.arange(p).repeat(p)              # 0,1,...,p-1,0,1,... pair index % p
    result = _OPS[op](a, b, p)

    eq_id = p
    eq_col = torch.full_like(a, eq_id)
    x = torch.stack([a, b, eq_col], dim=1)     # (p*p, 3): [a, b, '=']
    y = result                                  # read at last position

    vocab = p + 1
    tr, te, ntr, nte = _make_loaders(x, y, train_frac, batch_size, seed)
    meta = DatasetMeta("modular", vocab, 3, 2, ntr, nte, {"p": p, "op": op})
    return tr, te, meta


# --------------------------------------------------------------------------- #
# Task 2: key -> value memorization  (MLP capacity)
# --------------------------------------------------------------------------- #
def make_kv(n_keys=512, n_values=None, train_frac=1.0, batch_size=None, seed=0, unique=False):
    """A fixed table of random key -> value pairs. Sequence is just [key].

    Values are random, so there is NOTHING to generalize to -- this measures
    raw storage. Keep train_frac=1.0 and read TRAIN accuracy as the fraction
    of pairs memorized. Sweep n_keys against your d_mlp to trace a capacity
    curve (MLPs act as key-value memories, so capacity scales with d_mlp).

    seq_len is 1, so the single position attends only to itself: this is a
    pure embed -> MLP -> unembed test with attention factored out entirely.
    """
    set_seed(seed)
    if n_values is None:
        n_values = n_keys

    g = torch.Generator().manual_seed(seed)
    keys = torch.arange(n_keys)
    if unique:
        values = torch.randperm(n_values, generator=g)
    else:   
        values = torch.randint(0, n_values, (n_keys,), generator=g)

    x = keys.unsqueeze(1)                       # (n_keys, 1): [key]
    y = values
    vocab = max(n_keys, n_values)

    tr, te, ntr, nte = _make_loaders(x, y, train_frac, batch_size, seed)
    meta = DatasetMeta("kv", vocab, 1, 0, ntr, nte,
                       {"n_keys": n_keys, "n_values": n_values})
    return tr, te, meta

def make_kv_seq(n_keys=512, n_values=None, seq_len=3, train_frac=1.0, batch_size=None, seed=0):
    """A fixed table of random key -> value pairs. Sequence is just [key].

    Values are random, so there is NOTHING to generalize to -- this measures
    raw storage. Keep train_frac=1.0 and read TRAIN accuracy as the fraction
    of pairs memorized. Sweep n_keys against your d_mlp to trace a capacity
    curve (MLPs act as key-value memories, so capacity scales with d_mlp).

    seq_len is 3, so the single position attends only to itself: this is a
    pure embed -> MLP -> unembed test with attention factored out entirely.
    """
    set_seed(seed)
    if n_values is None:
        n_values = n_keys

    g = torch.Generator().manual_seed(seed)
    keys = torch.randint(0, n_values, (n_keys, 3), generator=g)
    values = torch.randint(0, n_values, (n_keys,), generator=g)

    x = keys
    y = values
    vocab = max(n_keys, n_values)

    tr, te, ntr, nte = _make_loaders(x, y, train_frac, batch_size, seed)
    meta = DatasetMeta("kv", vocab, seq_len, seq_len-1, ntr, nte,
                       {"n_keys": n_keys, "n_values": n_values})
    return tr, te, meta


# --------------------------------------------------------------------------- #
# Task 3: boolean functions  (per-token nonlinear computation)
# --------------------------------------------------------------------------- #
def _bits_of(n_bits):
    """All 2**n_bits inputs as a (2**n_bits, n_bits) tensor of {0,1} tokens."""
    idx = torch.arange(1 << n_bits)
    return (idx.unsqueeze(1) >> torch.arange(n_bits)) & 1


def _truth_table(bits, fn, n_bits, seed):
    s = bits.sum(dim=1)
    if fn == "parity":
        return s % 2                                   # XOR of all bits
    if fn == "majority":
        return (s > n_bits // 2).long()
    if fn == "random":
        g = torch.Generator().manual_seed(seed)
        return torch.randint(0, 2, (bits.shape[0],), generator=g)
    raise ValueError(f"unknown fn {fn!r}; choose parity | majority | random")


def make_boolean(n_bits=8, fn="parity", train_frac=0.7, batch_size=None, seed=0):
    """Sequence is the n_bits input bits; predict f(bits) in {0,1} at the last bit.

    Tokens are just 0/1, so vocab_size = 2. 'parity' is the classic hard case
    that genuinely needs the MLP nonlinearity; 'random' tests memorizing an
    arbitrary 2**n_bits truth table; 'majority' is a structured threshold.
    With train_frac < 1 you can also test generalization to held-out inputs.
    """
    set_seed(seed)
    bits = _bits_of(n_bits)                     # (2**n_bits, n_bits) in {0,1}
    table = _truth_table(bits, fn, n_bits, seed)

    x = bits                                     # tokens ARE the bits
    y = table                                    # read at last bit position
    vocab = 2

    tr, te, ntr, nte = _make_loaders(x, y, train_frac, batch_size, seed)
    meta = DatasetMeta("boolean", vocab, n_bits, n_bits - 1, ntr, nte,
                       {"fn": fn, "n_bits": n_bits})
    return tr, te, meta


# --------------------------------------------------------------------------- #
# Task 4: toy English grammar  (next-word prediction)
# --------------------------------------------------------------------------- #
def make_grammar(n_sentences=20000, seq_len=40, train_frac=0.9, batch_size=256,
                 seed=0, n_agreement=2000):
    """Next-word prediction on sentences from toy_grammar.py.

    Every word of every sentence becomes one example: the input is the sentence
    so far (starting with a <bos> token), LEFT-padded with <pad> to seq_len, and
    the target is the next word. So the target is always read at the last
    position, exactly like the other tasks:

        "<pad> ... <pad> <bos> the dogs that"  ->  "chase"

    The split is by SENTENCE, so no test sentence is seen during training.
    Sentences longer than seq_len - 1 words are dropped (with the default 40
    that's extremely rare) so no example ever has its context cut off.

    batch_size defaults to 256 rather than full-batch: there are ~7 examples
    per sentence, so 20000 sentences give ~150k examples.

    meta.extra holds:
      floor_test_nats  the best possible test loss (mean cross-entropy of the TRUE
                       next-word distribution on your test set), computed exactly.
                       Compare your test loss to this, not to 0.
      floor_train_nats same for the train split
      acc_ceiling_test expected next-word accuracy of a perfect model (~50%).
                       Many next words are genuinely random (which noun, which
                       adjective), so 100% is impossible. acc_ceiling_train too.
      itos / stoi      id <-> word, pad_id, bos_id
      agreement        subject-verb agreement test, as tensors:
                         x       (n, seq_len)  prefix up to the main verb
                         good    (n,)          id of the correct verb form
                         bad     (n,)          id of the wrong verb form
                         has_rc  (n,) bool     a relative clause sits between them
                         mismatch(n,) bool     the noun right before the verb has
                                               the other number ("the dogs that
                                               chase the cat ___") -- hardest case
                       a pair is passed when logits[good] > logits[bad]
    """
    import math
    import random
    import toy_grammar as tg

    set_seed(seed)
    rng = random.Random(seed)

    words = tg.vocabulary()
    itos = ["<pad>", "<bos>"] + words
    stoi = {w: i for i, w in enumerate(itos)}
    pad_id, bos_id = 0, 1

    def encode_prefix(prefix):
        ids = [bos_id] + [stoi[w] for w in prefix]
        return [pad_id] * (seq_len - len(ids)) + ids

    # ---- sample sentences and split by sentence ----
    sentences, dropped = [], 0
    while len(sentences) < n_sentences:
        toks = tg.sample_sentence(rng)[0]
        if len(toks) > seq_len - 1:
            dropped += 1
            continue
        sentences.append(toks)
    n_tr_sent = int(round(train_frac * n_sentences))

    # ---- turn sentences into (prefix -> next word) examples ----
    def build(sents):
        xs, ys, nll, best = [], [], 0.0, 0.0
        for toks in sents:
            states = tg.start_states()
            for i, w in enumerate(toks):
                dist = tg.next_dist(states)
                nll -= math.log(dist[w])
                best += max(dist.values())   # chance the best guess is right
                xs.append(encode_prefix(toks[:i]))
                ys.append(stoi[w])
                states = tg.advance(states, w)
        x = torch.tensor(xs, dtype=torch.long)
        y = torch.tensor(ys, dtype=torch.long)
        n = max(len(ys), 1)
        return x, y, nll / n, best / n

    x_tr, y_tr, floor_tr, ceil_tr = build(sentences[:n_tr_sent])
    x_te, y_te, floor_te, ceil_te = build(sentences[n_tr_sent:])

    g = torch.Generator().manual_seed(seed)
    bs = len(y_tr) if batch_size is None else batch_size
    train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=max(bs, 1),
                              shuffle=True, num_workers=4, generator=g)
    test_loader = None
    if len(y_te) > 0:
        test_loader = DataLoader(TensorDataset(x_te, y_te), batch_size=len(y_te),
                                 shuffle=False, num_workers=4)

    # ---- agreement test: half with a relative clause, half without ----
    sg_nouns = {"dog", "cat", "bird", "child", "farmer", "fox"}
    pl_nouns = {"dogs", "cats", "birds", "children", "farmers", "foxes"}
    ax, ag, ab, arc, amm = [], [], [], [], []
    want_rc = n_agreement // 2
    while len(ag) < n_agreement:
        toks, _, vi, has_rc = tg.sample_sentence(rng)
        if vi > seq_len - 1:
            continue
        n_rc = sum(arc)
        if has_rc and n_rc >= want_rc:
            continue
        if not has_rc and len(ag) - n_rc >= n_agreement - want_rc:
            continue
        nouns = [w for w in toks[:vi] if w in sg_nouns or w in pl_nouns]
        mismatch = (nouns[0] in sg_nouns) != (nouns[-1] in sg_nouns)
        ax.append(encode_prefix(toks[:vi]))
        ag.append(stoi[toks[vi]])
        ab.append(stoi[tg.VERB_FLIP[toks[vi]]])
        arc.append(has_rc)
        amm.append(mismatch)
    agreement = {
        "x": torch.tensor(ax, dtype=torch.long),
        "good": torch.tensor(ag, dtype=torch.long),
        "bad": torch.tensor(ab, dtype=torch.long),
        "has_rc": torch.tensor(arc, dtype=torch.bool),
        "mismatch": torch.tensor(amm, dtype=torch.bool),
    }

    meta = DatasetMeta("grammar", len(itos), seq_len, seq_len - 1,
                       len(y_tr), len(y_te),
                       {"floor_train_nats": floor_tr, "floor_test_nats": floor_te,
                        "acc_ceiling_train": ceil_tr, "acc_ceiling_test": ceil_te,
                        "itos": itos, "stoi": stoi, "pad_id": pad_id, "bos_id": bos_id,
                        "n_sentences_train": n_tr_sent,
                        "n_sentences_test": n_sentences - n_tr_sent,
                        "dropped_long_sentences": dropped,
                        "agreement": agreement})
    return train_loader, test_loader, meta


@torch.no_grad()
def agreement_accuracy(model, meta, device="cpu"):
    """Fraction of agreement pairs where the model prefers the correct verb.
    Returns overall accuracy and the breakdown by condition."""
    a = meta.extra["agreement"]
    logits = model(a["x"].to(device))[:, meta.target_pos, :].cpu()
    idx = torch.arange(len(a["good"]))
    ok = logits[idx, a["good"]] > logits[idx, a["bad"]]
    def acc(mask):
        return ok[mask].float().mean().item() if mask.any() else float("nan")
    return {"all": ok.float().mean().item(),
            "no_rc": acc(~a["has_rc"]),
            "rc": acc(a["has_rc"]),
            "rc_mismatch": acc(a["has_rc"] & a["mismatch"])}


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #
_TASKS = {"modular": make_modular, "kv": make_kv, "boolean": make_boolean, "kv_seq": make_kv_seq,
          "grammar": make_grammar}


def make_dataset(task="modular", **kwargs) -> Tuple[DataLoader, DataLoader, DatasetMeta]:
    """Build any task by name. Returns (train_loader, test_loader, meta)."""
    if task not in _TASKS:
        raise ValueError(f"unknown task {task!r}; choose one of {list(_TASKS)}")
    return _TASKS[task](**kwargs)


if __name__ == "__main__":
    demos = [
        ("modular", dict(p=7, op="add")),
        ("modular", dict(p=7, op="mul")),
        ("kv",      dict(n_keys=256)),
        ("boolean", dict(n_bits=8, fn="parity")),
        ("grammar", dict(n_sentences=2000)),
    ]
    for task, kw in demos:
        tr, te, meta = make_dataset(task, **kw)
        xb, yb = next(iter(tr))
        n_test = meta.n_test
        print(f"[{task:8s} {kw}]")
        print(f"    vocab_size={meta.vocab_size}  seq_len={meta.seq_len}  "
              f"target_pos={meta.target_pos}  n_train={meta.n_train}  n_test={n_test}")
        print(f"    first batch: x={tuple(xb.shape)} y={tuple(yb.shape)}  "
              f"example  {xb[0].tolist()} -> {yb[0].item()}")