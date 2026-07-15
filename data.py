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
    train_loader = DataLoader(train_ds, batch_size=max(bs, 1), shuffle=True)

    test_loader = None
    if len(te_idx) > 0:
        test_ds = TensorDataset(x[te_idx], y[te_idx])
        test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)

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
def make_kv(n_keys=512, n_values=None, train_frac=1.0, batch_size=None, seed=0):
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
# Dispatcher
# --------------------------------------------------------------------------- #
_TASKS = {"modular": make_modular, "kv": make_kv, "boolean": make_boolean, "kv_seq": make_kv_seq}


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