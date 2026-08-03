"""
benchmark.py

Zero-shot multiple-choice evaluation of a train_gpt.py checkpoint on SciQ and
ARC-Easy, scored the same way as the EleutherAI lm-evaluation-harness:

  * every answer option is scored by the summed log-likelihood of its tokens
    given the prompt (`acc`),
  * and by that log-likelihood divided by the length of the option in
    characters (`acc_norm`), which removes most of the bias towards short
    answers.

The architecture below is an inference-only copy of the one in train_gpt.py
(the reproj / injective_init hooks and the distributed plumbing are dropped);
keep it in sync if train_gpt.py's model changes.

Usage:
    python benchmark.py checkpoints/g=0.02_<uuid>_trial0.pt
    python benchmark.py checkpoints/*.pt --tasks sciq --limit 200
    python benchmark.py ckpt.pt --num-fewshot 5 --output results.json
"""

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F


########################################
#             Architecture             #
########################################

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), weight=self.gains.type_as(x))

class Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(dim//4)]))

    def forward(self, x_BTHD: Tensor):
        pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim=128):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim)

    def forward(self, x: Tensor):
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                           v.transpose(1, 2), scale=0.12, is_causal=True).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.proj(y)

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.fc = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)

    def forward(self, x: Tensor):
        return self.proj(self.fc(x).relu())

class Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim)
        self.mlp = MLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([Block(model_dim) for _ in range(num_layers)])
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)

    def forward(self, inputs: Tensor):
        """Returns the softcapped logits, i.e. train_gpt.py's forward without the loss."""
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        logits = self.proj(self.norm2(x)).float()
        return 15 * logits * (logits.square() + 15**2).rsqrt()


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = GPT(**ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt


########################################
#                Tasks                 #
########################################

# Prompt formats follow lm-evaluation-harness (sciq.yaml / arc_easy.yaml), so
# the numbers are comparable to published zero-shot results.

def _sciq_doc(doc):
    support = doc["support"].strip()
    context = (support + "\n" if support else "") + f"Question: {doc['question'].strip()}\nAnswer:"
    choices = [doc["distractor1"], doc["distractor2"], doc["distractor3"], doc["correct_answer"]]
    return dict(context=context, choices=choices, gold=3)

def _arc_doc(doc):
    labels, texts = doc["choices"]["label"], doc["choices"]["text"]
    if doc["answerKey"] not in labels:
        return None  # a handful of items carry a key that is not among the options
    context = f"Question: {doc['question'].strip()}\nAnswer:"
    return dict(context=context, choices=texts, gold=labels.index(doc["answerKey"]))

TASKS = {
    "sciq":     dict(path="allenai/sciq",       name=None,         split="test", fewshot_split="train", fmt=_sciq_doc),
    "arc_easy": dict(path="allenai/ai2_arc",    name="ARC-Easy",   split="test", fewshot_split="train", fmt=_arc_doc),
}

def load_task(task, limit=None, num_fewshot=0, seed=1234):
    from datasets import load_dataset
    spec = TASKS[task]
    ds = load_dataset(spec["path"], spec["name"])
    docs = [d for d in (spec["fmt"](x) for x in ds[spec["split"]]) if d is not None]
    if limit is not None:
        docs = docs[:limit]

    prefix = ""
    if num_fewshot > 0:
        pool = [d for d in (spec["fmt"](x) for x in ds[spec["fewshot_split"]]) if d is not None]
        shots = random.Random(seed).sample(pool, num_fewshot)
        prefix = "\n\n".join(f"{d['context']} {d['choices'][d['gold']]}" for d in shots) + "\n\n"
    return docs, prefix


########################################
#               Scoring                #
########################################

def encode_pair(tok, context, continuation):
    """Split the tokenisation of context+continuation at the context boundary."""
    whole = tok.encode(context + continuation)
    ctx = tok.encode(context)
    n = len(ctx)
    while n > 0 and whole[:n] != ctx[:n]:  # the boundary token may have merged
        n -= 1
    return whole[:n], whole[n:]

@torch.no_grad()
def score_requests(model, requests, batch_size, max_len, vocab_limit, device, progress_every=200):
    """requests: list of (context_tokens, continuation_tokens) -> list of summed logprobs."""
    order = sorted(range(len(requests)), key=lambda i: -(len(requests[i][0]) + len(requests[i][1])))
    scores = [0.0] * len(requests)
    for start in range(0, len(order), batch_size):
        idxs = order[start:start + batch_size]
        seqs, spans = [], []
        for i in idxs:
            ctx, cont = requests[i]
            cont = cont[-max_len:]
            full = (ctx + cont)[-(max_len + 1):]
            inp = full[:-1]
            seqs.append(inp)
            spans.append((len(inp) - len(cont), len(inp), cont))  # predict cont from these positions
        width = max(len(s) for s in seqs)
        batch = torch.zeros(len(seqs), width, dtype=torch.int32, device=device)
        for row, s in enumerate(seqs):
            batch[row, :len(s)] = torch.tensor(s, dtype=torch.int32, device=device)
        logits = model(batch)  # right padding + causal attention => real positions are unaffected
        if vocab_limit is not None:
            logits = logits[..., :vocab_limit]
        for row, i in enumerate(idxs):
            lo, hi, cont = spans[row]
            logprobs = logits[row, lo:hi].log_softmax(dim=-1)
            tgt = torch.tensor(cont, dtype=torch.long, device=device)
            scores[i] = logprobs.gather(-1, tgt[:, None]).sum().item()
        del logits
        if progress_every and (start // batch_size) % progress_every == 0:
            print(f"  scored {min(start + batch_size, len(order))}/{len(order)} continuations", flush=True)
    return scores


def evaluate(model, tok, task, args, device):
    docs, prefix = load_task(task, limit=args.limit, num_fewshot=args.num_fewshot, seed=args.seed)
    requests, owner = [], []
    for d_i, doc in enumerate(docs):
        for choice in doc["choices"]:
            ctx, cont = encode_pair(tok, prefix + doc["context"], " " + choice.strip())
            requests.append((ctx, cont))
            owner.append(d_i)
    print(f"[{task}] {len(docs)} docs, {len(requests)} continuations", flush=True)
    scores = score_requests(model, requests, args.batch_size, args.max_len,
                            None if args.full_vocab else tok.vocab_size, device)

    n_correct = n_correct_norm = 0
    per_doc = [[] for _ in docs]
    for r_i, d_i in enumerate(owner):
        per_doc[d_i].append(scores[r_i])
    for doc, s in zip(docs, per_doc):
        norm = [x / max(len(c.strip()) + 1, 1) for x, c in zip(s, doc["choices"])]  # +1 for the leading space
        n_correct += int(max(range(len(s)), key=s.__getitem__) == doc["gold"])
        n_correct_norm += int(max(range(len(norm)), key=norm.__getitem__) == doc["gold"])

    n = len(docs)
    acc, acc_norm = n_correct / n, n_correct_norm / n
    stderr = lambda p: math.sqrt(p * (1 - p) / n)
    chance = sum(1 / len(d["choices"]) for d in docs) / n
    return dict(task=task, n=n, acc=acc, acc_stderr=stderr(acc),
                acc_norm=acc_norm, acc_norm_stderr=stderr(acc_norm), chance=chance)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="+", type=Path, help="checkpoint .pt file(s) written by train_gpt.py")
    p.add_argument("--tasks", default="sciq,arc_easy", help="comma separated subset of " + ",".join(TASKS))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-len", type=int, default=1024, help="context window used at training time")
    p.add_argument("--limit", type=int, default=None, help="only evaluate the first N docs (smoke tests)")
    p.add_argument("--num-fewshot", type=int, default=0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--tokenizer", default="gpt2", help="must match the tokenizer the .bin shards were made with")
    p.add_argument("--full-vocab", action="store_true",
                   help="score over all 50304 output slots instead of the tokenizer's 50257 real tokens")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=Path, default=None, help="write results as JSON")
    args = p.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in tasks if t not in TASKS]
    assert not unknown, f"unknown task(s) {unknown}, choose from {list(TASKS)}"

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    device = torch.device(args.device)

    all_results = {}
    for ckpt_path in args.checkpoints:
        model, ckpt = load_checkpoint(ckpt_path, device)
        meta = " ".join(f"{k}={ckpt[k]}" for k in ("gamma", "train_steps", "trial", "val_loss") if k in ckpt)
        print(f"\n=== {ckpt_path} ===\n{meta}")
        results = [evaluate(model, tok, t, args, device) for t in tasks]
        all_results[str(ckpt_path)] = dict(gamma=ckpt.get("gamma"), val_loss=ckpt.get("val_loss"),
                                           num_fewshot=args.num_fewshot, results=results)

        print(f"\n{'task':<12}{'n':>6}{'acc':>10}{'±':>9}{'acc_norm':>12}{'±':>9}{'chance':>9}")
        for r in results:
            print(f"{r['task']:<12}{r['n']:>6}{r['acc']:>10.4f}{r['acc_stderr']:>9.4f}"
                  f"{r['acc_norm']:>12.4f}{r['acc_norm_stderr']:>9.4f}{r['chance']:>9.4f}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.output:
        args.output.write_text(json.dumps(all_results, indent=2))
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
