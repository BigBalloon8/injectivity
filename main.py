import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from scipy.linalg import null_space
import matplotlib.pyplot as plt
import math
import hashlib
from dataclasses import dataclass
import os

from data import make_dataset, agreement_accuracy
from toy_model import Transformer
from logger import Logger
from analysisv2 import find_n_pairs_collapsing_torch
from grapher import create_graphs
import time

torch.set_float32_matmul_precision("high")

@dataclass
class RunConfig:
    """Class for keeping track of an item in inventory."""
    task: str
    task_kwargs: dict
    seed: int

    num_epochs: int = 2**15
    log_freq: int = 64
    mbs: int = 2**15

    device: str = "cuda"
    dim: int = 4
    h_dim: int = 16
    n_heads: int = 1
    n_layers: int = 1
    gamma: float = 1.0
    low_collapse_init: bool= False
    track_activations: bool  = True
    freeze_embedding: bool = False
    # Learned position embeddings added to the token embeddings. toy_model's
    # Transformer has none, so without this every prefix is a bag of words.
    # Needed for grammar; off by default so the other tasks are unchanged.
    positions: bool = False

    def file_name(self, dir=".", include_seed=False) -> str:
        if not os.path.exists(f"./{dir}"):
            os.makedirs(f"./{dir}")
        return f"./{dir}/{self.task}_{self.task_kwargs.__repr__()}{f'_{self.seed}' if include_seed else ''}_{self.gamma}_{self.dim}_{self.h_dim}_{self.n_layers}_{self.low_collapse_init}_{self.num_epochs}_{self.track_activations}{'_pos' if self.positions else ''}"


class WithPositions(nn.Module):
    """Wraps the token embedding so it returns token + learned position embedding.

    Swapped in for model.embeddings, so toy_model.py doesn't need changing and
    model.layers / get_realized / proj all work exactly as before.
    """
    def __init__(self, tok: nn.Embedding, seq_len: int):
        super().__init__()
        self.tok = tok
        self.pos = nn.Embedding(seq_len, tok.embedding_dim)

    def forward(self, token_ids):
        return self.tok(token_ids) + self.pos.weight[: token_ids.shape[1]]


@torch.no_grad()
def evaluate(model, loader, meta, device):
    """Accuracy and mean cross-entropy at target_pos, from the same forward pass.

    Runs in eval mode (attention dropout off) and switches back to train mode.
    Splits over 65535 examples are run in chunks of 2**15, as before; the
    remainder chunk is now included (previously it was skipped).
    Returns (accuracy, loss), or (None, None) if there is no loader.
    """
    if loader is None:
        return None, None
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        step = x.shape[0] if x.shape[0] <= 65535 else 2**15
        for i in range(0, x.shape[0], step):
            logits = model(x[i:i+step])[:, meta.target_pos, :]
            yc = y[i:i+step]
            correct  += (logits.argmax(-1) == yc).sum().item()
            loss_sum += F.cross_entropy(logits, yc, reduction="sum").item()
            total    += yc.numel()
    model.train()
    return correct / total, loss_sum / total


@torch.no_grad()
def realized_without_padding(model, x, pad_id, device):
    """FFN activations on x, keeping only real-token positions (pads dropped).

    In a grammar example most of the seq_len positions are <pad>, so counting
    every position would make the realized regions mostly about padding.
    Returns one (n_real_positions, h_dim) tensor per block, like get_realized().
    """
    model.eval()
    x = x.to(device)
    model(x)
    keep = x != pad_id                                       # (B, T)
    out = [r.reshape(x.shape[0], x.shape[1], -1)[keep] for r in model.get_realized()]
    model.train()
    return out


def run(config: RunConfig, logger=None):

    train_loader, test_loader, meta = make_dataset(config.task, **config.task_kwargs, seed=0)
    is_grammar = config.task == "grammar"

    if is_grammar:
        logger.log(f"Grammar: {meta.n_train} train / {meta.n_test} test examples, "
                   f"vocab {meta.vocab_size}, seq_len {meta.seq_len}")
        logger.log(f"Best possible test loss: {meta.extra['floor_test_nats']:.4f} nats, "
                   f"best possible test accuracy: {meta.extra['acc_ceiling_test']:.2%}")
        if not config.positions:
            logger.log("WARNING: positions=False, so the model can't see word order "
                       "and won't be able to reach the best possible loss.")
        # Fixed inputs for the realized-region analysis: the test prefixes
        # (capped at 2**15 examples so it's one forward pass).
        x_realized = next(iter(test_loader))[0][: 2**15]

    torch.manual_seed(config.seed)

    model = Transformer(
        dim= config.dim,
        h_dim=config.h_dim,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        vocab_size=meta.vocab_size,
        realized=config.track_activations,
        freeze_embedding=config.freeze_embedding
    )
    if config.low_collapse_init:
        with torch.no_grad():
            up_proj = model.layers[0].ffn.l1.weight
            orth = up_proj.T*0.5 # this keeps kaiming uniform distribution
            model.layers[0].ffn.l2.weight = torch.nn.Parameter(orth)
    if config.positions:
        # Created after the Transformer, so the Transformer's own init (and
        # hence its RNG draws) is the same with or without positions.
        model.embeddings = WithPositions(model.embeddings, meta.seq_len)


    model = torch.compile(model).to(config.device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    metrics = {
            "accuracy": [],
            "losses": [],
            "log_freq": config.log_freq
        }
    if config.track_activations:
        metrics["realised_p"] = [[] for _ in model.layers]
    else:
        metrics["collapses"] = [[] for _ in model.layers]
    if is_grammar:
        metrics["test_losses"] = []
        metrics["test_loss_gap"] = []     # test loss minus the best possible
        metrics["agreement"] = []         # dicts: all / no_rc / rc / rc_mismatch

    if len(train_loader) == 1:
        train_loader = [x for x in train_loader]

    for epoch in (range(config.num_epochs)):
        # training
        epoch_loss = 0.0
        epoch_n = 0
        for x, y in train_loader:                  # x: (batch, seq_len) longs, y: (batch,)
            x, y = x.to(config.device), y.to(config.device)
            n = x.shape[0]
            batch_loss = 0.0
            # Gradient accumulation over chunks of size mbs. Each chunk's loss is
            # weighted by its share of the batch, so the gradient equals the
            # full-batch mean gradient, and the last partial chunk is included.
            for i in range(0, n, config.mbs):
                yc = y[i:i+config.mbs]
                logits = model(x[i:i+config.mbs])  # (chunk, seq_len, vocab_size)
                pred   = logits[:, meta.target_pos, :] # the one prediction position
                loss   = F.cross_entropy(pred, yc) * (yc.numel() / n)
                loss.backward()
                batch_loss = batch_loss + loss.detach()
            opt.step()
            if config.gamma < 1:
                model.proj(gamma=config.gamma)
            opt.zero_grad()
            epoch_loss = epoch_loss + batch_loss * n
            epoch_n += n

        if (epoch+1) % config.log_freq==0 and epoch != 0:
            train_loss = (epoch_loss / epoch_n).item()   # mean training loss over the epoch
            if config.task in ("kv", "kv_seq"):
                acc, eval_loss = evaluate(model, train_loader, meta, config.device)
            else:
                acc, eval_loss = evaluate(model, test_loader, meta, config.device)


            logger.log(f"Loss at Epoch {epoch+1}: {train_loss:.5f}")
            logger.log(f"Accuracy at Epoch {epoch+1}: {acc:.2%}")

            if config.track_activations:
                if is_grammar:
                    realized_blocks = realized_without_padding(model, x_realized, meta.extra["pad_id"], config.device)
                else:
                    # other tasks: activations from the evaluation pass above, as before
                    realized_blocks = model.get_realized()
                for b_idx, realized in enumerate(realized_blocks):
                    up_proj = model.layers[b_idx].ffn.l1.weight
                    up_proj_b = model.layers[b_idx].ffn.l1.bias
                    down_proj = model.layers[b_idx].ffn.l2.weight
                    realized_signs = torch.sign(realized).to(torch.int8)
                    unique_signs = torch.unique(realized_signs, dim=0).squeeze()
                    unique_signs = 2*unique_signs - 1
                    #print(unique_signs.shape)
                    logger.log(f"Num Regions Realized: {unique_signs.shape[0]}")
                    with torch.no_grad():
                        realised_pairs_collapsed, R = find_n_pairs_collapsing_torch(up_proj.detach(),
                                                                           up_proj_b.detach(),
                                                                           down_proj.detach(),
                                                                           patterns=unique_signs.detach(),
                                                                           alpha_check=1,
                                                                           verbose=False)
                    logger.log(f"Num Realised Collapsed pairs {epoch+1}: {realised_pairs_collapsed}")
                    logger.log(f"p Realised collapsed pairs {epoch+1}: {realised_pairs_collapsed/math.comb(R,2):.2%}")
                    metrics["realised_p"][b_idx].append(realised_pairs_collapsed/math.comb(R,2))

            else:
                for b_idx, layer in enumerate(model.layers):
                    up_proj = layer.ffn.l1.weight
                    up_proj_b = layer.ffn.l1.bias
                    down_proj = layer.ffn.l2.weight
                    with torch.no_grad():
                        sample_prop = 0.01
                        pairs_collapsed, R = find_n_pairs_collapsing_torch(up_proj.detach(),
                                                                up_proj_b.detach(),
                                                                down_proj.detach(),
                                                                alpha_check=sample_prop,
                                                                verbose=False)
                    logger.log(f"Num Collapsed pairs {epoch+1}: {int(pairs_collapsed/sample_prop)}")
                    logger.log(f"p Collapsed pairs {epoch+1}: {pairs_collapsed/math.comb(R,2):.2%}")
                    metrics["collapses"][b_idx].append(pairs_collapsed)

            if is_grammar:
                floor = meta.extra["floor_test_nats"]
                model.eval()
                agr = agreement_accuracy(model, meta, config.device)
                model.train()
                logger.log(f"Test loss at Epoch {epoch+1}: {eval_loss:.4f} "
                           f"(best possible {floor:.4f}, gap {eval_loss - floor:.4f})")
                logger.log(f"Agreement at Epoch {epoch+1}: all {agr['all']:.2%} | "
                           f"no clause {agr['no_rc']:.2%} | clause {agr['rc']:.2%} | "
                           f"hardest {agr['rc_mismatch']:.2%}")
                metrics["test_losses"].append(eval_loss)
                metrics["test_loss_gap"].append(eval_loss - floor)
                metrics["agreement"].append(agr)

            print()
            metrics["accuracy"].append(acc)
            metrics["losses"].append(train_loss)


    return metrics, model, logger


# Settings per task. Pick one with TASK in main().
PRESETS = {
    "modular": dict(task_kwargs={"p": 17, "op": "add"},      # mlp >= 64
                    mbs=2**13, num_epochs=2**13, log_freq=256),
    # Full-batch, like the other tasks, so one epoch = one optimizer step.
    # 20000 sentences ~ 134k training examples; mbs only splits the forward/
    # backward pass to save memory, the update is still full-batch.
    "grammar": dict(task_kwargs={"n_sentences": 20000, "seq_len": 40, "batch_size": None},
                    mbs=2**13, num_epochs=2**13, log_freq=256, positions=True),
    # "kv":      dict(task_kwargs={"n_keys": 2**12}, ...),    # mlp >= 2/4x n_keys
    # "kv_seq":  dict(task_kwargs={"n_keys": 256}, ...),
    # "boolean": dict(task_kwargs={"n_bits": 4}, ...),        # mlp >= 32
}


def main():
    TASK = "grammar"
    n_runs = 3
    seeds = (torch.arange(n_runs) + 10).tolist()

    for gamma in [1]:
        run_metrics = {}
        config = RunConfig(
                    task = TASK,
                    seed = 0,
                    n_layers=1,
                    **PRESETS[TASK],
                    )

        logger = Logger(hashlib.sha1(config.file_name().encode()).hexdigest(), f"{config.file_name("logs")}.log")

        for seed in seeds:
            config.seed = seed
            logger.log(config.__repr__())
            print(f"seed: {seed}")

            metrics, _, logger = run(config, logger)

            run_metrics[seed] = metrics

        accuracies = [run_metrics[seed]["accuracy"] for seed in seeds]
        losses = [run_metrics[seed]["losses"] for seed in seeds]

        if config.track_activations:
            realised_p = [run_metrics[seed]["realised_p"][0] for seed in seeds]
            create_graphs(accuracies, realised_p, "accuracy", "realised p collapses", None, config.file_name("figs", include_seed=True), config.log_freq)
            create_graphs(losses, realised_p, "losses", "realised p collapses", None, config.file_name("figs", include_seed=True), config.log_freq)
            run_idx=0
            realised_p_per_layer = [run_metrics[seeds[run_idx]]["realised_p"][b_idx] for b_idx in range(config.n_layers)]
            create_graphs(accuracies[run_idx], realised_p_per_layer, "accuracy", "realised p collapses per block", None, config.file_name("figs", include_seed=True), config.log_freq)
            if config.task == "grammar":
                # Accuracy tops out around 50% on grammar, so these are the more
                # informative curves: distance from the best possible loss, and
                # agreement on the hardest case.
                gaps = [run_metrics[seed]["test_loss_gap"] for seed in seeds]
                hardest = [[a["rc_mismatch"] for a in run_metrics[seed]["agreement"]] for seed in seeds]
                create_graphs(gaps, realised_p, "test loss gap to best possible", "realised p collapses", None, config.file_name("figs", include_seed=True), config.log_freq)
                create_graphs(hardest, realised_p, "agreement (hardest case)", "realised p collapses", None, config.file_name("figs", include_seed=True), config.log_freq)
        else:
            collapses = [run_metrics[seed]["collapses"][0] for seed in seeds]
            create_graphs(accuracies, collapses, "accuracy", "collapses", None, config.file_name("figs", include_seed=True), config.log_freq)
            run_idx = 0
            collapses_per_layer = [run_metrics[seeds[run_idx]]["collapses"][b_idx] for b_idx in range(config.n_layers)]
            create_graphs(accuracies[run_idx], collapses_per_layer, "accuracy", "collapses per block", None, config.file_name("figs", include_seed=True), config.log_freq)









if __name__ == "__main__":
    main()