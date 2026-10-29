import torch
from tqdm import tqdm
from scipy.linalg import null_space
import matplotlib.pyplot as plt
import math
import hashlib
from dataclasses import dataclass

from data import make_dataset
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
    mbs: int = 2**13
    
    device: str = "cuda"
    dim: int = 4
    h_dim: int = 16
    n_heads: int = 1
    n_layers: int = 1
    gamma: float = 1.0
    low_collapse_init: bool= False
    track_activations: bool  = True
    
    
    def file_name(self, dir=".", include_seed=False) -> str:
        return f"./{dir}/{self.task}_{self.task_kwargs.__repr__()}{f'_{self.seed}' if include_seed else ''}_{self.gamma}_{self.dim}_{self.h_dim}_{self.n_layers}_{self.low_collapse_init}_{self.num_epochs}_{self.track_activations}"
    
        

@torch.no_grad()
def accuracy(model, loader, meta, device):
    if loader is None:
        return None
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if x.shape[0] > 65535:
            for i in range(x.shape[0]//(2**15)):
                pred = model(x[i*2**15:(i+1)*2**15])[:, meta.target_pos, :].argmax(-1)
                correct += (pred == y[i*2**15:(i+1)*2**15]).sum().item()
                total   += y.numel()
        else:
            pred = model(x)[:, meta.target_pos, :].argmax(-1)
            correct += (pred == y).sum().item()
            total   += y.numel()
    return correct / total


def run(config: RunConfig, logger=None):
    torch.manual_seed(config.seed)
    
    train_loader, test_loader, meta = make_dataset(config.task, **config.task_kwargs, seed=int(config.seed))


    model = Transformer(
        dim= config.dim,
        h_dim=config.h_dim,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        vocab_size=meta.vocab_size,
        realized=config.track_activations
    )
    if config.low_collapse_init:
        with torch.no_grad():
            up_proj = model.layers[0].ffn.l1.weight
            orth = up_proj.T*0.5 # this keeps kaiming uniform distribution
            model.layers[0].ffn.l2.weight = torch.nn.Parameter(orth)
            

    model = torch.compile(model).to(config.device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    metrics = {
            "accuracy": [],
            "losses": [],
            "collapses": [],
            "log_freq": config.log_freq
        }
    if config.track_activations:
        metrics["realised_p"] = []
    
    if len(train_loader) == 1:
        train_loader = [x for x in train_loader]
    
    for epoch in (range(config.num_epochs)):
        # training
        for x, y in train_loader:                  # x: (batch, seq_len) longs, y: (batch,)
            x, y = x.to(config.device), y.to(config.device)
            n_mbs_steps = max(1, x.shape[0]//config.mbs)
            for i in range(n_mbs_steps):
                logits = model(x[i*config.mbs:(i+1)*config.mbs])                      # (batch, seq_len, vocab_size)
                pred   = logits[:, meta.target_pos, :] # the one prediction position
                loss   = torch.nn.functional.cross_entropy(pred, y[i*config.mbs:(i+1)*config.mbs])
                loss.backward()
            opt.step()
            if config.gamma < 1:
                model.proj(gamma=config.gamma)
            opt.zero_grad() 
        
        if (epoch+1) % config.log_freq==0 and epoch != 0: 
            if config.task in ("kv", "kv_seq"):
                acc = accuracy(model, train_loader, meta, config.device)
            else:
                acc = accuracy(model, test_loader, meta, config.device)
            
            up_proj = model.layers[0].ffn.l1.weight
            up_proj_b = model.layers[0].ffn.l1.bias
            down_proj = model.layers[0].ffn.l2.weight
            
            logger.log(f"Loss at Epoch {epoch+1}: {loss.mean().item():.5f}")
            logger.log(f"Accuracy at Epoch {epoch+1}: {acc:.2%}")
            
            if config.track_activations:
                realized_blocks = model.get_realized()
                for realized in realized_blocks:
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
                    metrics["realised_p"].append(realised_pairs_collapsed/math.comb(R,2))
                    
            
            with torch.no_grad():
                sample_prop = 0.01
                pairs_collapsed, R = find_n_pairs_collapsing_torch(up_proj.detach(), 
                                                          up_proj_b.detach(), 
                                                          down_proj.detach(), 
                                                          alpha_check=sample_prop,
                                                          verbose=False)
            logger.log(f"Num Collapsed pairs {epoch+1}: {int(pairs_collapsed/sample_prop)}")
            logger.log(f"p Collapsed pairs {epoch+1}: {realised_pairs_collapsed/math.comb(R,2):.2%}")
            print()
            metrics["accuracy"].append(acc)
            metrics["losses"].append(loss.mean().item())
            metrics["collapses"].append(pairs_collapsed)

    return metrics, model, logger
    
    
# if config.task == "modular":  # mlp >= 64
#     kwargs = {
#         "p": 17,
#         "op": "add",
#     }
# elif config.task == "kv":  # mlp >= 2/4x n_keys
#     kwargs = {
#                 "n_keys": 2**12,  # 32
#                 #"batch_size": 1024
#             }
# elif config.task == "kv_seq":
#     kwargs = {
#         "n_keys": 256  # 32
#     }
# elif config.task == "boolean":  # mlp >= 32
#     kwargs = {
#         "n_bits": 4
#     }
    
    
def main():
    n_runs = 1
    seeds = (torch.arange(n_runs) + 889).tolist()
    
    for gamma in [1, 0.5, 0.2]:
        run_metrics = {}
        config = RunConfig(
                    task = "kv",
                    task_kwargs= {
                                "n_keys": 2**15
                                
                            },
                    gamma=gamma,
                    seed = 0,
                    num_epochs=2**13,
                    log_freq=256
                    )
        
        logger = Logger(hashlib.sha1(config.file_name().encode()).hexdigest(), f"{config.file_name("logs")}.log")
        
        for seed in seeds:
            config.seed = seed
            logger.log(config.__repr__())
            print(f"seed: {seed}")
            
            metrics, _, logger =run(config, logger)
            
            run_metrics[seed] = metrics
        
        accuracies = [run_metrics[seed]["accuracy"] for seed in seeds]
        losses = [run_metrics[seed]["losses"] for seed in seeds]
        collapses = [run_metrics[seed]["collapses"] for seed in seeds]
        realised_p = [run_metrics[seed]["realised_p"] for seed in seeds]
        
        create_graphs(accuracies, collapses, "accuracy", "collapses", None, config.file_name("figs", include_seed=True), config.log_freq)
        create_graphs(accuracies, realised_p, "accuracy", "realised p collapses", None, config.file_name("figs", include_seed=True), config.log_freq)
        create_graphs(losses, realised_p, "losses", "realised p collapses", None, config.file_name("figs", include_seed=True), config.log_freq)
        
        
    
    
    
    

if __name__ == "__main__":
    main()