import torch
from tqdm import tqdm

from data import make_dataset
from toy_model import Transformer
from logger import Logger
from analysis import find_kway_collisions

@torch.no_grad()
def accuracy(model, loader, meta, device):
    if loader is None:
        return None
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)[:, meta.target_pos, :].argmax(-1)
        correct += (pred == y).sum().item()
        total   += y.numel()
    return correct / total

def main():
    task = "kv"

    if task == "modular":  # mlp >= 64
        kwargs = {
            "p": 12,
            "op": "mul",
        }
    elif task == "kv":  # mlp >= 2/4x n_keys
        kwargs = {
            "n_keys": 256  # 32
        }
    elif task == "kv_seq":
        kwargs = {
            "n_keys": 256  # 32
        }
    elif task == "boolean":  # mlp >= 32
        kwargs = {
            "n_bits": 4
        }
    
    train_loader, test_loader, meta = make_dataset(task, **kwargs)

    print(train_loader.batch_size)

    num_epochs = 2**15
    device = "cpu"

    dim=4
    h_dim=16
    n_layers=1

    model = Transformer(
        dim= dim,
        h_dim=h_dim,
        n_heads=1,
        n_layers=n_layers,
        vocab_size=meta.vocab_size
    )
    file_name = lambda folder: f"{folder}/{task}_{kwargs.__repr__()}_{dim}_{h_dim}_{n_layers}_{num_epochs}"
    logger = Logger(task, f"{file_name("logs")}.log")

    #model = torch.compile(model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)

    for epoch in tqdm(range(num_epochs)):
        for x, y in train_loader:                  # x: (batch, seq_len) longs, y: (batch,)
            x, y = x.to(device), y.to(device)
            logits = model(x)                      # (batch, seq_len, vocab_size)
            pred   = logits[:, meta.target_pos, :] # the one prediction position
            loss   = torch.nn.functional.cross_entropy(pred, y)
            loss.backward()
            opt.step()
            opt.zero_grad() 
        if epoch % 64==0: 
            if task in ("kv", "kv_seq"):
                acc = accuracy(model, train_loader, meta, device)
            else:
                acc = accuracy(model, test_loader, meta, device)
            up_proj = model.layers[0].ffn.l1.weight
            up_proj_b = model.layers[0].ffn.l1.bias
            down_proj = model.layers[0].ffn.l2.weight
            with torch.no_grad():
                pairs_collapsed = find_kway_collisions(up_proj.detach(), up_proj_b.detach(), down_proj.detach(), logger, pairs=True, cache=False)
            logger.log(f"Loss at Epoch {epoch+1}: {loss.mean().item():.5f}")
            logger.log(f"Accuracy at Epoch {epoch+1}: {acc:.2%}")
            logger.log(f"Num Collapsed pairs {epoch+1}: {pairs_collapsed}")



    # Checkpoint
    torch.save(model.state_dict(), f"{file_name("models")}.pt")


if __name__ == "__main__":
    main()