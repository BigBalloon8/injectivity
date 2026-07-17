import torch
from analysis import find_kway_collisions
from logger import Logger

def main():
    p = {"lr": 0.001}
    A = torch.randn(16,4, requires_grad=True)
    b = torch.randn(16, requires_grad=True)
    B = torch.randn(4, 16, requires_grad=True)
    opt = torch.optim.Adam([A, b, B], lr=p["lr"])
    
    logger = Logger("Random Walk", f"./logs/random_walk_{p}.log")
    
    epochs = 2**13
    
    for i in range(epochs):
        if i%64 ==0 and i > 576:
            with torch.no_grad():
                pairs_collapsed = find_kway_collisions(A.detach(), b.detach(), B.detach(), logger, pairs=True, cache=False)
            logger.log(f"Step {i}: {pairs_collapsed}")
        A.grad = 1-2*torch.rand_like(A)
        b.grad = 1-2*torch.rand_like(b)
        B.grad = 1-2*torch.rand_like(B)
        opt.step()
        opt.zero_grad()
    
if __name__ == "__main__":
    main()
