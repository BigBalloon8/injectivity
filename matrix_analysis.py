import torch
from analysis import find_kway_collisions, matrix_from_kernel
from logger import Logger
from scipy.linalg import null_space

def main():
    torch.manual_seed(0)
    
    logger = Logger("matrix_analysis", "./logs/matrix_analysis.log")
    
    A = torch.randn(16, 4)
    b = torch.randn(16)
    B = torch.tensor(matrix_from_kernel(null_space(A.T.numpy())))
    alpha = torch.tensor(
        [0.001, 0.01, 0.1, 1, 0.005, 0.05, 0.5, 5]
    ).sort()[0]
    noise = torch.randn_like(B)
    for a in alpha:
        down = B + a*noise
        num_pairs = find_kway_collisions(A, b, down, logger, pairs=True, cache=False, alpha_check=1)
        logger.log(f"Alpha={a.item():.3f}: {num_pairs}")
        
if __name__ == "__main__":
    main()
