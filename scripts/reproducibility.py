import random
import numpy as np
import torch

def seed_all(seed: int = 2026):
    """
    Set all random seeds for reproducibility.

    This ensures:
    - Same model initialization
    - Same data shuffling
    - Same training results (as much as possible)

    Args:
        seed (int): Random seed value
    """

    # Python random
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch GPU (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensures deterministic behavior (slightly slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[INFO] Random seed set to {seed}")