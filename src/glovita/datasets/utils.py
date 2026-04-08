import random

import numpy as np
import torch


def seed_worker(worker_id):
    """Per-worker RNG seeding for reproducible DataLoader behaviour."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
