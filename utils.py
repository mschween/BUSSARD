import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def date_folders(base_dir):
    # Create log path with timestamp
    day_timestamp = datetime.now().strftime("%Y%m%d")
    hour_timestamp = datetime.now().strftime("%H%M%S")

    # Create log directory if it doesn't exist
    out_path = Path(base_dir, day_timestamp, hour_timestamp)
    out_path.mkdir(parents=True, exist_ok=True)

    return out_path


def set_seed(seed: int):
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # For single GPU
    torch.cuda.manual_seed_all(seed)  # For multiple GPUs
    torch.backends.cudnn.deterministic = True
