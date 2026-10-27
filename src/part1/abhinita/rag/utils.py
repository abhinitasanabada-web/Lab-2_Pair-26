"""Small helpers shared by the notebook."""
from __future__ import annotations

import subprocess
import sys


def run_info(seed: int = 266) -> dict:
    """Cell-1 information required by the lab (commit, library versions, GPU, seed); never raises."""
    import random
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = ""
    return {"commit": commit or "n/a (not a git checkout)", "python": sys.version.split()[0],
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "seed": seed}
