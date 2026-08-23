import torch
from pathlib import Path
p = Path("checkpoints/last.pt")
s = torch.load(p, map_location="cpu", weights_only=False)
print("epoch", s["epoch"], "best resume from epoch", s["epoch"] + 1)
