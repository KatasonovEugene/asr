import torch
from torch import nn
from pathlib import Path
from hydra.utils import instantiate


class Ensemble(nn.Module):
    def __init__(self, model_dir, aggregate, device, single, **kwargs):
        super().__init__()
        model_dir = Path(model_dir)
        self.aggregate = aggregate
        self.device = device

        model_files = list(model_dir.glob("*.pth"))
        self.models = nn.ModuleList()
        for path in model_files:
            model = instantiate(single)
            state = torch.load(path, map_location=self.device, weights_only=False)
            model.load_state_dict(state['state_dict'])
            model.to(self.device)
            model.eval()
            self.models.append(model)
            print(f"{path} model loaded")

    def forward(self, *args, **kwargs):
        outputs = []
        log_probs_length = None
        for m in self.models:
            out = m(*args, **kwargs)
            outputs.append(out["log_probs"])
            log_probs_length = out["log_probs_length"]

        log_probs = torch.stack(outputs, dim=0)
        if self.aggregate == "mean":
            log_probs = log_probs.mean(dim=0)
        elif self.aggregate == "max":
            log_probs = log_probs.max(dim=0).values

        return dict(log_probs=log_probs, log_probs_length=log_probs_length)

