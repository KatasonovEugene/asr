import torch
from torch import nn
import torchaudio
import torch.nn.functional as F


class SpecAugment(nn.Module):
    def __init__(self, freq_masks, time_masks, freq_width, time_width):
        super().__init__()
        self.transforms = torch.nn.Sequential(
            *[torchaudio.transforms.FrequencyMasking(freq_width) for _ in range(freq_masks)],
            *[torchaudio.transforms.TimeMasking(time_width) for _ in range(time_masks)]
        )

    def forward(self, spec):
        return self.transforms(spec)
    

class RandomTimeStretch(nn.Module):
    def __init__(self, min_rate, max_rate, p, hop_length, n_freq):
        super().__init__()
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.p = p
        self.stretch = torchaudio.transforms.TimeStretch(hop_length, n_freq)

    def forward(self, spec):
        if torch.rand(1).item() > self.p:
            return spec
        rate = torch.empty(1).uniform_(self.min_rate, self.max_rate).item()
        return self.stretch(spec, rate).absolute()


