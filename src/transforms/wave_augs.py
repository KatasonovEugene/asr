from torch import nn
from torch_audiomentations import AddColoredNoise, PitchShift, Gain

class AddWhiteNoise(nn.Module):
    def __init__(self, min_snr_db, max_snr_db, p):
        super().__init__()
        self.add_noise = AddColoredNoise(
            min_snr_db=min_snr_db,
            max_snr_db=max_snr_db,
            p=p
        )

    def forward(self, audio):
        return self.add_noise(audio)


class RandomPitchShift(nn.Module):
    def __init__(self, min_transpose_semitones, max_transpose_semitones, p):
        self.pitch_shift = PitchShift(
            min_transpose_semitones=min_transpose_semitones,
            max_transpose_semitones=max_transpose_semitones,
            p=p
        )
        self.sample_rate = 16000

    def forward(self, audio):
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)
        return self.pitch_shift(audio, sample_rate=self.sample_rate).squeeze(0)
    

class Gain(nn.Module):
    def __init__(self, p):
        super().__init__()
        self._aug = Gain(p)

    def __call__(self, data):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)