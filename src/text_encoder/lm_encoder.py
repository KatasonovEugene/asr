import torch
import re
from string import ascii_lowercase
from pyctcdecode import build_ctcdecoder


class CTCTextEncoderWithLM:
    EMPTY_TOK = ""

    def __init__(self, alpha, beta, beam_width, kenlm_model_path, alphabet=None, **kwargs):
        assert kenlm_model_path is not None

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        self.decoder = build_ctcdecoder(
            labels=self.vocab,
            kenlm_model_path=kenlm_model_path,
            alpha=alpha,
            beta=beta
        )
        self.beam_width = beam_width

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, log_probs, log_probs_lengths):
        probs = log_probs.exp().cpu().detach().numpy()
        decoded = []
        for i, length in enumerate(log_probs_lengths):
            sample_probs = probs[i][:length]
            text = self.ctc_decode(sample_probs)
            decoded.append(text)
        return decoded

    def ctc_decode(self, probs):
        text = self.decoder.decode(probs, beam_width=self.beam_width)
        return text.strip()

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
