import re
from string import ascii_lowercase
import torch
from transformers import AutoTokenizer


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, use_bpe=False, bpe_model_name="bert-base-uncased", alphabet=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")
        self.alphabet = alphabet
        self.use_bpe = use_bpe

        if self.use_bpe:
            self.tokenizer = AutoTokenizer.from_pretrained(bpe_model_name)
            self.vocab = self.tokenizer.get_vocab().keys()
            self.char2ind = self.tokenizer.get_vocab()
            self.ind2char = {v: k for k, v in self.char2ind.items()}
        else:
            self.vocab = [self.EMPTY_TOK] + list(self.alphabet)
            self.ind2char = dict(enumerate(self.vocab))
            self.char2ind = {v: k for k, v in self.ind2char.items()}
        
    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            if self.use_bpe:
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                return torch.tensor(ids).unsqueeze(0)
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, log_probs, log_probs_lengths):
        inds = log_probs.argmax(dim=-1)
        decoded = []
        for ind, length in zip(inds, log_probs_lengths):
            ind = ind[:length]
            decoded.append(self.ctc_decode(ind))
        return decoded

    def ctc_decode(self, inds) -> str:
        prev_token = None
        decoded = []
        for ind in inds:
            char = self.ind2char[ind.item()]
            if char != self.EMPTY_TOK and char != prev_token:
                decoded.append(char)
            prev_token = char
        return "".join(decoded).strip()

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
