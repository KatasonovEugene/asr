import torch
import re
from string import ascii_lowercase
from pyctcdecode import build_ctcdecoder
from multiprocessing import Pool
from src.text_encoder.ctc_text_encoder import CTCTextEncoder

def _decode_single(args):
    decoder, probs, beam_width = args
    return decoder.decode(probs, beam_width=beam_width).strip()

class CTCTextEncoderWithLM(CTCTextEncoder):
    def __init__(self, alpha, beta, beam_width, kenlm_model_path, vocab_path, num_workers, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert kenlm_model_path is not None

        self.kenlm_model_path = kenlm_model_path
        unigrams = self._normalize(vocab_path)

        self.decoder = build_ctcdecoder(
            labels=self.vocab,
            kenlm_model_path=kenlm_model_path,
            unigrams=unigrams,
            alpha=alpha,
            beta=beta
        )
        self.beam_width = beam_width
        self.num_workers = num_workers

    def decode(self, log_probs, log_probs_lengths):
        probs = log_probs.exp().cpu().detach().numpy()
        decode_args = []

        for i, length in enumerate(log_probs_lengths):
            sample_probs = probs[i][:length]
            decode_args.append((self.decoder, sample_probs, self.beam_width))

        if self.num_workers > 1:
            with Pool(self.num_workers) as pool:
                decoded = pool.map(_decode_single, decode_args)
        else:
            decoded = [_decode_single(args) for args in decode_args]

        return decoded

    def _normalize(self, vocab_path):
        grams = open(self.kenlm_model_path).read()
        uni = open(vocab_path).readlines()

        grams = grams.lower().replace("'", "")
        with open(self.kenlm_model_path, "w") as file:
            file.write(grams)

        uni = list(filter(lambda s : len(s) > 0, uni))
        return uni