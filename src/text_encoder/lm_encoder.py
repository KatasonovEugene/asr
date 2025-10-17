from pyctcdecode import build_ctcdecoder
from multiprocessing import Pool, cpu_count
from src.text_encoder.ctc_text_encoder import CTCTextEncoder


def _lm_decode_single(args):
    decoder, probs, beam_width = args
    return decoder.decode(probs, beam_width=beam_width).strip()


class CTCTextEncoderWithLM(CTCTextEncoder):
    def __init__(self, beam_width, kenlm_model_path, vocab_path, alpha=0.5, beta=1.0, num_workers=None, *args, **kwargs):
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
        self.num_workers = num_workers if num_workers is not None else cpu_count()

    def decode(self, log_probs, log_probs_lengths):
        probs = log_probs.exp().cpu().detach().numpy()
        decode_args = []

        for i, length in enumerate(log_probs_lengths):
            sample_probs = probs[i][:length]
            decode_args.append((self.decoder, sample_probs, self.beam_width))

        with Pool(self.num_workers) as pool:
            decoded = pool.map(_lm_decode_single, decode_args)

        return decoded

    def _normalize(self, vocab_path):
        grams = open(self.kenlm_model_path).read()
        uni = open(vocab_path).readlines()

        grams = grams.lower().replace("'", "")
        with open(self.kenlm_model_path, "w") as file:
            file.write(grams)

        uni = list(filter(lambda s : len(s) > 0, uni))
        return uni