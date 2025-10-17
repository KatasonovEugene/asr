import torch
from src.text_encoder import CTCTextEncoder
from multiprocessing import Pool, cpu_count
from string import ascii_lowercase


def _beam_search_decode_single(args):
    probs, length, ind2char, blank, beam_size = args
    _, V = probs.shape
    beams = {"": (1.0, 0.0)}

    for t in range(int(length)):
        p_t, nextb = probs[t], {}
        for p, (pb, pnb) in beams.items():
            total = pb + pnb
            nb = nextb.get(p, (0.0, 0.0))
            nextb[p] = (nb[0] + total * p_t[blank].item(), nb[1])
            for k in range(1, V):
                c = ind2char[k]
                same = len(p) and p[-1] == c
                s = p if same else p + c
                prob = p_t[k].item() * (pnb if same else total)
                nb = nextb.get(s, (0.0, 0.0))
                nextb[s] = (nb[0], nb[1] + prob)

        beams = dict(sorted(
            ((s, v) for s, v in nextb.items()),
            key=lambda x: x[1][0] + x[1][1],
            reverse=True
        )[:beam_size])

    best = max(beams.items(), key=lambda x: x[1][0] + x[1][1])[0]
    return best.strip()


class CTCTextEncoderWithBeamSearch(CTCTextEncoder):
    EMPTY_TOK = ""

    def __init__(self, beam_width, num_workers=None, alphabet=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")
        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.beam_width = beam_width
        self.num_workers = num_workers if num_workers is not None else cpu_count()

    def encode(self, text):
        text = self.normalize_text(text)
        return torch.tensor([self.char2ind[c] for c in text]).unsqueeze(0)

    def decode(self, log_probs, log_probs_lengths):
        log_probs = log_probs.exp().cpu().detach().numpy()
        args = [
            (log_probs[b, :l], l, self.ind2char, 0, self.beam_width)
            for b, l in enumerate(log_probs_lengths)
        ]
        with Pool(self.num_workers) as pool:
            res = pool.map(_beam_search_decode_single, args)
        return res


