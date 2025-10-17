from typing import List
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_cer, calc_wer


class CER(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        cers = []
        predictions = self.text_encoder.decode(log_probs, log_probs_length)
        for pred_text, target_text in zip(predictions, text):
            target_text = self.text_encoder.normalize_text(target_text)
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)

class WER(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        wers = []
        predictions = self.text_encoder.decode(log_probs, log_probs_length)
        for pred_text, target_text in zip(predictions, text):
            target_text = self.text_encoder.normalize_text(target_text)
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)
