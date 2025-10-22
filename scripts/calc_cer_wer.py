import hydra
from pathlib import Path
import re
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.metrics.utils import calc_cer, calc_wer


def normalize_text(text):
    text = text.strip().lower()
    text = re.sub(r"[^a-z ]", "", text)
    return text


@hydra.main(version_base=None, config_path="../src/configs", config_name="calc_cer_wer")
def evaluate(cfg):
    pred_path = cfg.prediction_path
    trg_path = cfg.target_path
    cer, wer, total = 0, 0, 0

    for pred_file in Path(pred_path).iterdir():
        trg_file = Path(trg_path) / pred_file.name

        pred = normalize_text(pred_file.open().read())
        trg = normalize_text(trg_file.open().read())

        cer += calc_cer(trg, pred)
        wer += calc_wer(trg, pred)

        total += 1

    cer /= total
    wer /= total

    print(f"CER: {cer:.4f}")
    print(f"WER: {wer:.4f}")


if __name__ == "__main__":
    evaluate()
