import os
import urllib.request
import gzip
import shutil
from pathlib import Path

LM_URL = "https://www.openslr.org/resources/11/4-gram.arpa.gz"
VOCAB_URL = "https://www.openslr.org/resources/11/librispeech-vocab.txt"


def download_lm(save_dir="data/lm"):
    if (Path(save_dir) / "librispeech-4gram.arpa").exists():
        return Path(save_dir) / "librispeech-4gram.arpa"

    os.makedirs(save_dir, exist_ok=True)
    archive_path = os.path.join(save_dir, "librispeech-4gram.arpa.gz")
    urllib.request.urlretrieve(LM_URL, archive_path)
    arpa_path = os.path.join(save_dir, "librispeech-4gram.arpa")
    with gzip.open(archive_path, 'rb') as f_in:
        with open(arpa_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(archive_path)
    return arpa_path


def download_lm_vocab(save_dir="data/lm"):
    vocab_path = Path(save_dir) / "librispeech-vocab.txt"
    if Path(vocab_path).exists():
        return vocab_path
    
    urllib.request.urlretrieve(VOCAB_URL, vocab_path)
    return vocab_path


if __name__ == "__main__":
    path = download_lm()
    print(f"LM file saved at: {path}")
    path = download_lm_vocab()
    print(f"Vocab file saved at: {path}")