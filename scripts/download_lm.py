import os
import urllib
import gzip
import shutil

LM_URL = "https://www.openslr.org/resources/11/4-gram.arpa.gz"


def download_lm(save_dir="data/lm"):
    os.makedirs(save_dir, exist_ok=True)
    archive_path = os.path.join(save_dir, "librispeech-4gram.arpa.gz")
    urllib.request.urlretrieve(LM_URL, archive_path)
    arpa_path = os.path.join(save_dir, "librispeech-4gram.arpa")
    with gzip.open(archive_path, 'rb') as f_in:
        with open(arpa_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(archive_path)
    return arpa_path


if __name__ == "__main__":
    path = download_lm()
    print(f"LM file saved at: {path}")
