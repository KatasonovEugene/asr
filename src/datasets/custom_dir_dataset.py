from pathlib import Path
import torchaudio

from src.datasets.base_dataset import BaseDataset


class CustomDirAudioDataset(BaseDataset):
    def __init__(self, directory, *args, **kwargs):
        assert directory is not None, "Set directory name in config"
        audio_dir = Path(directory) / "audio"
        transcription_dir = Path(directory) / "transcriptions"
        data = []
        for path in Path(audio_dir).iterdir():
            if path.suffix in [".mp3", ".wav", ".flac", ".m4a"]:
                audio_path = str(path.absolute().resolve())
                text = ""
                if transcription_dir and Path(transcription_dir).exists():
                    transc_path = Path(transcription_dir) / (path.stem + ".txt")
                    if transc_path.exists():
                        with transc_path.open() as f:
                            text = f.read().strip()
                t_info = torchaudio.info(audio_path)
                data.append(
                    {
                        "path": audio_path,
                        "text": text,
                        "audio_len": t_info.num_frames / t_info.sample_rate,
                    }
                )
        super().__init__(data, *args, **kwargs)