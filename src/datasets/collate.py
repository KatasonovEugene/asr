import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    result_batch = {}
    result_batch['audio'] = [item['audio'] for item in dataset_items]
    result_batch['audio_path'] = [item['audio_path'] for item in dataset_items]
    
    text_encoded_tensors = [torch.tensor(item['text_encoded'], dtype=torch.long).squeeze(0) for item in dataset_items]
    result_batch['text'] = [item['text'] for item in dataset_items]
    result_batch['text_encoded'] = pad_sequence(text_encoded_tensors, batch_first=True, padding_value=0)
    result_batch['text_encoded_length'] = torch.tensor([item['text_encoded'].numel() for item in dataset_items])
    
    specs = [item['spectrogram'].squeeze(0).transpose(0, 1) for item in dataset_items]
    result_batch['spectrogram'] = pad_sequence(specs, batch_first=True, padding_value=0)
    result_batch['spectrogram_length'] = torch.tensor([spec.shape[0] for spec in specs])
    
    return result_batch