import torch
#import soundfile as sf
import torchaudio
import pandas as pd


class VocalSetDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file_path, transform=None, target_transform=None):
        self.labels = pd.read_csv(annotations_file_path, names=["path", "label"], dtype = {"path":"string", "label":"int64"})
        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, id):
        wave_path = self.labels.iloc[id, 0]
        wave, sample_rate = torchaudio.load(wave_path)
        label = self.labels.iloc[id, 1]
        
        if self.transform is not None:
            wave = self.transform(wave)
        if self.target_transform is not None:
            label = self.target_transform(label)
        

        return wave, label

    def get_metadata(self):
        sample_wave_path = self.labels.iloc[0, 0]
        metadata = torchaudio.info(sample_wave_path)
        return metadata

    def get_max_duration(self, sample_rate):
        return self._max_duration * sample_rate
