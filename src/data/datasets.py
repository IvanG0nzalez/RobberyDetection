import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path

class VideoFeatureDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.classes = {'Normal': 0, 'Robbery': 1}
        root_path = Path(root_dir)

        for cls in ["Normal", "Robbery"]:
            cls_dir = root_path / cls
            for file in cls_dir.glob('*.npy'):
                self.samples.append((file, self.classes[cls]))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        path, label = self.samples[index]
        features = np.load(path) #shape: (n_clips, 512)
        features = torch.tensor(features, dtype=torch.float32)
        return features, label
    
def collate_fn(batch):
    features, labels = zip(*batch) #separa features y labels
    
    #aplica padding (relleno con 0s) para que todas las secuencias tengan la misma longitud
    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    return features_padded, labels

def build_dataloaders(features_root, batch_size=32, seed=42):
    root_path = Path(features_root)
    train_dataset = VideoFeatureDataset(root_path / 'train')
    val_dataset = VideoFeatureDataset(root_path / 'val')
    test_dataset = VideoFeatureDataset(root_path / 'test')

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn, 
        generator=g
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, 
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, 
    )

    return train_loader, val_loader, test_loader