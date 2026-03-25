import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path


class VideoFeatureDataset(Dataset):
    def __init__(self, manifest_csv, split_filter=None):
        """
        Dataset que lee features desde un manifest CSV.
        
        Args:
            manifest_csv (str or Path): Ruta al CSV con columnas: class, directoryname, featurepath, split
            split_filter (str, optional): Filtrar por split específico ('train', 'val', 'test'). Si es None, usa todos.
        """
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.samples = []
        self.classes = {'Normal': 0, 'Robbery': 1}
        
        # Leer el manifest
        df = pd.read_csv(manifest_csv)
        
        # Filtrar por split si se especifica
        if split_filter:
            df = df[df['split'] == split_filter]
        
        # Cargar las muestras
        for _, row in df.iterrows():
            feature_path = self.project_root / row['featurepath']
            label = self.classes[row['class']]
            self.samples.append((feature_path, label))
        
        #print(f"Dataset cargado: {len(self.samples)} muestras" + 
        #      (f" (split: {split_filter})" if split_filter else ""))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        path, label = self.samples[index]
        features = np.load(path)  # shape: (n_clips, 512)
        features = torch.tensor(features, dtype=torch.float32)
        return features, label
    
def collate_fn(batch):
    features, labels = zip(*batch)  # separa features y labels
    
    # aplica padding (relleno con 0s) para que todas las secuencias tengan la misma longitud
    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    return features_padded, labels


def build_dataloaders(features_manifest, batch_size=32, seed=42):
    """
    Construye dataloaders desde un manifest de features.
    
    Args:
        features_manifest (str or Path): Ruta al CSV manifest de features
        batch_size (int): Tamaño del batch
        seed (int): Semilla para reproducibilidad
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_dataset = VideoFeatureDataset(features_manifest, split_filter='train')
    val_dataset = VideoFeatureDataset(features_manifest, split_filter='val')
    test_dataset = VideoFeatureDataset(features_manifest, split_filter='test')

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