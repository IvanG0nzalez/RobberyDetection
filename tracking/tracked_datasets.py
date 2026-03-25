"""
Dataset con tracking de samples para análisis de clips problemáticos.
Permite rastrear qué muestras (archivos .npy) se procesan en cada batch.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path


class TrackedVideoFeatureDataset(Dataset):
    """
    Dataset que devuelve features junto con metadata para tracking.
    
    Retorna: (features, label, sample_info)
    donde sample_info contiene: {'index': int, 'path': str, 'class': str, 'directoryname': str}
    """
    def __init__(self, manifest_csv, split_filter=None, project_root=None):
        """
        Args:
            manifest_csv (str or Path): Ruta al CSV con columnas: class, directoryname, featurepath, split
            split_filter (str, optional): Filtrar por split específico ('train', 'val', 'test')
            project_root (Path, optional): Raíz del proyecto. Si None, usa la ubicación de este archivo
        """
        if project_root is None:
            # Por defecto, buscar la raíz del proyecto (3 niveles arriba: tracking -> Dataset)
            self.project_root = Path(__file__).resolve().parent.parent
        else:
            self.project_root = Path(project_root)
            
        self.samples = []
        self.classes = {'Normal': 0, 'Robbery': 1}
        
        # Leer el manifest
        df = pd.read_csv(manifest_csv)
        
        # Filtrar por split si se especifica
        if split_filter:
            df = df[df['split'] == split_filter]
        
        # Cargar las muestras con metadata
        for idx, row in df.iterrows():
            feature_path = self.project_root / row['featurepath']
            label = self.classes[row['class']]
            
            sample_info = {
                'class': row['class'],
                'directoryname': row['directoryname'],
                'featurepath': str(row['featurepath']),
                'full_path': str(feature_path)
            }
            
            self.samples.append((feature_path, label, sample_info))
        
        print(f"TrackedDataset cargado: {len(self.samples)} muestras" + 
              (f" (split: {split_filter})" if split_filter else ""))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        path, label, sample_info = self.samples[index]
        features = np.load(path)  # shape: (n_clips, 512)
        features = torch.tensor(features, dtype=torch.float32)
        
        # Añadir el índice al sample_info
        sample_info['index'] = index
        
        return features, label, sample_info


def tracked_collate_fn(batch):
    """
    Collate function que maneja features, labels y sample_info.
    
    Returns:
        features_padded: Tensor con features paddeadas
        labels: Tensor con labels
        sample_infos: Lista de diccionarios con información de cada sample
    """
    features, labels, sample_infos = zip(*batch)
    
    # Aplicar padding a las features
    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    
    # sample_infos ya es una tupla de diccionarios, convertir a lista
    sample_infos = list(sample_infos)
    
    return features_padded, labels, sample_infos


def build_tracked_dataloaders(features_manifest, batch_size=32, seed=42, project_root=None):
    """
    Construye dataloaders con tracking de samples.
    
    Args:
        features_manifest (str or Path): Ruta al CSV manifest de features
        batch_size (int): Tamaño del batch
        seed (int): Semilla para reproducibilidad
        project_root (Path, optional): Raíz del proyecto
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_dataset = TrackedVideoFeatureDataset(features_manifest, split_filter='train', project_root=project_root)
    val_dataset = TrackedVideoFeatureDataset(features_manifest, split_filter='val', project_root=project_root)
    test_dataset = TrackedVideoFeatureDataset(features_manifest, split_filter='test', project_root=project_root)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=tracked_collate_fn, 
        generator=g
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=tracked_collate_fn, 
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=tracked_collate_fn, 
    )

    return train_loader, val_loader, test_loader
