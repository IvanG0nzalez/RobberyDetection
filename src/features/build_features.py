import torch
from torchvision.models.video import r3d_18, R3D_18_Weights
from torchvision import transforms
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from tqdm import tqdm

# Clase base pra extracción
class BaseFeatureExtractor:
    def __init__(self, device):
        self.device = device
        self.model = self._load_model().to(self.device).eval()

    def _load_model(self):
        raise NotImplementedError("Cada extractor debe implementar su propio método de carga de modelo. (_load_model)")
    
    def __call__(self, clip_path):
        raise NotImplementedError("Cada extractor debe ser 'callable'. (__call__)")
    
# Extractor
class R3DExtractor(BaseFeatureExtractor):
    def __init__(self, device):
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
        ])
        super().__init__(device)
    
    def _load_model(self):
        print("Cargando modelo R3D_18")
        weights = R3D_18_Weights.DEFAULT
        model = r3d_18(weights=weights)
        model.fc = nn.Identity()  # Elimina la capa fully connected
        return model
    
    def _load_clip(self, path, clip_len=16):
        cap = cv2.VideoCapture(path)
        frames = []
        for _ in range(clip_len):
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(self.transform(frame))
        cap.release()

        while len(frames) < clip_len:
            frames.append(frames[-1])  # Repetir el último frame si es necesario

        return torch.stack(frames, dim=1).unsqueeze(0)  # Forma: (1, 3, clip_len, H, W)
    
    def __call__(self, clip_path):
        clip_tensor = self._load_clip(clip_path).to(self.device)
        with torch.no_grad():
            features = self.model(clip_tensor)
        return features.cpu().squeeze().numpy()  # Forma: (feature_dim,)


def process_clips_from_manifest(manifest_csv, output_dir, extractor):
    """
    Procesa clips basándose en un manifest CSV y extrae características.
    Guarda las características en output_dir/<class>/<directoryname>.npy
    
    Args:
        manifest_csv: Ruta al CSV con columnas: class, directoryname, directorypath, numclips
        output_dir: Directorio donde se guardarán los archivos .npy
        extractor: Instancia del extractor de features
    """
    import pandas as pd
    
    project_root = Path(__file__).resolve().parent.parent.parent
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(manifest_csv)
    
    print(f"Procesando {len(df)} directorios de clips desde el manifest...")
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Procesando clips"):
        class_name = row['class']
        dir_name = row['directoryname']
        dir_path = project_root / row['directorypath']
        
        # Ruta de salida para este directorio
        output_path = output_dir / class_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        save_name = output_path / f"{dir_name}.npy"
        
        # Si ya existe, omitir
        if save_name.exists():
            continue
        
        # Procesar todos los clips del directorio
        features_list = []
        for clip_file in sorted(dir_path.glob("*.mp4")):
            try:
                feat = extractor(str(clip_file))
                features_list.append(feat)
            except Exception as e:
                print(f"Error procesando {clip_file}: {e}")
        
        # Guardar features si se procesaron clips
        if features_list:
            features_array = np.array(features_list)
            np.save(save_name, features_array)


def main():
    parser = argparse.ArgumentParser(description="Extraer features de un dataset de clips de video.")
    parser.add_argument('--manifest_csv', type=str, default=None, help="Ruta al manifest CSV con los clips a procesar.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directorio donde se guardarán los archivos .npy de features.")
    parser.add_argument('--extractor', type=str, choices=['r3d'], default='r3d', help="Tipo de extractor a usar.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    if args.extractor == 'r3d':
        extractor = R3DExtractor(device)
    else:
        raise ValueError(f"Extractor '{args.extractor}' no implementado.")
    
    if args.manifest_csv:
        # Nuevo método: procesar desde manifest
        print(f"Procesando clips desde manifest: {args.manifest_csv}")
        process_clips_from_manifest(args.manifest_csv, Path(args.output_dir), extractor)
    else:
        raise ValueError("Debe especificar --manifest_csv o --input_dir")

if __name__ == "__main__":
    main()
            