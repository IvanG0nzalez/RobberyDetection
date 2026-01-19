import torch
import torchvision
from torchvision.models.video import r3d_18, R3D_18_Weights
from torchvision import transforms
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from tqdm import tqdm

# Clase base pra extracción (estructura)
class BaseFeatureExtractor:
    def __init__(self, device):
        self.device = device
        self.model = self._load_model().to(self.device).eval()

    def _load_model(self):
        raise NotImplementedError("Cada extractor debe implementar su propio método de carga de modelo. (_load_model)")
    
    def __call__(self, clip_path):
        raise NotImplementedError("Cada extractor debe ser 'callable'. (__call__)")
    
# Extractor 1: Solo R3D_18
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


def process_dataset(input_dir, output_dir, extractor):
    """
    Procesa todos los clips en el directorio de entrada, extrae características usando el extractor dado,
    y guarda las características en el directorio de salida.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    for split_dir in input_dir.iterdir():
        if not split_dir.is_dir(): continue
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir(): continue
            
            output_path = output_dir / split_dir.name / class_dir.name
            output_path.mkdir(parents=True, exist_ok=True)

            print(f"Procesando: {class_dir.relative_to(project_root)}")

            for video_folder in tqdm(sorted(class_dir.iterdir()), desc=f"{split_dir.name}/{class_dir.name}"):
                if not video_folder.is_dir(): continue
                
                save_name = output_path / f"{video_folder.name}.npy"
                if save_name.exists():
                    continue

                features_list = []
                for clip_file in sorted(video_folder.glob("*.mp4")):
                    try:
                        feat = extractor(clip_file)
                        features_list.append(feat)
                    except Exception as e:
                        print(f"Error procesando {clip_file}: {e}")

                if features_list:
                    features_array = np.array(features_list)
                    np.save(save_name, features_array)

def main():
    parser = argparse.ArgumentParser(description="Extraer features de un dataset de clips de video.")
    parser.add_argument('--input_dir', type=str, required=True, help="Directorio raíz del dataset de clips con splits.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directorio donde se guardarán los archivos .npy de features.")
    parser.add_argument('--extractor', type=str, choices=['r3d'], default='r3d', help="Tipo de extractor a usar.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    if args.extractor == 'r3d':
        extractor = R3DExtractor(device)
    else:
        raise ValueError(f"Extractor '{args.extractor}' no implementado.")
    
    process_dataset(Path(args.input_dir), Path(args.output_dir), extractor)

if __name__ == "__main__":
    main()
            