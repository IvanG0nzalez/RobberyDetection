
import torch
import torch.nn as nn
import numpy as np
import json
import cv2
from pathlib import Path
from torchvision import transforms
from torchvision.models.video import r3d_18, R3D_18_Weights
from PIL import Image
try:
    from src.lstm_classifier import LSTMClassifier
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent))
    from src.lstm_classifier import LSTMClassifier

class HybridRobberyDetector(nn.Module):
    """
    Modelo híbrido end-to-end para detección de robos.
    Combina R3D-18 (extractor de características) + LSTM (clasificador).
    Incluye explicabilidad (XAI) mediante el mecanismo de atención del LSTM.
    """
    def __init__(self, config):
        super(HybridRobberyDetector, self).__init__()
        self.config = config

        # Feature Extractor: R3D-18 sin FC layer
        self.feature_extractor = self._build_feature_extractor()

        # Classifier: LSTM
        classifier_config = config['hybrid_architecture']['classifier']
        self.classifier = LSTMClassifier(
            input_size=classifier_config['input_size'],
            hidden_size=classifier_config['hidden_size'],
            num_layers=classifier_config['num_layers'],
            bidirectional=classifier_config['bidirectional'],
            dropout_fc=classifier_config['dropout_fc'],
            use_attention=classifier_config.get('use_attention', True)
        )

        # Preprocessing
        preproc = config['preprocessing']
        self.transform = transforms.Compose([
            transforms.Resize(tuple(preproc['resize'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=preproc['normalize_mean'], std=preproc['normalize_std'])
        ])

    def _build_feature_extractor(self):
        """Construye el extractor R3D-18 sin la capa FC."""
        weights = R3D_18_Weights.DEFAULT
        model = r3d_18(weights=weights)
        model.fc = nn.Identity()  # Eliminar fully connected layer

        # Congelar parámetros (opcional, para inferencia)
        for param in model.parameters():
            param.requires_grad = False

        return model

    def forward(self, clip_features):
        """
        Forward pass a través de LSTM.

        Args:
            clip_features: Tensor de shape (batch, n_clips, feature_dim)

        Returns:
            Probabilidad de robo (batch,)
        """
        return self.classifier(clip_features)


class RobberyDetector:
    """
    Interfaz de alto nivel para detección de robos en videos, con explicabilidad.
    """
    def __init__(self, model_dir='results/inference'):
        self.model_dir = Path(model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Cargar configuración
        with open(self.model_dir / 'model_config.json', 'r') as f:
            self.config = json.load(f)

        # Cargar modelo híbrido
        self.model = self._load_hybrid_model()
        self.threshold = self.config['threshold']

    def _load_hybrid_model(self):
        """Carga el modelo híbrido completo."""
        model = HybridRobberyDetector(self.config).to(self.device)

        # Cargar pesos de LSTM
        lstm_weights = torch.load(self.model_dir / 'lstm_model.pth', map_location=self.device)
        model.classifier.load_state_dict(lstm_weights)

        model.eval()
        return model

    def predict_from_features(self, features):
        """
        Predice a partir de features pre-extraídos.
        """
        with torch.no_grad():
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            output = self.model(features_tensor)
            prob = output.item()

            # Obtener pesos de atención para XAI
            attention = self.model.classifier.get_attention_weights()
            if attention is not None:
                attn_weights = attention.squeeze(0).cpu().numpy()
                most_important_clip = int(np.argmax(attn_weights))
            else:
                attn_weights = None
                most_important_clip = None

        is_robbery = prob >= self.threshold
        class_name = self.config['class_names'][1] if is_robbery else self.config['class_names'][0]

        return {
            'class': class_name,
            'probability': prob,
            'is_robbery': is_robbery,
            'attention_weights': attn_weights,
            'most_important_clip_idx': most_important_clip
        }

    def predict_from_video(self, video_path):
        """
        Inferencia completa sobre un archivo de video. 
        Aplica procesamiento (extraccion de clips a intervalos segun config de entrenamiento), 
        lo pasa por el extractor R3D_18 para sacar features y finalmente predice con la LSTM.
        Muestra en que segundo ocurrio la situacion mas riesgosa si la clase detectada es Robbery.

        Args:
            video_path: Ruta al archivo de video (.mp4)

        Returns:
            dict: Resultados con predicción, probabilidades y momento detonante (XAI)
        """
        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        vp_config = self.config.get('video_processing', {})
        clip_length = vp_config.get('clip_length', 16)
        stride = vp_config.get('stride', 8)
        overlapping = vp_config.get('overlapping', True)

        # Calcular los intervalos de cada clip simulando el dataset
        clip_intervals = []
        if overlapping:
            start = 0
            while start + clip_length <= total_frames:
                clip_intervals.append((start, start + clip_length))
                start += stride
            if not clip_intervals and total_frames > 0:
                clip_intervals.append((0, total_frames))
        else:
            max_segments = vp_config.get('max_segments_per_video', 32)
            if total_frames > clip_length:
                for i in range(max_segments):
                    start_idx = int(i * (total_frames - clip_length) / max_segments)
                    clip_intervals.append((start_idx, start_idx + clip_length))
            else:
                clip_intervals.append((0, total_frames if total_frames > 0 else 1))

        # Extraer frames, transform y guardar
        clips_tensors = []
        clip_timestamps = [] # Store time in seconds (start, end)

        for start, end in clip_intervals:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            frames = []
            for _ in range(end - start):
                ret, frame = cap.read()
                if not ret: 
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frames.append(self.model.transform(frame))

            if not frames: 
                continue

            # Pad si faltan frames
            while len(frames) < clip_length:
                frames.append(frames[-1])

            # Formato tensor r3d_18: (3, clip_len, H, W)
            clip_tensor = torch.stack(frames, dim=1)
            clips_tensors.append(clip_tensor)

            start_sec = start / fps if fps else 0
            end_sec = end / fps if fps else 0
            clip_timestamps.append((start_sec, end_sec))

        cap.release()

        if not clips_tensors:
            raise ValueError("No se pudieron extraer clips válidos de este video.")

        # Batch features extraction en bloques para no llenar memoria
        clips_batch = torch.stack(clips_tensors)
        batch_size = 16
        features_list = []

        with torch.no_grad():
            from tqdm import tqdm
            print(f"Extrayendo características de {len(clips_batch)} clips...")
            for i in range(0, len(clips_batch), batch_size):
                batch_features = self.model.feature_extractor(clips_batch[i:i+batch_size].to(self.device))
                features_list.append(batch_features.cpu())

        features = torch.cat(features_list, dim=0).numpy()

        # Inferencia con la red LSTM
        result = self.predict_from_features(features)

        # Integrar las marcas de tiempo en segundos (XAI)
        result['clip_timestamps'] = clip_timestamps
        most_important_idx = result['most_important_clip_idx']

        if most_important_idx is not None and most_important_idx < len(clip_timestamps):
            trigger_time = clip_timestamps[most_important_idx]
            result['trigger_time_seconds'] = trigger_time
            if result['is_robbery']:
                result['xai_message'] = f"Robo mayormente detectado entre los segundos {trigger_time[0]:.2f}s y {trigger_time[1]:.2f}s (Clip {most_important_idx})"
            else:
                result['xai_message'] = f"No se detectó ningún robo."
        else:
            result['trigger_time_seconds'] = None
            result['xai_message'] = "No se pudo determinar el momento de mayor alerta."

        return result

    def get_model_info(self):
        """Retorna información del modelo."""
        return {
            'experiment': self.config['experiment_name'],
            'architecture': self.config['hybrid_architecture'],
            'metrics': self.config['performance_metrics']
        }


# Ejemplo de uso
if __name__ == "__main__":
    import os
    print("=" * 50)
    print("Robbery Detection - Full Video Inference (XAI)")
    print("=" * 50)

    detector = RobberyDetector(model_dir=os.path.dirname(os.path.abspath(__file__)))
    info = detector.get_model_info()

    print(f"\nExperimento: {info['experiment']}")
    print(f"Feature Extractor: {info['architecture']['feature_extractor']['backbone']}")
    print(f"Classifier: {info['architecture']['classifier']['type']} (Atención Habilitada: {info['architecture']['classifier'].get('use_attention', True)})")

    # Simulación de Inferencia a un video completo. 
    print("\n" + "=" * 50)
    print("Ejemplo: Predicción de Video End-to-End")
    print("=" * 50)
    print("Para probar desde tu consola ejecuta:\n")
    print(">>> from inference import RobberyDetector")
    print(">>> detector = RobberyDetector(model_dir='.')")
    print(">>> resultado = detector.predict_from_video('ruta_al_video.mp4')")
    print(">>> print(resultado['class'], resultado['xai_message'])\n")
