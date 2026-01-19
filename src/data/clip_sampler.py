import os
import cv2
import math
import re
import argparse
from pathlib import Path
from tqdm import tqdm

def save_clip(frames, out_path, fps=30):
    """
    Guarda una lista de frames como un archivo de video MP4
    Args:
        frames (list of np.array): Lista de frames (imágenes) a guardar.
        out_path (str): Ruta de salida para el archivo de video.
        fps (int): Fotogramas por segundo del video de salida.
    """
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()

def simplify_name(filename, cls):
    """
    Crea un nombre de directorio estandarizado basado en la clase y un número extraído del nombre del archivo.
    Args:
        filename (str): Nombre del archivo original.
        cls (str): Clase del archivo ('Robbery' o 'Normal').
    """
    base = os.path.splitext(filename)[0]
    match = re.search(r'(\d+)', base)
    number = match.group(1).zfill(3) if match else '000'

    return f"{cls}_{number}"

def process_video(video_path, output_dir, cls, clip_length, max_segments, fps=30, overlapping=False, stride=8):
    """
    Extrae clips de un solo video y los guarda en el directorio de salida.
    Args:
        video_path (str): Ruta al archivo de video.
        output_dir (str): Directorio donde se guardarán los clips.
        cls (str): Clase del video ('Robbery' o 'Normal').
        clip_length (int): Número de frames por clip.
        max_segments (int): Número máximo de clips a extraer del video.
        fps (int): Fotogramas por segundo para el video de salida.
        overlapping (bool): Si es True, los clips se solaparan.
        stride (int): Número de frames a solapar entre clips si overlapping es True.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    chosen_start_frames = []

    if overlapping:
        possible_start_frames = list(range(0, total_frames - clip_length + 1, stride))
        num_possible_clips = len(possible_start_frames)

        if num_possible_clips == 0:
            print(f"Advertencia: El video {video_path} es demasiado corto para extraer clips.")
            cap.release()
            return
        
        if num_possible_clips <= max_segments:
            chosen_start_frames = possible_start_frames
        else:
            start_index = (num_possible_clips - max_segments) // 2
            chosen_start_frames = possible_start_frames[start_index : start_index + max_segments]
    else:

        num_segments = total_frames // clip_length

        if num_segments == 0:
            print(f"Advertencia: El video {video_path} es demasiado corto para extraer clips.")
            cap.release()
            return
    
        if num_segments <= max_segments:
            chosen = list(range(num_segments))
        else:
            chosen = [math.floor(i * num_segments / max_segments) for i in range(max_segments)]

        chosen_start_frames = [seg_idx * clip_length for seg_idx in chosen]

    video_folder_name = simplify_name(video_path.name, cls)
    video_out_dir = Path(output_dir) / video_folder_name
    video_out_dir.mkdir(parents=True, exist_ok=True)

    for idx, start_frame in enumerate(chosen_start_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for _ in range(clip_length):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        if len(frames) == clip_length:
            out_name = f"{video_folder_name}_clip{idx:03d}.mp4"
            out_path = video_out_dir / out_name
            save_clip(frames, out_path, fps=fps)
        elif len(frames) > 0:
            print(f"Advertencia: Clip incompleto extraído de {video_path} comenzando en frame {start_frame}. Se esperaban {clip_length} frames, pero se obtuvieron {len(frames)}.")
    
    cap.release()

def main():
    """Función principal para procesar videos y extraer clips."""
    parser = argparse.ArgumentParser(description="Extraer clips de videos de un dataset.")
    parser.add_argument('--input_dir', type=str, required=True, help='Ruta al directorio de videos raw. (data/raw/).')
    parser.add_argument('--output_dir', type=str, required=True, help='Ruta al directorio de salida para los clips. (data/interim/).')
    parser.add_argument('--clip_length', type=int, default=16, help='Número de frames por clip.')
    parser.add_argument('--max_segments', type=int, default=32, help='Número máximo de clips a extraer por video.')

    parser.add_argument('--overlapping', action='store_true', help='Usar muestreo denso con solapamiento (stride). Si no se usa, se usará el muestreo uniforme.')
    parser.add_argument('--stride', type=int, default=8, help='Paso (en frames) para el muestreo con solapamiento. Se usa solo si --overlapping está activado.')

    args = parser.parse_args()

    print("Iniciando extracción de clips...")
    print(f"Modo de muestreo: {'Denso (Overlapping)' if args.overlapping else 'Uniforme (Disperso)'}")
    if args.overlapping:
        print(f"Stride: {args.stride}")

    for cls in ['Robbery', 'Normal']:
        class_input_dir = Path(args.input_dir) / cls
        class_output_dir = Path(args.output_dir) / cls

        if not class_input_dir.exists():
            print(f"Advertencia: El directorio de entrada para la clase '{cls}' no existe: {class_input_dir}")
            continue

        class_output_dir.mkdir(parents=True, exist_ok=True)

        video_paths = sorted([f for f in class_input_dir.iterdir() if f.is_file() and f.suffix in ['.mp4', '.avi', '.mov']])

        for video_path in tqdm(video_paths, desc=f"Procesando '{cls}'"):
            process_video(
                video_path,
                class_output_dir,
                cls,
                args.clip_length,
                args.max_segments,
                overlapping=args.overlapping,
                stride=args.stride
            )
    
    print("Extracción de clips completada.")

if __name__ == "__main__":
    main()
