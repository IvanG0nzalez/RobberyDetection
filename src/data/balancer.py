import os
import csv
import random
import shutil
import numpy as np
import cv2
import argparse
from pathlib import Path
from collections import Counter
from typing import List
from tqdm import tqdm

# Funciones utilitarias
def set_seed(seed = 42):
    """Fija la semilla para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    cv2.setRNGSeed(seed)

def read_csv_clips(csv_path):
    """
    Lee un archivo CSV de clips con columnas: class,directoryname,directorypath,numclips 
    y devuelve una lista de diccionarios con los metadatos.
    """
    rows = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        required = {"class", "directoryname", "directorypath", "numclips"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV debe contener columnas: {sorted(required)}")
        for r in reader:
            # normalizar tipos
            r['numclips'] = int(r['numclips']) if str(r['numclips']).strip() != '' else 0
            rows.append(r)
    return rows

def ensure_dir(path):
    """Asegura que un directorio exista, creándolo si es necesario."""
    Path(path).mkdir(parents=True, exist_ok=True)

def copy_dir_clips(src_dir, dst_dir):
    """Copia todos los clips .mp4 de src_dir a dst_dir."""
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)

    ensure_dir(dst_path)
    count = 0

    if not src_path.exists():
        print(f"[Advertencia] Directorio fuente no existe: {src_path}")
        return 0
    
    for src_file in src_path.glob('*.mp4'):
        dst_file = dst_path / src_file.name

        if not dst_file.exists():
            try:
                shutil.copy2(src_file, dst_file)
            except Exception as e:
                print(f"[Error] No se pudo copiar {src_file} a {dst_file}: {e}")
                continue
        
        count += 1
    
    return count

def apply_transforms_to_dir(src_dir, dst_dir, transforms):
    """
    Aplica transformaciones a los clips de src_dir y los guarda en dst_dir.
    Retorna el número de clips generados.
    """
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)

    ensure_dir(dst_path)

    if not src_path.exists():
        print(f"[Advertencia] Directorio fuente no existe: {src_path}")
        return 0
    
    clip_files = sorted(src_path.glob('*.mp4'))
    generated_count = 0

    for clip_file in clip_files:
        dst_file = dst_path / clip_file.name

        cap = cv2.VideoCapture(str(clip_file))
        if not cap.isOpened():
            print(f"[Error] No se pudo abrir el clip: {clip_file}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if not frames:
            print(f"[Advertencia] No se encontraron frames en el clip: {clip_file}")
            continue

        frames_out = frames
        for transform in transforms:
            frames_out = transform(frames_out)

        h, w = frames_out[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        vw = cv2.VideoWriter(str(dst_file), fourcc, fps, (w, h))
        for frame in frames_out:
            vw.write(frame)
        vw.release()

        generated_count += 1

    return generated_count

# Funciones para aumento de datos
def aug_flip(frames):
    """Aplica una transformación de flip horizontal a una lista de frames."""
    return [cv2.flip(frame, 1) for frame in frames]

def aug_rotate(frames, max_deg = 5):
    """Aplica una rotación aleatoria dentro de un rango a una lista de frames."""
    angle = random.uniform(-max_deg, max_deg)
    h, w = frames[0].shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return [cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE) for frame in frames]

def aug_brightness(frames, delta_range=(-35, 35)):
    """Ajusta el brillo de una lista de frames."""
    delta = random.randint(* delta_range)
    return [cv2.convertScaleAbs(frame, alpha=1.0, beta=delta) for frame in frames]

def aug_translate(frames, max_shift: int = 12):
    """Aplica una translación aleatoria a una lista de frames."""
    h, w = frames[0].shape[:2]
    tx = random.randint(-max_shift, max_shift)
    ty = random.randint(-max_shift, max_shift)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return [cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT_101) for frame in frames]

AUG_FUNCS = [
    ('flip', lambda fr: aug_flip(fr)),
    ('rotate', lambda fr: aug_rotate(fr)),
    ('brightness', lambda fr: aug_brightness(fr)),
    ('translate', lambda fr: aug_translate(fr)),
]

def balanced_dataset(input_csv, output_dir, output_csv, mode = "undersample", max_ratio = 1.2, seed = 42):
    """
    Balancea un dataset de clips a partir de un manifiesto CSV.
    Args:
        input_csv (str): Ruta al archivo CSV manifest del dataset.
        output_dir (str): Directorio donde se guardará el dataset balanceado.
        output_csv (str): Ruta al archivo CSV manifest del dataset balanceado.
        mode (str): Modo de balanceo: "undersample", "oversample".
        max_ratio (float): Ratio máximo permitido entre la clase mayoritaria y minoritaria.
        seed (int): Semilla para reproducibilidad.
    """
    assert mode in {"undersample", "oversample"}, "Modo debe ser 'undersample' o 'oversample'"
    set_seed(seed)

    project_root = Path(__file__).resolve().parent.parent.parent

    output_dir_path = Path(output_dir).resolve()
    output_csv_path = Path(output_csv).resolve()

    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = read_csv_clips(input_csv)
    if not rows:
        print(f"[Error] No se encontraron registros en el CSV: {input_csv}")
        return None
    
    by_class = {}
    for row in rows:
        by_class.setdefault(row['class'], []).append(row)

    counts = {cls: len(videos) for cls, videos in by_class.items()}
    if len(counts) < 2:
        print(f"[Error] Se requieren al menos dos clases para balancear. Clases encontradas: {list(counts.keys())}")
        return None
    
    classes_sorted = sorted(counts.items(), key=lambda x: x[1])
    minority_cls, minority_count = classes_sorted[0]
    majority_cls, majority_count = classes_sorted[-1]

    print(f"Clase minoritaria: '{minority_cls}' con {minority_count} videos")
    print(f"Clase mayoritaria: '{majority_cls}' con {majority_count} videos")
    ratio = majority_count / max(1, minority_count)
    print(f"Ratio actual: {ratio:.2f}")

    if mode == "undersample":
        target_max = int(np.ceil(minority_count * max_ratio))

        selected_rows = by_class[minority_cls]

        maj_dirs = by_class[majority_cls][:]
        random.shuffle(maj_dirs)
        selected_rows.extend(maj_dirs[:target_max])

        print(f"Aplicando undersampling. Total de videos seleccionados: {len(selected_rows)}")

        for row in tqdm(selected_rows, desc="Copiando videos (undersample)"):
            src_dir = project_root / row["directorypath"]
            dst_dir = output_dir_path / row["class"] / row["directoryname"]
            copy_dir_clips(src_dir, dst_dir)

            row["directorypath"] = dst_dir.relative_to(project_root).as_posix()

        final_rows = selected_rows

    elif mode == "oversample":
        new_rows = []

        for cls, video_list in by_class.items():
            for row in tqdm(video_list, desc=f"Copiando originales de '{cls}'"):
                src_dir = project_root / row["directorypath"]
                dst_dir = output_dir_path / row["class"] / row["directoryname"]
                copied_count = copy_dir_clips(src_dir, dst_dir)
                new_rows.append({
                    'class': row['class'],
                    'directoryname': row['directoryname'],
                    'directorypath': dst_dir.relative_to(project_root).as_posix(),
                    'numclips': copied_count
                })

        curr_min = minority_count
        curr_maj = majority_count
        aug_index = 1

        pbar = tqdm(total=curr_maj, desc="Generando aumentos (oversample)", initial=curr_min)
        while (curr_maj / max(1, curr_min)) > max_ratio:
            base_video = random.choice(by_class[minority_cls])
            base_dir = project_root / base_video["directorypath"]
            base_name = base_video["directoryname"]

            k = random.randint(1, 2)
            chosen_transforms = random.sample(AUG_FUNCS, k)
            transforms = [fn for _, fn in chosen_transforms]

            aug_name = f"{base_name}_aug{aug_index}"
            aug_dir = output_dir_path / minority_cls / aug_name

            if aug_dir.exists():
                aug_index += 1
                continue

            generated = apply_transforms_to_dir(base_dir, aug_dir, transforms)
            if generated > 0:
                new_rows.append({
                    'class': minority_cls,
                    'directoryname': aug_name,
                    'directorypath': aug_dir.relative_to(project_root).as_posix(),
                    'numclips': generated
                })
                curr_min += 1
                pbar.update(1)
            
            aug_index += 1
        pbar.close()
        final_rows = new_rows
    
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "directoryname", "directorypath", "numclips"])
        writer.writeheader()
        writer.writerows(final_rows)

    final_counts = Counter(row['class'] for row in final_rows)
    print(f"\nBalanceo completado en modo '{mode}'.")
    print(f"Dataset balanceado guardado en: {output_dir_path.resolve()}")
    print(f"Nuevo manifest CSV guardado en: {output_csv_path.resolve()}")
    print("Distribución final de los videos:")
    for cls, c in final_counts.items():
        print(f"  Clase '{cls}': {c} videos")
    
    return str(output_csv_path)

def main():
    parser = argparse.ArgumentParser(description="Balancea un dataset de clips de video usando undersampling u oversampling.")
    parser.add_argument("--input_csv", type=str, required=True, help="Ruta al archivo CSV manifest del dataset original.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directorio donde se guardará el dataset balanceado.")
    parser.add_argument("--output_csv", type=str, required=True, help="Ruta para guardar el nuevo CSV manifest del dataset balanceado.")
    parser.add_argument("--mode", type=str, choices=["undersample", "oversample"], default="undersample", help="Modo de balanceo: 'undersample' o 'oversample'.")
    parser.add_argument("--max_ratio", type=float, default=1.2, help="Ratio máximo deseado entre la clase mayoritaria y la minoritaria.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad.")

    args = parser.parse_args()

    balanced_dataset(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        output_csv=args.output_csv,
        mode=args.mode,
        max_ratio=args.max_ratio,
        seed=args.seed
    )

if __name__ == "__main__":
    main()