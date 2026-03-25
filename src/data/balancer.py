import csv
import random
import shutil
import numpy as np
import pandas as pd
import cv2
import argparse
from pathlib import Path
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
        print(f"Directorio fuente no existe: {src_path}")
        return 0
    
    for src_file in src_path.glob('*.mp4'):
        dst_file = dst_path / src_file.name

        if not dst_file.exists():
            try:
                shutil.copy2(src_file, dst_file)
            except Exception as e:
                print(f"   No se pudo copiar {src_file} a {dst_file}: {e}")
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
        print(f"Directorio fuente no existe: {src_path}")
        return 0
    
    clip_files = sorted(src_path.glob('*.mp4'))
    generated_count = 0

    for clip_file in clip_files:
        dst_file = dst_path / clip_file.name

        cap = cv2.VideoCapture(str(clip_file))
        if not cap.isOpened():
            print(f"   No se pudo abrir el clip: {clip_file}")
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
            print(f"No se encontraron frames en el clip: {clip_file}")
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

def balanced_dataset(input_csv, output_csv, output_oversample_dir=None, mode="undersample", max_ratio=1.2, seed=42):
    """
    Balancea el split de entrenamiento de un dataset de clips a partir de un manifiesto CSV.
    
    - Para undersample: solo modifica el manifest CSV eliminando registros de forma aleatoria de train de la clase mayoritaria.
    - Para oversample: genera clips aumentados solo si no existen, y los guarda en output_oversample_dir.
    
    Args:
        input_csv (str): Ruta al archivo CSV manifest del dataset con columna 'split'.
        output_csv (str): Ruta al archivo CSV manifest del dataset balanceado.
        output_oversample_dir (str): Directorio donde se guardarán los clips aumentados (solo para oversample).
        mode (str): Modo de balanceo: "undersample", "oversample".
        max_ratio (float): Ratio máximo permitido entre la clase mayoritaria y minoritaria.
        seed (int): Semilla para reproducibilidad.
    """
    assert mode in {"undersample", "oversample", "none"}, "Modo debe ser 'undersample', 'oversample' o 'none'"
    
    if mode == "none":
        print("Modo de balanceo: 'none'. No se aplicará balanceo.")
        # Simplemente copiar el CSV de entrada al de salida
        import shutil
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(input_csv, output_csv)
        print(f"Manifest copiado a: {output_csv}")
        return str(output_csv)
    
    set_seed(seed)

    project_root = Path(__file__).resolve().parent.parent.parent
    output_csv_path = Path(output_csv).resolve()
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    
    # Verificar que existe la columna 'split'
    if 'split' not in df.columns:
        raise ValueError("El CSV de entrada debe contener la columna 'split'")
    
    # Separar train de val/test
    train_df = df[df['split'] == 'train'].copy()
    other_df = df[df['split'] != 'train'].copy()
    
    if len(train_df) == 0:
        print("No hay datos de entrenamiento para balancear.")
        df.to_csv(output_csv_path, index=False)
        return str(output_csv_path)

    # Analizar distribución de clases en train
    class_counts = train_df['class'].value_counts().to_dict()
    
    if len(class_counts) < 2:
        print(f"   Se requieren al menos dos clases para balancear. Clases encontradas: {list(class_counts.keys())}")
        df.to_csv(output_csv_path, index=False)
        return str(output_csv_path)
    
    classes_sorted = sorted(class_counts.items(), key=lambda x: x[1])
    minority_cls, minority_count = classes_sorted[0]
    majority_cls, majority_count = classes_sorted[-1]

    print(f"\n=== Balanceo del conjunto de ENTRENAMIENTO ===")
    print(f"Clase minoritaria: '{minority_cls}' con {minority_count} videos")
    print(f"Clase mayoritaria: '{majority_cls}' con {majority_count} videos")
    ratio = majority_count / max(1, minority_count)
    print(f"Ratio actual: {ratio:.2f}")

    if mode == "undersample":
        target_max = int(np.ceil(minority_count * max_ratio))

        # Mantener todos los de la clase minoritaria
        minority_train = train_df[train_df['class'] == minority_cls].copy()
        
        # Submuestrear la clase mayoritaria
        majority_train = train_df[train_df['class'] == majority_cls].copy()
        majority_sampled = majority_train.sample(n=min(target_max, len(majority_train)), random_state=seed)
        
        # Combinar
        balanced_train_df = pd.concat([minority_train, majority_sampled])
        
        print(f"Aplicando undersampling. Videos de train seleccionados: {len(balanced_train_df)}")
        print(f"  - {minority_cls}: {len(minority_train)}")
        print(f"  - {majority_cls}: {len(majority_sampled)}")
        
        # Combinar con val y test
        final_df = pd.concat([balanced_train_df, other_df]).sort_index()

    elif mode == "oversample":
        if not output_oversample_dir:
            raise ValueError("Para oversample se requiere especificar output_oversample_dir")
        
        output_oversample_path = Path(output_oversample_dir).resolve()
        output_oversample_path.mkdir(parents=True, exist_ok=True)
        
        # Crear lista de nuevos registros para clips aumentados
        new_train_rows = []
        
        # Agregar todos los clips originales de train
        for _, row in train_df.iterrows():
            new_train_rows.append(row.to_dict())
        
        curr_min = minority_count
        curr_maj = majority_count
        aug_index = 1
        
        minority_train = train_df[train_df['class'] == minority_cls]

        pbar = tqdm(total=curr_maj, desc="Generando aumentos (oversample)", initial=curr_min)
        while (curr_maj / max(1, curr_min)) > max_ratio:
            base_video = minority_train.sample(n=1, random_state=seed + aug_index).iloc[0]
            base_dir = project_root / base_video["directorypath"]
            base_name = base_video["directoryname"]

            k = random.randint(1, 2)
            chosen_transforms = random.sample(AUG_FUNCS, k)
            transforms = [fn for _, fn in chosen_transforms]

            aug_name = f"{base_name}_aug{aug_index}"
            aug_dir = output_oversample_path / minority_cls / aug_name

            # Solo generar si no existe
            if aug_dir.exists():
                # Ya existe, solo agregarlo al manifest
                num_clips = len(list(aug_dir.glob('*.mp4')))
                new_train_rows.append({
                    'class': minority_cls,
                    'directoryname': aug_name,
                    'directorypath': aug_dir.relative_to(project_root).as_posix(),
                    'numclips': num_clips,
                    'split': 'train'
                })
                curr_min += 1
                pbar.update(1)
                aug_index += 1
                continue

            generated = apply_transforms_to_dir(base_dir, aug_dir, transforms)
            if generated > 0:
                new_train_rows.append({
                    'class': minority_cls,
                    'directoryname': aug_name,
                    'directorypath': aug_dir.relative_to(project_root).as_posix(),
                    'numclips': generated,
                    'split': 'train'
                })
                curr_min += 1
                pbar.update(1)
            
            aug_index += 1
        pbar.close()
        
        balanced_train_df = pd.DataFrame(new_train_rows)
        
        # Combinar con val y test
        final_df = pd.concat([balanced_train_df, other_df])
    
    final_df.to_csv(output_csv_path, index=False)
    
    final_train = final_df[final_df['split'] == 'train']
    final_counts = final_train['class'].value_counts().to_dict()
    print(f"\nBalanceo completado en modo '{mode}'.")
    print(f"Nuevo manifest CSV guardado en: {output_csv_path.resolve()}")
    print("Distribución final de los videos de TRAIN:")
    for cls, c in final_counts.items():
        print(f"  Clase '{cls}': {c} videos")
    
    return str(output_csv_path)

def main():
    parser = argparse.ArgumentParser(description="Balancea el split de entrenamiento de un dataset de clips de video usando undersampling u oversampling.")
    parser.add_argument("--input_csv", type=str, required=True, help="Ruta al archivo CSV manifest del dataset original con columna 'split'.")
    parser.add_argument("--output_csv", type=str, required=True, help="Ruta para guardar el nuevo CSV manifest del dataset balanceado.")
    parser.add_argument("--output_oversample_dir", type=str, default=None, help="Directorio donde se guardarán los clips aumentados (solo para oversample).")
    parser.add_argument("--mode", type=str, choices=["undersample", "oversample", "none"], default="undersample", help="Modo de balanceo: 'undersample', 'oversample' o 'none'.")
    parser.add_argument("--max_ratio", type=float, default=1.2, help="Ratio máximo deseado entre la clase mayoritaria y la minoritaria.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad.")

    args = parser.parse_args()

    balanced_dataset(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        output_oversample_dir=args.output_oversample_dir,
        mode=args.mode,
        max_ratio=args.max_ratio,
        seed=args.seed
    )

if __name__ == "__main__":
    main()