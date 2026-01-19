import os
import csv
from collections import Counter
from pathlib import Path

def create_dataset_manifest(input_dir, output_csv):
    """
    Escanea un directorio de clips estructurado (clase/video/clips)
    y genera un archivo CSV con metadatos.
    Args:
        input_dir (str): Ruta al directorio raíz que contiene las carpetas de clases.
        output_csv (str): Ruta donde se guardará el archivo CSV resultante.
    """
    project_root = Path(__file__).resolve().parent.parent.parent

    input_path = Path(input_dir)
    output_path = Path(output_csv)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"Escaneando directorio: {input_path}")

    for class_dir in sorted(input_path.iterdir()):
        if not class_dir.is_dir():
            continue

        cls = class_dir.name
        for video_dir in sorted(class_dir.iterdir()):
            if not video_dir.is_dir():
                continue

            clip_count = len(list(video_dir.glob("*.mp4")))

            relative_path = video_dir.resolve().relative_to(project_root)

            rows.append([cls, video_dir.name, relative_path.as_posix(), clip_count])
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "directoryname", "directorypath", "numclips"])
        writer.writerows(rows)

    class_counts = Counter(row[0] for row in rows)

    print(f"Archivo CSV creado con {len(rows)} registros en: {output_path}")
    for cls, count in class_counts.items():
        print(f"  Clase '{cls}': {count} videos")
    
    return str(output_path)