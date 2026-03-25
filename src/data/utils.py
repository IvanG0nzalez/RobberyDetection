import os
import csv
import json
import pandas as pd
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


def create_features_manifest(clips_manifest_csv, features_dir, output_csv):
    """
    Genera un manifest de features (.npy) basado en el manifest de clips.
    
    Args:
        clips_manifest_csv (str): Ruta al CSV manifest de clips con columnas: class, directoryname, directorypath, split
        features_dir (str): Directorio raíz donde se encuentran los features (.npy)
        output_csv (str): Ruta donde se guardará el manifest de features
    
    Returns:
        str: Ruta al archivo CSV generado
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    features_path = Path(features_dir).resolve()
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Leer el manifest de clips
    df_clips = pd.read_csv(clips_manifest_csv)
    
    # Crear lista para el nuevo manifest
    features_rows = []
    missing_features = []
    
    print(f"Generando manifest de features desde {clips_manifest_csv}")
    print(f"Buscando archivos .npy en: {features_path}")
    
    for _, row in df_clips.iterrows():
        class_name = row['class']
        dir_name = row['directoryname']
        split = row.get('split', 'unknown')  # Si no tiene split, usar 'unknown'
        
        # Ruta esperada del archivo .npy
        feature_file = features_path / class_name / f"{dir_name}.npy"
        
        if feature_file.exists():
            # Calcular ruta relativa al proyecto
            relative_path = feature_file.relative_to(project_root)
            
            features_rows.append({
                'class': class_name,
                'directoryname': dir_name,
                'featurepath': relative_path.as_posix(),
                'split': split
            })
        else:
            missing_features.append(f"{class_name}/{dir_name}")
    
    # Guardar el manifest de features
    df_features = pd.DataFrame(features_rows)
    df_features.to_csv(output_path, index=False)
    
    # Reportar estadísticas
    print(f"\nManifest de features creado: {output_path}")
    print(f"Total de features encontrados: {len(features_rows)}")
    
    if 'split' in df_features.columns:
        print("\nDistribución por split:")
        print(df_features.groupby(['split', 'class']).size().unstack(fill_value=0))
    
    if missing_features:
        print(f"\nADVERTENCIA: {len(missing_features)} features no encontrados:")
        for mf in missing_features[:10]:  # Mostrar solo los primeros 10
            print(f"  - {mf}")
        if len(missing_features) > 10:
            print(f"  ... y {len(missing_features) - 10} más")
    
    return str(output_path)


def create_base_splits_manifest(input_manifest_csv, output_csv, ratios=(0.64, 0.16, 0.20), random_seed=42):
    """
    Crea un manifest BASE de splits que contiene solo: class, directoryname, split.
    Este manifest es independiente de la configuración de clips y se usa como base
    para todos los experimentos, garantizando la misma distribución de splits.
    
    Args:
        input_manifest_csv (str): Ruta al manifest de clips base
        output_csv (str): Ruta donde guardar el manifest base de splits
        ratios (tuple): Proporciones (train, val, test)
        random_seed (int): Semilla para reproducibilidad
    
    Returns:
        str: Ruta al archivo generado
    """
    from sklearn.model_selection import train_test_split
    import numpy as np
    
    df = pd.read_csv(input_manifest_csv)
    
    # Extraer solo las columnas base: class y directoryname
    df_base = df[['class', 'directoryname']].copy()
    
    # Verificar que hay suficientes muestras
    class_counts = df_base["class"].value_counts()
    if (class_counts < 2).any():
        raise ValueError("Cada clase debe tener al menos 2 muestras para división estratificada.")
    
    # División train/val/test estratificada
    train_val_ratio = ratios[0] + ratios[1]
    test_ratio = ratios[2]
    
    train_val_df, test_df = train_test_split(
        df_base,
        test_size=test_ratio,
        random_state=random_seed,
        stratify=df_base["class"]
    )
    test_df['split'] = 'test'
    
    val_ratio_in_train_val = ratios[1] / train_val_ratio
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_in_train_val,
        random_state=random_seed,
        stratify=train_val_df["class"]
    )
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    
    # Combinar y ordenar
    final_df = pd.concat([train_df, val_df, test_df]).sort_index()
    
    # Guardar solo: class, directoryname, split
    final_df = final_df[['class', 'directoryname', 'split']]
    
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    
    print(f"\nManifest base de splits creado: {output_path}")
    print(f"Columnas: {list(final_df.columns)}")
    print("\nDistribución por split y clase:")
    print(pd.crosstab(final_df['split'], final_df['class']))
    
    return str(output_path)


def enrich_splits_manifest(base_splits_csv, clips_manifest_csv, output_csv):
    """
    Enriquece el manifest base de splits con información específica del experimento:
    directorypath y numclips.
    
    Args:
        base_splits_csv (str): Ruta al manifest base de splits (class, directoryname, split)
        clips_manifest_csv (str): Ruta al manifest de clips del experimento actual
        output_csv (str): Ruta donde guardar el manifest enriquecido
    
    Returns:
        str: Ruta al archivo generado
    """
    # Leer el manifest base de splits
    df_splits = pd.read_csv(base_splits_csv)
    
    # Leer el manifest de clips del experimento actual
    df_clips = pd.read_csv(clips_manifest_csv)
    
    # Hacer merge usando class y directoryname como keys
    df_enriched = df_splits.merge(
        df_clips[['class', 'directoryname', 'directorypath', 'numclips']],
        on=['class', 'directoryname'],
        how='left'
    )
    
    # Verificar que no haya valores faltantes
    if df_enriched['directorypath'].isna().any():
        missing = df_enriched[df_enriched['directorypath'].isna()]
        print(f"\nADVERTENCIA: {len(missing)} registros sin directorypath en el manifest de clips:")
        print(missing[['class', 'directoryname']].head())
    
    # Guardar manifest enriquecido con columnas ordenadas
    df_enriched = df_enriched[['class', 'directoryname', 'directorypath', 'numclips', 'split']]
    
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_enriched.to_csv(output_path, index=False)
    
    print(f"\nManifest de splits enriquecido guardado: {output_path}")
    print(f"Columnas: {list(df_enriched.columns)}")
    print(f"Total de registros: {len(df_enriched)}")
    
    return str(output_path)


def validate_clips_parameters(clips_dir, clip_length, max_segments, overlapping, stride):
    """
    Valida si los clips en un directorio fueron generados con los parámetros especificados.
    Lee el archivo generation_metadata.json guardado durante la generación.
    
    Args:
        clips_dir (str or Path): Directorio que contiene los clips
        clip_length (int): Longitud de clip esperada
        max_segments (int): Número máximo de segmentos esperado
        overlapping (bool): Si se esperan clips con solapamiento
        stride (int): Stride esperado
    
    Returns:
        tuple: (bool, dict) - (parámetros_coinciden, metadata_encontrada)
               Si no existe metadata, devuelve (False, None)
    """
    clips_path = Path(clips_dir)
    metadata_file = clips_path / "generation_metadata.json"
    
    if not metadata_file.exists():
        print(f"No se encontró metadata de generación en: {clips_path}")
        print(f"Los clips pueden haber sido generados con parámetros diferentes")
        return False, None
    
    try:
        with open(metadata_file, 'r') as f:
            saved_metadata = json.load(f)
        
        # Comparar parámetros
        current_params = {
            "clip_length": clip_length,
            "max_segments_per_video": max_segments,
            "overlapping": overlapping,
            "stride": stride if overlapping else None
        }
        
        match = (
            saved_metadata.get("clip_length") == current_params["clip_length"] and
            saved_metadata.get("max_segments_per_video") == current_params["max_segments_per_video"] and
            saved_metadata.get("overlapping") == current_params["overlapping"] and
            saved_metadata.get("stride") == current_params["stride"]
        )
        
        if not match:
            print(f"\nLos clips existentes fueron generados con parámetros DIFERENTES:")
            print(f"  Parámetros guardados:")
            for key, value in saved_metadata.items():
                print(f"    {key}: {value}")
            print(f"  Parámetros actuales:")
            for key, value in current_params.items():
                print(f"    {key}: {value}")
        else:
            print(f"Los clips existentes fueron generados con los MISMOS parámetros")
        
        return match, saved_metadata
        
    except Exception as e:
        print(f"Error al leer metadata: {e}")
        return False, None


def extract_dataset_identifier(raw_videos_path):
    """
    Extrae un identificador único del dataset desde la ruta raw_videos_dir.
    Esto permite diferenciar entre múltiples datasets (original, recortados, etc.).
    
    Args:
        raw_videos_path (str or Path): Ruta al directorio de videos raw
    
    Returns:
        str: Identificador del dataset
    """
    path = Path(raw_videos_path)
    dataset_name = path.name
    
    # Intentar extraer el sufijo después de 'dataset_videos_'
    if dataset_name.startswith('dataset_videos_'):
        identifier = dataset_name.replace('dataset_videos_', '')
    else:
        # Si no sigue el patrón, usar el nombre completo del directorio
        identifier = dataset_name
    
    return identifier


def get_clips_directory_name(clip_length, max_segments, overlapping, stride, dataset_id='original'):
    """
    Genera un nombre de directorio único basado en los parámetros de generación y dataset.
    Útil para tener múltiples versiones de clips con diferentes configuraciones y datasets.
    
    Args:
        clip_length (int): Longitud del clip
        max_segments (int): Número máximo de segmentos
        overlapping (bool): Si usa solapamiento
        stride (int): Stride (solo relevante si overlapping=True)
        dataset_id (str): Identificador del dataset fuente
    
    Returns:
        str: Nombre del directorio
    """
    base_name = f"clips_{dataset_id}"
    suffix = f"_len{clip_length}_seg{max_segments}"
    
    if overlapping:
        suffix += f"_overlap_str{stride}"
    else:
        suffix += "_uniform"
    
    return base_name + suffix


def get_oversample_directory_name(clip_length, max_segments, overlapping, stride, max_ratio, dataset_id='original'):
    """
    Genera un nombre de directorio para clips aumentados (oversample) basado en los parámetros y dataset.
    Los clips aumentados deben ser específicos para cada configuración de clips, dataset Y balance_max_ratio.
    
    Args:
        clip_length (int): Longitud del clip
        max_segments (int): Número máximo de segmentos
        overlapping (bool): Si usa solapamiento
        stride (int): Stride (solo relevante si overlapping=True)
        max_ratio (float): Ratio máximo de balanceo (afecta cantidad de clips generados)
        dataset_id (str): Identificador del dataset fuente
    
    Returns:
        str: Nombre del directorio
    """
    
    base_name = f"clips_oversample_{dataset_id}"
    suffix = f"_len{clip_length}_seg{max_segments}"
    
    if overlapping:
        suffix += f"_overlap_str{stride}"
    else:
        suffix += "_uniform"
    
    ratio_suffix = int(max_ratio * 10)
    suffix += f"_ratio{ratio_suffix}"
    
    return base_name + suffix