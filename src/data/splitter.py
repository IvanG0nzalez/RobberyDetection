import pandas as pd
import numpy as np
import shutil
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def split_and_copy_dataset(
        input_csv,
        output_dir,
        output_csv,
        ratios = (0.7, 0.15, 0.15),
        random_seed = 42
):
    """
    Lee un manifiesto CSV, divide los datos en train/val/test de forma estratificada,
    y copia los directorios de clips a una nueva estructura de carpetas.
    Args:
        input_csv (str): Ruta al archivo CSV manifest del dataset a dividir.
        output_dir (str): Directorio raíz donde se copiarán los datos divididos.
        output_csv (str): Ruta para guardar el nuevo manifest con la columna 'split'.
        ratios (tuple): Tupla con las proporciones para train, val y test. Debe sumar 1.0.
        random_seed (int): Semilla para la reproducibilidad de la división.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    class_counts = df["class"].value_counts()
    if (class_counts < 2).any():
        raise ValueError("Cada clase debe tener al menos 2 muestras para una división estratificada.")
    
    train_val_ratio = ratios[0] + ratios[1]
    test_ratio = ratios[2]

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        random_state=random_seed,
        stratify=df["class"]
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

    final_df = pd.concat([train_df, val_df, test_df]).sort_index()

    print("\nDistribución de videos por split y clase:")
    print(pd.crosstab(final_df['split'], final_df['class']))

    print(f"\nCopiando directorios de clips a {output_dir_path.resolve()}")
    for _, row in tqdm(final_df.iterrows(), total=len(final_df), desc="Copiando videos"):
        src_dir = project_root / row["directorypath"]

        dst_dir = output_dir_path / row["split"] / row["class"] / row["directoryname"]

        dst_dir.parent.mkdir(parents=True, exist_ok=True)

        if src_dir.exists() and not dst_dir.exists():
            shutil.copytree(src_dir, dst_dir)
    
    if output_csv:
        output_csv_path = Path(output_csv)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(output_csv_path, index=False)
        print(f"\nNuevo manifiesto con splits guardado en: {output_csv_path.resolve()}")

def main():
    parser = argparse.ArgumentParser(description="Divide un dataset de clips en train/val/test y copia los archivos.")
    parser.add_argument("--input_csv", type=str, required=True, help="Ruta al archivo CSV manifest del dataset a dividir.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directorio raíz donde se copiarán los datos divididos.")
    parser.add_argument("--output_csv", type=str, required=True, help="Ruta para guardar el nuevo manifest con la columna 'split'.")
    parser.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.15, 0.15], help="Proporciones para train, val y test. Debe sumar 1.0.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para la reproducibilidad de la división.")
    args = parser.parse_args()

    if not np.isclose(sum(args.ratios), 1.0):
        raise ValueError("Las proporciones deben sumar 1.0")

    split_and_copy_dataset(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        output_csv=args.output_csv,
        ratios=tuple(args.ratios),
        random_seed=args.seed
    )

if __name__ == "__main__":
    main()