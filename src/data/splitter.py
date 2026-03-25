import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split

def split_dataset(
        input_csv,
        output_csv,
        ratios = (0.7, 0.15, 0.15),
        random_seed = 42
):
    """
    Lee un manifiesto CSV y divide los datos en train/val/test de forma estratificada.
    No copia archivos físicamente, solo genera el manifest con la columna 'split'.
    Args:
        input_csv (str): Ruta al archivo CSV manifest del dataset a dividir.
        output_csv (str): Ruta para guardar el nuevo manifest con la columna 'split'.
        ratios (tuple): Tupla con las proporciones para train, val y test. Debe sumar 1.0.
        random_seed (int): Semilla para la reproducibilidad de la división.
    """
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
    
    if output_csv:
        output_csv_path = Path(output_csv)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(output_csv_path, index=False)
        print(f"\nManifiesto con splits guardado en: {output_csv_path.resolve()}")

def main():
    parser = argparse.ArgumentParser(description="Divide un dataset de clips en train/val/test sin copiar archivos.")
    parser.add_argument("--input_csv", type=str, required=True, help="Ruta al archivo CSV manifest del dataset a dividir.")
    parser.add_argument("--output_csv", type=str, required=True, help="Ruta para guardar el nuevo manifest con la columna 'split'.")
    parser.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.15, 0.15], help="Proporciones para train, val y test. Debe sumar 1.0.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para la reproducibilidad de la división.")
    args = parser.parse_args()

    if not np.isclose(sum(args.ratios), 1.0):
        raise ValueError("Las proporciones deben sumar 1.0")

    split_dataset(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        ratios=tuple(args.ratios),
        random_seed=args.seed
    )

if __name__ == "__main__":
    main()