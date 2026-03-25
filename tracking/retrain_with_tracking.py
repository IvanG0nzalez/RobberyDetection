"""
Script principal del flujo de analisis de clips problematicos.

Flujo completo:
  1. Lee la configuracion del experimento (experiments/exp_XX/)
  2. Replica el entrenamiento con tracking de cada sample en cada epoch
  3. Analiza automaticamente los clips problematicos
  4. Genera exclusion_list.csv con los clips recomendados para eliminar

El output se guarda en tracking/output/{exp_name}/:
    replication_config.json
    tracking/test_sample_losses.json
    tracking/training_tracking_detailed.json
    analysis/exclusion_list.csv
    analysis/test_problematic_list.csv
"""

import argparse
import yaml
import json
import torch
import shutil
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYZE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ANALYZE_DIR))

from tracked_trainer import train_model_with_tracking
import pandas as pd
import numpy as np

# Funcion de deteccion de clips problematicos
def detect_problematic_clips(tracking_dir, analysis_dir, manifest_file, loss_pct=90, conf_error_threshold=0.35, min_error_freq=1):
    tracking_dir = Path(tracking_dir)
    analysis_dir = Path(analysis_dir)

    manifest_df = pd.read_csv(manifest_file)
    manifest_df['clip_id'] = manifest_df['directoryname'].astype(str)
    manifest_df['video_id'] = manifest_df['directoryname'].astype(str).str.replace(r'_clip_\d+$', '', regex=True)

    losses_records = []
    
    # Extraer Test losses
    tracked_test_losses_path = tracking_dir / 'test_sample_losses.json'
    if tracked_test_losses_path.exists():
        with open(tracked_test_losses_path, 'r', encoding='utf-8') as f:
            for s in json.load(f):
                losses_records.append({
                    'clip_id': str(s['directoryname']),
                    'loss': float(s['loss']),
                    'prediction': float(s.get('prediction', np.nan))
                })

    # Extraer Train y Val losses
    tracking_detailed_path = tracking_dir / 'training_tracking_detailed.json'
    if tracking_detailed_path.exists():
        with open(tracking_detailed_path, 'r', encoding='utf-8') as f:
            detailed = json.load(f)
            
        # Train (contiene perdida y predicción por batch)
        for batch in detailed.get('epoch_batch_losses', []):
            for s in batch.get('samples', []):
                losses_records.append({
                    'clip_id': str(s['directoryname']),
                    'loss': float(s['loss']),
                    'prediction': float(s.get('prediction', np.nan))
                })
                
        # Val (contiene promedios globales de perdida, sin predicción individual por batch)
        for val_epoch in detailed.get('epoch_sample_stats_val', []):
            for s in val_epoch:
                losses_records.append({
                    'clip_id': str(s['sample']),
                    'loss': float(s['mean_loss']),
                    'prediction': np.nan
                })

    losses_df = pd.DataFrame(losses_records)
    if losses_df.empty:
        print("No se encontraron registros de pérdidas.")
        return pd.DataFrame()

    master_df = losses_df.merge(
        manifest_df[['clip_id', 'video_id', 'class', 'split', 'featurepath']],
        on='clip_id',
        how='left',
        suffixes=('', '_manifest')
    )

    # Computar conf_error: para 'val' u omitidos asume un default (1.0) para no perderlos
    def compute_conf_error(s):
        clean = s.dropna()
        if clean.empty:
            return 1.0
        return np.mean(np.abs(clean - 0.5))

    clip_rank_df = (
        master_df.groupby('clip_id', as_index=False)
        .agg(
            mean_loss=('loss', 'mean'),
            max_loss=('loss', 'max'),
            conf_error=('prediction', compute_conf_error),
            error_freq=('loss', lambda s: np.mean(s > s.quantile(0.75))),
            cls=('class', 'first'),
            split=('split', 'first'),
            video_id=('video_id', 'first'),
        )
        .sort_values('mean_loss', ascending=False)
    )

    loss_threshold = clip_rank_df['mean_loss'].quantile(loss_pct / 100.0)

    rules_df = clip_rank_df.copy()
    rules_df['rule_high_loss'] = rules_df['mean_loss'] >= loss_threshold
    rules_df['rule_conf_error'] = rules_df['conf_error'] >= conf_error_threshold
    rules_df['rule_error_freq'] = rules_df['error_freq'] >= min_error_freq

    rules_df['n_rules_triggered'] = (
        rules_df[['rule_high_loss', 'rule_conf_error', 'rule_error_freq']]
        .sum(axis=1)
    )

    # Decision: filtrar si cumple al menos 2 reglas, sino mantener
    rules_df['decision'] = np.where(rules_df['n_rules_triggered'] >= 2, 'filter', 'keep')

    exclusion_full_df = rules_df[rules_df['decision'] == 'filter'].copy()
    
    # Separar en lista de exclusión (train/val) y lista de problemáticos (test)
    exclusion_df = exclusion_full_df[exclusion_full_df['split'].isin(['train', 'val'])].copy()
    test_problematic_df = exclusion_full_df[exclusion_full_df['split'] == 'test'].copy()
    
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar lista de exclusión para reentrenamiento (solo train y val)
    exclusion_csv_path = analysis_dir / 'exclusion_list.csv'
    exclusion_df.to_csv(exclusion_csv_path, index=False)
    
    # Guardar lista referencial de evaluación para análisis extras
    test_problematic_path = analysis_dir / 'test_problematic_list.csv'
    test_problematic_df.to_csv(test_problematic_path, index=False)
    
    return exclusion_df

# Carga de configuracion del experimento
def load_experiment_config(exp_dir):
    """
    Extrae toda la informacion necesaria para replicar el entrenamiento.

    Lee:
        exp_dir/config_run.yml          -> parametros de entrenamiento y semilla
        exp_dir/results/tables/lstm_final_metrics.json -> mejores hiperparametros
        exp_dir/results/tables/manifest_features.csv   -> manifest de features

    Returns:
        (seed, training_params, model_config, manifest_path, replication_info)
    """
    exp_dir = Path(exp_dir)

    config_file = exp_dir / 'config_run.yml'
    if not config_file.exists():
        raise FileNotFoundError(f"No se encontro config_run.yml en {exp_dir}")
    with open(config_file, 'r') as f:
        run_config = yaml.safe_load(f)

    seed = run_config.get('random_seed', 42)
    training_params = run_config.get('training', {})
    num_epochs = training_params.get('epochs', 60)
    patience = training_params.get('patience', 10)

    metrics_file = exp_dir / 'results' / 'tables' / 'lstm_final_metrics.json'
    if metrics_file.exists():
        with open(metrics_file, 'r', encoding='utf-8-sig') as f:
            metrics = json.load(f)
        best_hp = metrics.get('best_hyperparameters', {})
        original_test_metrics = metrics.get('test_metrics', {})
    else:
        best_hp = {}
        original_test_metrics = {}
        print("  AVISO: No se encontro lstm_final_metrics.json, usando hiperparametros por defecto")

    manifest_file = exp_dir / 'results' / 'tables' / 'manifest_features.csv'
    if not manifest_file.exists():
        raise FileNotFoundError(
            f"No se encontro manifest_features.csv en {exp_dir / 'results' / 'tables'}"
        )

    model_config = {
        'hidden_size':   best_hp.get('hidden_size', 128),
        'num_layers':    best_hp.get('num_layers', 2),
        'bidirectional': best_hp.get('bidirectional', True),
        'dropout_fc':    best_hp.get('dropout_fc', 0.3),
        'lr':            best_hp.get('lr', 0.001),
        'weight_decay':  best_hp.get('weight_decay', 1e-5),
        'batch_size':    best_hp.get('batch_size', 32),
        'input_size':    run_config.get('lstm_model', {}).get('input_size', 512),
    }

    replication_info = {
        'source_experiment': exp_dir.name,
        'seed': seed,
        'num_epochs': num_epochs,
        'patience': patience,
        'model_config': model_config,
        'original_test_metrics': original_test_metrics,
        'manifest_features': str(manifest_file.relative_to(PROJECT_ROOT)),
    }

    return seed, num_epochs, patience, model_config, manifest_file, replication_info


# Helpers
def _print_config(exp_name, seed, num_epochs, patience, model_config, manifest_file, output_dir):
    print(f"\n{'='*70}")
    print(f"RE-ENTRENAMIENTO CON TRACKING: {exp_name}")
    print(f"{'='*70}")
    print(f"\nExperimento fuente : {exp_name}")
    print(f"Semilla            : {seed}")
    print(f"Epocas             : {num_epochs}")
    print(f"Patience           : {patience}")
    print(f"\nHiperparametros del modelo:")
    for k, v in model_config.items():
        print(f"  {k}: {v}")
    print(f"\nManifest de features: {manifest_file.name}")
    print(f"\nOutput dir: {output_dir.relative_to(PROJECT_ROOT)}/")
    print(f"\nArchivos que se generaran:")
    print(f"  replication_config.json")
    print(f"  tracking/test_sample_losses.json")
    print(f"  tracking/training_tracking_detailed.json")
    print(f"  analysis/exclusion_list.csv (solo train/val)")
    print(f"  analysis/test_problematic_list.csv (solo test)")
    print(f"{'='*70}")


# CLI
def main():
    parser = argparse.ArgumentParser(
        description='Re-entrena un experimento con tracking de samples y analiza clips problematicos.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--exp', type=str, required=True,
                        help='Nombre del experimento (e.g., exp_22)')
    parser.add_argument('--top', type=int, default=20,
                        help='Top N clips problematicos por epoca (default: 20)')
    parser.add_argument('--loss-pct', type=int, default=90,
                        help='Percentil de loss para la exclusion list (default: 90)')
    parser.add_argument('--min-appearances', type=int, default=2,
                        help='Apariciones minimas para exclusion list (default: 2)')
    parser.add_argument('--gpu', action='store_true',
                        help='Usar GPU si esta disponible')
    parser.add_argument('--skip-analysis', action='store_true',
                        help='Solo re-entrenar, sin analizar clips')
    parser.add_argument('--only-analysis', action='store_true',
                        help='Solo analizar (requiere que el tracking ya exista)')
    parser.add_argument('--yes', action='store_true',
                        help='No pedir confirmacion')

    args = parser.parse_args()

    # Device
    if args.gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Usando GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print(f"Usando CPU")

    # Verificar experimento
    exp_dir = PROJECT_ROOT / 'experiments' / args.exp
    if not exp_dir.exists():
        print(f"ERROR: No existe el experimento '{args.exp}'")
        experiments_dir = PROJECT_ROOT / 'experiments'
        if experiments_dir.exists():
            available = sorted(d.name for d in experiments_dir.iterdir() if d.is_dir())
            print(f"\nExperimentos disponibles:")
            for e in available[:15]:
                has_manifest = (exp_dir.parent / e / 'results' / 'tables' / 'manifest_features.csv').exists()
                mark = "OK" if has_manifest else "sin manifest"
                print(f"  {e}  [{mark}]")
            if len(available) > 15:
                print(f"  ... y {len(available) - 15} mas")
        sys.exit(1)

    # Directorio de output del experimento
    output_dir = ANALYZE_DIR / 'output' / args.exp
    tracking_dir = output_dir / 'tracking'
    analysis_dir = output_dir / 'analysis'

    # Modo: solo analisis
    if args.only_analysis:
        print(f"\nModo: solo analisis sobre tracking existente")
        if not (tracking_dir / 'test_sample_losses.json').exists():
            print(f"ERROR: No existe test_sample_losses.json o training_tracking_detailed.json en {tracking_dir}")
            print(f"Ejecuta primero sin --only-analysis para re-entrenar con tracking.")
            sys.exit(1)
        # Cargar configuracion del experimento para obtener la ruta del manifest
        try:
            _, _, _, _, manifest_file, _ = load_experiment_config(exp_dir)
        except (FileNotFoundError, Exception) as e:
            print(f"ERROR al cargar configuracion para el analisis: {e}")
            sys.exit(1)
            
        exclusion_df = detect_problematic_clips(
            tracking_dir=tracking_dir,
            analysis_dir=analysis_dir,
            manifest_file=manifest_file,
            loss_pct=args.loss_pct,
        )
        if not exclusion_df.empty:
            print(f"\nExclusion list guardada en: {(analysis_dir / 'exclusion_list.csv').relative_to(ANALYZE_DIR)}")
        return

    # Cargar configuracion del experimento
    try:
        seed, num_epochs, patience, model_config, manifest_file, replication_info = \
            load_experiment_config(exp_dir)
    except (FileNotFoundError, Exception) as e:
        print(f"ERROR al cargar configuracion: {e}")
        sys.exit(1)

    _print_config(args.exp, seed, num_epochs, patience, model_config, manifest_file, output_dir)

    # Confirmacion
    if not args.yes:
        resp = input('\n¿Continuar? (s/n): ')
        if resp.strip().lower() != 's':
            print('Cancelado.')
            return

    # Guardar replication_config.json
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'replication_config.json', 'w', encoding='utf-8') as f:
        json.dump(replication_info, f, indent=2, default=str)
    print(f"\nreplication_config.json guardado en {output_dir.relative_to(PROJECT_ROOT)}/")

    # Re-entrenamiento con tracking
    print(f"\n{'='*70}")
    print("INICIANDO RE-ENTRENAMIENTO CON TRACKING")
    print(f"{'='*70}\n")

    try:
        history, best_val_loss, best_val_auc, best_epoch, test_metrics, test_preds, _ = \
            train_model_with_tracking(
                config=model_config,
                features_manifest=str(manifest_file),
                output_dir=output_dir,
                device=device,
                num_epochs=num_epochs,
                patience=patience,
                seed=seed,
                project_root=PROJECT_ROOT,
            )
    except Exception as e:
        print(f"\nERROR durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\n{'='*70}")
    print("ENTRENAMIENTO COMPLETADO")
    print(f"{'='*70}")
    print(f"  Mejor epoca : {best_epoch}")
    print(f"  Val loss    : {best_val_loss:.4f}")
    print(f"  Val AUC     : {best_val_auc:.4f}")
    print(f"  Test metrics: {test_metrics}")

    # Analisis automatico de clips
    if not args.skip_analysis:
        print(f"\n{'='*70}")
        print("INICIANDO ANALISIS DE CLIPS PROBLEMATICOS")
        print(f"{'='*70}\n")

        exclusion_df = detect_problematic_clips(
            tracking_dir=tracking_dir,
            analysis_dir=analysis_dir,
            manifest_file=manifest_file,
            loss_pct=args.loss_pct,
        )

    # Limpieza de archivos innecesarios
    print(f"\nLimpiando archivos temporales no requeridos...")
    models_dir = output_dir / 'models'
    
    if models_dir.exists():
        shutil.rmtree(models_dir)
        
    for f in tracking_dir.iterdir():
        if f.is_file() and f.name not in ['test_sample_losses.json', 'training_tracking_detailed.json']:
            f.unlink()
    
    if not args.skip_analysis:
        print(f"\n{'='*70}")
        print("FLUJO COMPLETADO")
        print(f"{'='*70}")
        print(f"\nTodo guardado en: {output_dir.relative_to(PROJECT_ROOT)}/")
        print(f"\nSiguiente paso:")
        if not exclusion_df.empty:
            print(f"  Analizar los clips problemáticos en exclusion_list.csv.")
            print(f"  Analizar los clips de prueba en test_problematic_list.csv.")
        else:
            print(f"  No se genero exclusion list (no se detectaron clips problemáticos en train/val).")

    else:
        print(f"\nAnalisis omitido (--skip-analysis).")
        print(f"Para analizar: python retrain_with_tracking.py --exp {args.exp} --only-analysis")


if __name__ == '__main__':
    main()
