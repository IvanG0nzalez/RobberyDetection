"""
Trainer con logging detallado de samples y pérdidas por batch.
Permite analizar qué clips causan problemas durante el entrenamiento.

El output (modelo, logs, métricas) se guarda en el directorio
output_dir que se pasa como parámetro (tracking/output/{exp_name}/)

NOTA: Este módulo requiere acceso a las clases del proyecto principal.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from pathlib import Path
import time
import random
import json
import sys
from collections import defaultdict

# Agregar el proyecto principal al path para importar LSTMClassifier
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lstm_classifier import LSTMClassifier
from tracked_datasets import build_tracked_dataloaders


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_tracked(model, loader, criterion, device, log_samples=False):
    """
    Evalúa el modelo con tracking opcional de samples.
    
    Returns:
        Si log_samples=False: (avg_loss, acc, prec, rec, f1, auc, labels, preds, probs)
        Si log_samples=True: (..., sample_losses) donde sample_losses es lista de dicts con info de cada sample
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    total_loss = 0.0
    sample_losses = [] if log_samples else None

    with torch.no_grad():
        for batch_data in loader:
            features, labels, sample_infos = batch_data
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            
            # Pérdida por batch
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Si estamos loggeando samples, calcular pérdida individual
            if log_samples:
                # Calcular pérdida por sample (sin reducción)
                individual_losses = nn.BCELoss(reduction='none')(outputs, labels)
                individual_losses = individual_losses.cpu().numpy().reshape(-1)
                
                # Guardar info de cada sample
                batch_probs = outputs.cpu().numpy().reshape(-1)
                batch_labels = labels.cpu().numpy().reshape(-1)
                
                for i, sample_info in enumerate(sample_infos):
                    sample_losses.append({
                        'directoryname': sample_info['directoryname'],
                        'class': sample_info['class'],
                        'featurepath': sample_info['featurepath'],
                        'index': sample_info['index'],
                        'loss': float(individual_losses[i]),
                        'prediction': float(batch_probs[i]),
                        'label': int(batch_labels[i])
                    })

            probs = outputs.cpu().numpy().reshape(-1)
            preds = (probs >= 0.5).astype(int)
            all_labels.extend(labels.cpu().numpy().reshape(-1))
            all_preds.extend(preds)
            all_probs.extend(probs)

    avg_loss = total_loss / max(1, len(loader))

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0.0)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    
    if log_samples:
        return avg_loss, acc, prec, rec, f1, auc, all_labels, all_preds, all_probs, sample_losses
    else:
        return avg_loss, acc, prec, rec, f1, auc, all_labels, all_preds, all_probs


def train_model_with_tracking(config, features_manifest, output_dir, device=None,
                              num_epochs=60, patience=10, seed=42, project_root=None):
    """
    Entrena el modelo con logging detallado de samples y pérdidas.
    Todo el output (modelo + logs) se guarda en output_dir:
        output_dir/models/best_tracked_model.pth
        output_dir/tracking/training_tracking_detailed.json
        output_dir/tracking/training_history.json
        output_dir/tracking/test_sample_losses.json
        output_dir/tracking/final_metrics.json

    Args:
        config (dict): Configuración del modelo y entrenamiento
        features_manifest (str): Ruta al manifest CSV de features (lectura)
        output_dir (str or Path): Directorio raíz de salida del experimento
        device: Dispositivo para entrenamiento
        num_epochs (int): Número máximo de épocas
        patience (int): Paciencia para early stopping
        seed (int): Semilla para reproducibilidad
        project_root (Path): Raíz del proyecto principal

    Returns:
        (history, best_val_loss, best_val_auc, best_epoch, test_metrics, test_predictions, tracking_logs)
    """
    set_seed(seed)

    print(f"Entrenando con Tracking. Config: {config}\n")
    start_time = time.time()

    output_dir = Path(output_dir)
    save_path = output_dir / 'models' / 'best_tracked_model.pth'
    tracking_dir = output_dir / 'tracking'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tracking_dir.mkdir(parents=True, exist_ok=True)

    # Build tracked dataloaders
    train_loader, val_loader, test_loader = build_tracked_dataloaders(
        features_manifest, 
        batch_size=config["batch_size"], 
        seed=seed,
        project_root=project_root
    )

    model = LSTMClassifier(
        input_size=config.get("input_size", 512),
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        bidirectional=config["bidirectional"],
        dropout_fc=config["dropout_fc"]
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    best_val_loss = float('inf')
    best_val_auc_at_best_loss = -1.0
    epochs_no_improve = 0
    
    history = {
        'train_loss': [], 
        'val_loss': [], 
        'val_auc': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1_score': []
    }
    
    best_epoch = -1
    
    # Estructura para guardar tracking detallado
    tracking_logs = {
        'epoch_batch_losses': [],       # Compatibilidad: batches de train
        'epoch_sample_stats': [],       # Compatibilidad: stats de train
        'epoch_sample_stats_train': [], # Nuevo: stats agregadas por sample (train)
        'epoch_sample_stats_val': []    # Nuevo: stats agregadas por sample (val)
    }

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        epoch_batch_info = []
        epoch_sample_losses = defaultdict(list)  # Para agregar pérdidas por sample

        batch_num = 0
        for batch_data in train_loader:
            features, labels, sample_infos = batch_data
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(features)
            
            # Pérdida para backprop
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            total_train_loss += batch_loss
            
            # Calcular pérdidas individuales (sin backprop)
            with torch.no_grad():
                individual_losses = nn.BCELoss(reduction='none')(outputs, labels)
                individual_losses = individual_losses.cpu().numpy().reshape(-1)
                
                batch_probs = outputs.cpu().numpy().reshape(-1)
                batch_labels = labels.cpu().numpy().reshape(-1)
            
            # Registrar info del batch
            batch_samples = []
            for i, sample_info in enumerate(sample_infos):
                sample_log = {
                    'directoryname': sample_info['directoryname'],
                    'class': sample_info['class'],
                    'featurepath': sample_info['featurepath'],
                    'loss': float(individual_losses[i]),
                    'prediction': float(batch_probs[i]),
                    'label': int(batch_labels[i])
                }
                batch_samples.append(sample_log)
                
                # Agregar a estadísticas por sample
                sample_key = sample_info['directoryname']
                epoch_sample_losses[sample_key].append(float(individual_losses[i]))
            
            epoch_batch_info.append({
                'epoch': epoch + 1,
                'batch': batch_num,
                'batch_loss': float(batch_loss),
                'samples': batch_samples
            })
            
            batch_num += 1

        avg_train_loss = total_train_loss / max(1, len(train_loader))

        # Evaluación en validación con tracking por sample
        val_loss, acc, prec, rec, f1, auc_val, _, _, _, val_sample_losses = evaluate_tracked(
            model, val_loader, criterion, device, log_samples=True
        )

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(auc_val)
        history['val_accuracy'].append(acc)
        history['val_precision'].append(prec)
        history['val_recall'].append(rec)
        history['val_f1_score'].append(f1)

        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {auc_val:.4f}")

        # Guardar batch info de esta época
        tracking_logs['epoch_batch_losses'].extend(epoch_batch_info)
        
        # Calcular estadísticas agregadas por sample en esta época
        train_sample_stats = []
        for sample_name, losses in epoch_sample_losses.items():
            train_sample_stats.append({
                'epoch': epoch + 1,
                'split': 'train',
                'sample': sample_name,
                'mean_loss': float(np.mean(losses)),
                'max_loss': float(np.max(losses)),
                'min_loss': float(np.min(losses)),
                'std_loss': float(np.std(losses)),
                'count': len(losses)
            })
        tracking_logs['epoch_sample_stats'].append(train_sample_stats)
        tracking_logs['epoch_sample_stats_train'].append(train_sample_stats)

        # Calcular estadísticas agregadas por sample de validación en esta época
        val_losses_by_sample = defaultdict(list)
        for sample_log in val_sample_losses:
            val_losses_by_sample[sample_log['directoryname']].append(sample_log['loss'])

        val_sample_stats = []
        for sample_name, losses in val_losses_by_sample.items():
            val_sample_stats.append({
                'epoch': epoch + 1,
                'split': 'val',
                'sample': sample_name,
                'mean_loss': float(np.mean(losses)),
                'max_loss': float(np.max(losses)),
                'min_loss': float(np.min(losses)),
                'std_loss': float(np.std(losses)),
                'count': len(losses)
            })
        tracking_logs['epoch_sample_stats_val'].append(val_sample_stats)

        # Early stopping y guardado del mejor modelo
        is_better = val_loss < best_val_loss

        if is_better:
            best_val_loss = val_loss
            best_val_auc_at_best_loss = auc_val
            torch.save(model.state_dict(), save_path)
            epochs_no_improve = 0
            best_epoch = epoch + 1
            print(f" ✓ Mejor modelo guardado en epoch {best_epoch} con Val AUC: {best_val_auc_at_best_loss:.4f}")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"Early stopping activado en la época {epoch+1}.")
            break
    
    elapsed_time = time.time() - start_time
    print(f"\nEntrenamiento completado en {elapsed_time:.2f} segundos.")

    # Guardar logs de tracking
    tracking_file = tracking_dir / 'training_tracking_detailed.json'
    with open(tracking_file, 'w', encoding='utf-8') as f:
        json.dump(tracking_logs, f, indent=2)
    print(f"Logs de tracking guardados en: {tracking_file}")

    # Evaluar el mejor modelo en test
    print(f"\nCargando el mejor modelo para evaluación en test.")
    model.load_state_dict(torch.load(save_path))
    
    test_loss, t_acc, t_prec, t_rec, t_f1, t_auc, t_labels, t_preds, t_probs, test_sample_losses = \
        evaluate_tracked(model, test_loader, criterion, device, log_samples=True)
    
    print(f"\nMétricas finales en test:")
    print(f"Loss: {test_loss:.4f} | Acc: {t_acc:.4f} | Prec: {t_prec:.4f} | Rec: {t_rec:.4f} | F1: {t_f1:.4f} | AUC: {t_auc:.4f}")

    test_metrics = {
        'loss': test_loss, 
        'accuracy': t_acc, 
        'precision': t_prec,
        'recall': t_rec, 
        'f1_score': t_f1, 
        'auc': t_auc
    }
    
    test_predictions = {
        'labels': [int(label) for label in t_labels],
        'preds': [int(pred) for pred in t_preds],
        'probs': [float(prob) for prob in t_probs]
    }
    
    # Guardar historial de entrenamiento
    history_file = tracking_dir / 'training_history.json'
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    print(f"Historial de entrenamiento guardado en: {history_file}")

    # Guardar métricas finales
    final_metrics_file = tracking_dir / 'final_metrics.json'
    with open(final_metrics_file, 'w', encoding='utf-8') as f:
        json.dump({
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'best_val_auc': best_val_auc_at_best_loss,
            'test_metrics': test_metrics,
            'config': config
        }, f, indent=2)
    print(f"Métricas finales guardadas en: {final_metrics_file}")

    # Guardar losses de test por sample
    test_tracking_file = tracking_dir / 'test_sample_losses.json'
    with open(test_tracking_file, 'w', encoding='utf-8') as f:
        json.dump(test_sample_losses, f, indent=2)
    print(f"Logs de test por sample guardados en: {test_tracking_file}")

    return history, best_val_loss, best_val_auc_at_best_loss, best_epoch, test_metrics, test_predictions, tracking_logs
