import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from pathlib import Path
import time
import random

from src.models.lstm_classifier import LSTMClassifier
from src.data.datasets import build_dataloaders

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) #por si se usa multi-GPU
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate(model, loader, criterion, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

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
    
    return avg_loss, acc, prec, rec, f1, auc, all_labels, all_preds, all_probs


def evaluate_with_attention(model, loader, criterion, device):
    """
    Evalúa el modelo y captura los pesos de atención para explicabilidad.
    
    Returns:
        Mismas métricas que evaluate() + attention_weights (lista de arrays)
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    all_attention_weights = []
    total_loss = 0.0

    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = outputs.cpu().numpy().reshape(-1)
            preds = (probs >= 0.5).astype(int)
            all_labels.extend(labels.cpu().numpy().reshape(-1))
            all_preds.extend(preds)
            all_probs.extend(probs)
            
            # Capturar pesos de atención si están disponibles
            attention_weights = model.get_attention_weights()
            if attention_weights is not None:
                # Convertir a numpy y guardar para cada muestra del batch
                all_attention_weights.extend(attention_weights.cpu().numpy())

    avg_loss = total_loss / max(1, len(loader))

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0.0)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    
    return avg_loss, acc, prec, rec, f1, auc, all_labels, all_preds, all_probs, all_attention_weights

def train_model(config, features_manifest=None, save_path=None, device=None, num_epochs=60, patience=10, seed=42):
    """
    Entrena el modelo LSTM con una configuración dada y guarda el mejor modelo.
    Args:
        config (dict): Configuración del modelo y entrenamiento.
        features_manifest (str): Ruta al manifest CSV de features.
        save_path (str): Ruta para guardar el mejor modelo.
        device (torch.device): Dispositivo para entrenamiento (CPU o GPU).
        num_epochs (int): Número máximo de épocas para entrenar.
        patience (int): Paciencia para early stopping.
        seed (int): Semilla para reproducibilidad.
    """
    set_seed(seed)

    print(f"Entrenando con Config: {config}\n")
    start_time = time.time()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Usar manifest si está disponible, sino usar features_root (legacy)
    if features_manifest:
        train_loader, val_loader, test_loader = build_dataloaders(features_manifest, batch_size=config["batch_size"], seed=seed)
    else:
        raise ValueError("Debe especificar features_manifest")

    model = LSTMClassifier(
        input_size=config.get("input_size", 512),
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        bidirectional=config["bidirectional"],
        dropout_fc=config["dropout_fc"]
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    best_val_auc_at_best_loss = -1.0
    best_val_loss = float('inf')
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

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / max(1, len(train_loader))

        val_loss, acc, prec, rec, f1, auc, _, _, _ = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(auc)
        history['val_accuracy'].append(acc)
        history['val_precision'].append(prec)
        history['val_recall'].append(rec)
        history['val_f1_score'].append(f1)

        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {auc:.4f}")

        is_better = val_loss < best_val_loss

        if is_better:
            best_val_loss = val_loss
            best_val_auc_at_best_loss = auc
            torch.save(model.state_dict(), save_path)
            epochs_no_improve = 0
            best_epoch = epoch + 1
            print(f" Mejor modelo guardado en epoch {best_epoch} con Val AUC: {best_val_auc_at_best_loss:.4f}. Guardado en {save_path.resolve()}")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"Early stopping activado en la época {epoch+1}. No hubo mejora en {patience} épocas.")
            break
    
    elapsed_time = time.time() - start_time
    print(f"\nEntrenamiento completado en {elapsed_time:.2f} segundos.")

    # evaluar el mejor modelo en test CON captura de pesos de atención
    print(f"\nCargando el mejor modelo para evaluación en test.")
    model.load_state_dict(torch.load(save_path))
    
    # Usar evaluate_with_attention para capturar explicabilidad
    test_loss, t_acc, t_prec, t_rec, t_f1, t_auc, t_labels, t_preds, t_probs, t_attention = evaluate_with_attention(
        model, test_loader, criterion, device
    )
    
    print(f"\nMétricas finales en test")
    print(f"Loss: {test_loss:.4f} | Acc: {t_acc:.4f} | Prec: {t_prec:.4f} | Rec: {t_rec:.4f} | F1: {t_f1:.4f} | AUC: {t_auc:.4f}")

    test_metrics = {'loss': test_loss, 'accuracy': t_acc, 'precision': t_prec,
                    'recall': t_rec, 'f1_score': t_f1, 'auc': t_auc}
    
    # Convertir arrays de NumPy a listas de Python nativas para serialización JSON
    test_predictions = {
        'labels': [int(label) for label in t_labels],
        'preds': [int(pred) for pred in t_preds],
        'probs': [float(prob) for prob in t_probs]
    }

    return history, best_val_loss, best_val_auc_at_best_loss, best_epoch, test_metrics, test_predictions