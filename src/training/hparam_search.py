import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import random
import numpy as np

from src.models.lstm_classifier import LSTMClassifier
from src.data.datasets import build_dataloaders
from src.training.trainer import evaluate

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) #por si se usa multi-GPU
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def objective(trial, features_manifest=None, features_root=None, device=None, num_epochs=60, patience=10, input_size=512, seed=42):
    """Función objetivo para la búsqueda de hiperparámetros con Optuna."""

    # Usar el mismo seed global para todos los trials
    set_seed(seed)

    config = {
        "input_size": input_size,
        "hidden_size": trial.suggest_categorical("hidden_size", [64, 96, 128, 192, 256]),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "bidirectional": trial.suggest_categorical("bidirectional", [True, False]),
        "lr": trial.suggest_float("lr", 5e-4, 2e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "dropout_fc": trial.suggest_float("dropout_fc", 0.2, 0.6),
        "use_attention": True,  # Siempre usar atención para explicabilidad
    }

    # Usar manifest si está disponible, sino usar features_root
    if features_manifest:
        train_loader, val_loader, _ = build_dataloaders(features_manifest, batch_size=config["batch_size"], seed=seed)
    elif features_root:
        from src.data.datasets import build_dataloaders_legacy
        train_loader, val_loader, _ = build_dataloaders_legacy(features_root, batch_size=config["batch_size"], seed=seed)
    else:
        raise ValueError("Debe especificar features_manifest o features_root")

    model = LSTMClassifier(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        bidirectional=config["bidirectional"],
        dropout_fc=config["dropout_fc"],
        use_attention=True  # Siempre usar atención
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        val_loss, acc, prec, rec, f1, auc, _, _, _ = evaluate(model, val_loader, criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        trial.report(val_loss, epoch)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    trial.set_user_attr("best_val_auc", auc)  # Guardar AUC del trial para referencia
        
    return best_val_loss