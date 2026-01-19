import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pandas as pd
from pathlib import Path

def plot_training_history(history, save_path=None):
    """
    Grafica las curvas de pérdida y AUC de validación.
    """
    history_df = pd.DataFrame(history)

    metrics = {
        'loss': ('train_loss', 'val_loss'),
        'auc': ('val_auc',),
        'accuracy': ('val_accuracy',),
        'precision': ('val_precision',),
        'recall': ('val_recall',),
        'f1_score': ('val_f1_score',)
    }

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Historial de Entrenamiento', fontsize=20, y=1.03)
    
    ax_list = axes.flat

    for i, (metric_name, keys) in enumerate(metrics.items()):
        ax = ax_list[i]
        
        if 'train_loss' in keys and 'train_loss' in history_df:
            ax.plot(history_df['train_loss'], label='Train Loss', color='blue')
        if 'val_loss' in keys and 'val_loss' in history_df:
            ax.plot(history_df['val_loss'], label='Validation Loss', color='orange')

        val_key = f"val_{metric_name}"
        if val_key in keys and val_key in history_df and val_key != 'val_loss':
            ax.plot(history_df[val_key], label=f'Validation {metric_name.capitalize()}', color='green')

        ax.set_title(f"Historial de {metric_name.capitalize()}", fontsize=16)
        ax.set_xlabel('Época')
        ax.set_ylabel(metric_name.capitalize())
        ax.legend()
        ax.grid(True, linestyle='--')

    plt.tight_layout()
    
    if save_path:
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Gráfico de historial guardado en: {save_path.resolve()}")
        except Exception as e:
            print(f"Error al guardar el gráfico de historial: {e}")
    
    plt.show()
    plt.close(fig)

def plot_classification_analysis(y_true, y_pred, y_prob, class_names=['Normal', 'Robbery'], save_path=None):
    """
    Grafica la matriz de confusión y la curva AUC-ROC.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle('Análisis de Clasificación', fontsize=20, y=1.03)

    try:
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax1)

        ax1.set_title('Matriz de Confusión', fontsize=16)
        ax1.set_xlabel('Etiqueta Predicha')
        ax1.set_ylabel('Etiqueta Verdadera')


        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.4f})')
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Tasa de Falsos Positivos (FPR)')
        ax2.set_ylabel('Tasa de Verdaderos Positivos (TPR)')
        ax2.set_title('Curva ROC', fontsize=16)
        ax2.legend(loc="lower right")
        ax2.grid(True, linestyle='--')

        plt.tight_layout()

        if save_path:
            try:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, bbox_inches='tight')
                print(f"Gráfico de análisis de clasificación guardado en: {save_path.resolve()}")
            except Exception as e:
                print(f"Error al guardar el gráfico de análisis de clasificación: {e}")

        plt.show()
        plt.close(fig)

    except Exception as e:
        print(f"Error al graficar el análisis de clasificación: {e}")