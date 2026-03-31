
# Modelo Híbrido de Detección de Robos (XAI) - exp_69

## Arquitectura del Modelo

Este es un modelo **híbrido end-to-end** que combina:

1. **Feature Extractor: R3D-18 (3D ResNet-18)**
   - Pre-entrenado en Kinetics-400
   - Sin capa fully connected (extrae features de 512 dimensiones)
   - Input: Clips de video (16 frames de 112x112)
   - Output: Vector de features por clip

2. **Classifier: LSTM Bidireccional con Atención (XAI)**
   - Input Size: 512
   - Hidden Size: 192
   - Num Layers: 2
   - Bidirectional: True
   - Dropout: 0.26448851490160175
   - Mecanismo de Atención Temporal: Calcula la importancia de cada clip en la decisión.

**Pipeline completo:** Video Puro -> Slicing de Clips Automático -> R3D-18 (Features) -> LSTM -> Predicción Final y Respuesta Temporal (XAI)

## Inteligencia Artificial Explicable (XAI)

El modelo cuenta con explicabilidad provista por una capa de **Atención Temporal** a la salida del LSTM. 
Esta capa infiere un peso que refleja la contribución e importancia de ese clip temporal respecto a toda la secuencia evaluada. 
Esto permite que el modelo procese el video competo desde crudo y entregue una predicción y **el intervalo exacto de segundos que representó la mayor sospecha de robo.**

## Métricas de Rendimiento (Test Set)

- **Accuracy:** 0.8438
- **Precision:** 0.7778
- **Recall:** 0.8400
- **F1-Score:** 0.8077
- **AUC-ROC:** 0.8821

## Uso End-to-End

```python
from inference import RobberyDetector

# Inicializar detector
# Asegurarse de estar en el directorio de `results/inference` o pasar la ruta en `model_dir`
detector = RobberyDetector(model_dir='.')

# Hacer inferencia inyectando directamente todo el video MP4
result = detector.predict_from_video('video_captura_01.mp4')

print(f"Predicción Final: {result['class']}")
print(f"Probabilidad de ser robo: {result['probability']*100:.2f}%")

if result['is_robbery']:
    print(f"¡Alerta de Robo detectada!")

# Mensaje XAI con el intervalo de tiempo exacto que detonó la decisión
print(f"Explicabilidad: {result['xai_message']}")
```

## Estructura de Archivos

```
results/
└── inference/
    ├── lstm_model.pth             # Pesos del clasificador LSTM
    ├── model_config.json          # Configuración completa del modelo híbrido y procesamiento de videos
    ├── inference.py               # Script de inferencia end-to-end
    ├── hybrid_model_complete.pth  # Modelo completo empaquetado (opcional)
    ├── src/
    │   └── lstm_classifier.py     # Definición de la arquitectura con Mecanismo de Atención
    └── README.md                  # Esta documentación
```
