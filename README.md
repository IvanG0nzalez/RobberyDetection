# Proyecto de Detección de Robos con Deep Learning

Proyecto de clasificación de videos para detección de robos utilizando modelos de deep learning (R3D-18 + LSTM).

## Objetivo

Clasificar videos en dos categorías:
- **Normal**: Comportamiento normal
- **Robbery**: Intento o ejecución de robo

## Estructura del Proyecto

```
Dataset/
├── config/                    # Configuración del proyecto
│   ├── params.yml            # Parámetros configurables para ejecutar un experimento
│   └── requirements.txt      # Dependencias
├── data/                     # Datasets del proyecto
│   ├── raw/                  # Videos originales (solo muestras en el repositorio)
│   └── interim/              # Clips procesados (excluidos del repositorio)
├── experiments/              # Experimentos realizados
│   ├── exp_01/              # Experimento 1
│   ├── exp_02/              # Experimento 2
│   └── ...                  # Más experimentos
├── notebooks/               # Jupyter notebooks
│   ├── 0_eda.ipynb         # Análisis exploratorio
│   ├── 1_data_preparation.ipynb    # Pipeline de datos para el preprocesamiento de un experimento
│   ├── 2_model_training.ipynb      # Pipeline de entrenamiento de un modelo
│   ├── 3_results_analysis.ipynb    # Análisis de resultados de un experimento
│   └── 4_compare_experiments.ipynb # Comparación de resultados de múltiples experimentos
├── src/                     # Código fuente
│   ├── data/               # Procesamiento de datos
│   ├── features/           # Extracción de features
│   ├── models/             # Definiciones de modelos
│   ├── training/           # Lógica de entrenamiento
│   └── visualization/      # Visualizaciones
└── results/                # Resultados finales

```

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/IvanG0nzalez/RobberyDetection.git
cd RobberyDetection
```

### 2. Crear entorno virtual

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r config/requirements.txt
```

### 4. Obtener el dataset completo

**Nota importante**: Por limitaciones de tamaño de GitHub, este repositorio solo incluye **videos de muestra**. 

El repositorio incluye:
- 1 video Normal: `data/raw/dataset_videos_original/Normal/Normal_Videos001_x264.mp4`
- 1 video Normal: `data/raw/dataset_videos_recortados/Normal/Normal_Videos001_x264.mp4`
- 1 video Robbery: `data/raw/dataset_videos_original/Robbery/Robbery001_x264.mp4`
- 1 video Robbery: `data/raw/dataset_videos_recortados/Robbery/Robbery001_x264.mp4`

Para obtener el dataset completo:
1. Descargar los videos desde [fuente del dataset - agregar enlace]
2. Colocar los videos en los directorios correspondientes
3. Seguir la estructura indicada en `data/raw/README.md`

## Pipeline de Trabajo

### 1. Preparación de Datos

```bash
# Editar configuración en config/params.yml
# Ejecutar notebook
jupyter notebook notebooks/1_data_preparation.ipynb
```

Pasos realizados:
- Segmentación de videos en clips
- Balanceo de clases (opcional)
- División en train/val/test

### 2. Entrenamiento del Modelo

```bash
jupyter notebook notebooks/2_model_training.ipynb
```

Pasos realizados:
- Extracción de features con R3D-18
- Búsqueda de hiperparámetros con Optuna
- Entrenamiento de LSTM
- Evaluación en conjunto de test

### 3. Análisis de Resultados

```bash
jupyter notebook notebooks/3_results_analysis.ipynb
```

Pasos realizados:
- Visualización de historial de métricas durante entrenamiento
- Visualización de matriz de confusión y curva AUC-ROC
- 5 mejores intentos obtenidos por Optuna

### 4. Comparación de Experimentos

```bash
jupyter notebook notebooks/4_compare_experiments.ipynb
```

Pasos realizados:
- Visualización individual de todos los experimentos
- Tabla comparativa del mejor al peor experimento (mejor f1_score como métrica de decisión)

## Configuración

El archivo `config/params.yml` contiene todos los parámetros configurables para ejecutar un experimento:

- **Semilla**: Utilizada por todos los procesos que utilizan aleatoriedad para garantizar reproducibilidad del experimento
- **Experimento**: nombre y ubicación
- **Fuente**: Ubicación del dataset y clips a utilizar
- **Procesamiento de video**: longitud de clips, overlapping, etc.
- **Extracción de features**: tipo de extractor
- **Entrenamiento**: epochs, patience, etc.
- **Búsqueda de hiperparámetros**: número de trials de Optuna
- **Archivos resultantes**: Nombre

## Archivos Excluidos del Repositorio

Por limitaciones de tamaño de GitHub, los siguientes archivos están excluidos (ver `.gitignore`):

**Excluidos**:
- Videos originales (excepto muestras)
- Clips procesados (.mp4)
- Archivos de features (.npy)

**Incluidos**:
- Código fuente completo
- Notebooks
- Configuraciones
- Métricas y resultados (CSV, JSON)
- Muestras de datos (1 video por clase)
- Modelos LSTM por cada experimento

## Experimentos

Cada experimento se organiza en su propio directorio `experiments/exp_XX/` con:

- `config_run.yml`: Configuración específica del experimento
- `processed_data/`: Datos procesados (excluidos del repositorio)
- `results/`: Resultados y modelos entrenados

### Estructura

```
exp_xx/
├──processed_data/          # Datos procesados específicos del experimento (Excluído del repositorio)
│   ├── clips_splitted/    # Clips divididos en train/val/test
│   │   ├── train/
│   │   │   ├── Normal/
│   │   │   └── Robbery/
│   │   ├── val/
│   │   │   ├── Normal/
│   │   │   └── Robbery/
│   │   └── test/
│   │       ├── Normal/
│   │       └── Robbery/
│   └── features/          # Features extraídas (.npy)
│       ├── train/
│       ├── val/
│       └── test/
└──results/                 # Resultados del experimento
    ├── models/                    # Modelos entrenados (.pth)
    ├── tables/                    # Métricas y resultados (incluidos en el repositorio)
    │   ├── optuna_lstm_trials.csv
    │   ├── manifest_clips.csv
    │   ├── manifest_splitted.csv
    │   ├── lstm_final_metrics.json
    │   ├── lstm_training_history.json
    │   └── lstm_test_predictions.json
    └── plots/                     # Gráficos de resultados
        ├── 1_training_history.png
        ├── 2_classification_analysis.png
        └── 3_top5_optuna_trials.csv
```

Para regenerar los datos procesados:
1. Asegurarse de tener los videos correctos descritos en `config_run.yml` en `data/raw/`
2. Revisar la configuración en `config_run.yml` del experimento y copiarla a params.yml
3. Ejecutar el notebook `1_data_preparation.ipynb`
4. Ejecutar el notebook `2_model_training.ipynb` para extraer features
5. Los datos procesados se guardarán automáticamente en `processed_data/` y los resultados `results/`

## Autor

Iván Alejandro González Ortega

