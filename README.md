# Proyecto de Detección de Robos con Deep Learning

Proyecto de clasificación de videos para detección de robos utilizando modelos de deep learning (ResNet3D-18 para extracción de características espaciotemporales + LSTM para análisis secuencial).

## Objetivo

Clasificar videos continuos en dos categorías:
- **Normal**: Comportamiento normal en ambientes cotidianos o rutinas de tiendas.
- **Robbery**: Intento o ejecución de un robo/acciones delictivas.

## Estructura de Directorios

```text
│
├── config/                             # Configuración del proyecto
│   ├── params.yml                      # Parámetros base modificables para ejecutar un experimento
│   └── requirements.txt                # Dependencias
├── data/                               # Set de datos centralizado
│   ├── raw/                            # Videos originales (.mp4). (Solo muestras en el repositorio)
│   ├── interim/                        # Clips de fragmentos de tiempo, balanceados y listos (Excluídos del repositorio).
│   └── processed/                      # Tensores de características (.npy) obtenidos por R3D (Excluídos del repositorio).
├── experiments/                        # Experimentos realizados
│   ├── exp_01/
│   ├── exp_02/
│   └── ...
├── notebooks/                          # Cuadernos interactivos para el ciclo de vida del modelo
│   ├── 0_eda.ipynb                     # Análisis exploratorio de datos
│   ├── 1_data_preparation.ipynb        # Pipeline de datos para el preprocesamiento de un experimento
│   ├── 2_model_training.ipynb          # Pipeline de entrenamiento del modelo de un experimento
│   ├── 3_results_analysispynb          # Análisis de resultados de un experimento
│   └── 4_compare_experiments.ipynb     # Comparación de resultados de múltiples experimentos
├── results/                            # Directorio de recolección de reportes
│   ├── eda/                            # Reportes descriptivos del dataset en HTML interactivo
│   └── experiments/                    # Reporte HTML con ranking comparativo de los modelos
├── src/                                # Código fuente modularizado del sistema
│   ├── data/                           # Procesamiento de datos
│   ├── features/                       # Extracción de características
│   ├── models/                         # Definiciones de modelos
│   ├── training/                       # Lógica de entrenamiento
│   └── visualization/                  # Visualizaciones
└── tracking/                           # Entorno de auditoría avanzada y depuración de clips
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

## Exploración y Flujo de Trabajo (Notebooks)

El flujo iterativo clásico de trabajo se ha modularizado de forma secuencial a través de los directorios de `notebooks/`:

### `0_eda.ipynb` (Análisis Exploratorio)
Genera gráficos sobre variables descriptivas y estadísticas de distribución (resoluciones, FPS, cantidad y duración de los videos) originados en `data/raw/`. Los compila como reportes interactivos navegables y los guarda automáticamente en formato visual dentro del directorio `results/eda/`.

### `1_data_preparation.ipynb` (Preparación y Muestreo)
Realiza el pipeline primario leyendo un experimento configurado en `config/params.yml`. Escanea los videos de `data/raw/` y los divide temporalmente en pequeños fragmentos o *clips*. Estos son transferidos al área centralizada de `data/interim/` (estructurados por variaciones de longitud, solapamiento o balanceo). A su vez, genera archivos *manifest* CSV para trazar de manera persistente los sets de `train/val/test`.

### `2_model_training.ipynb` (Extracción y Entrenamiento)
Escanea los fragmentos visuales (*clips*) pasándolos por la base pre-entrenada espacial **ResNet3D** (r3d_18). Los resultados tabulados los guarda temporalmente en `data/processed/` (como tensores `.npy`). Luego levanta un motor **Optuna**, logrando optimizar la mejor arquitectura LSTM recurrente disponible. Termina calculando inferencias predictivas y guardando modelos y predicciones de validación/prueba en el directorio de su propio experimento (`experiments/exp_XX/results/`).

### `3_results_analysis.ipynb` (Conclusiones de un Experimento)
Visualiza internamente el desempeño aislado del experimento ejecutado, diagramando el historial de `Loss/Accuracy`, mostrando la Matriz de Confusión, reporte de clasificación general sobre la capacidad predictiva y curva ROC.

### `4_compare_experiments.ipynb` (Comparativa General)
Algoritmo enlazador global que recorre internamente a *todos los experimentos* del directorio `experiments`. Analiza profundamente si existe "Overfitting" o degradación en la convergencia al comparar las curvas de validación con entrenamiento iterativamente. 
Su *output* más valioso es que fabrica y guarda el histórico visual comparativo en **`results/experiments/compare_experiments_report.html`**, donde resalta objetivamente por qué y quién es el modelo ganador tras cruzar el factor de sensibilidad y la generalización.

## Rutinas de Tracker de Pérdidas Ocultas (`tracking/`)

Además del ciclo convencional, para combatir estancamientos de rendimiento (ruido en los datos), este proyecto usa un submódulo dedicado de nombre **`tracking/`**. 

Éste replica la sesión de un experimento en particular pero vigila silenciosamente con un Trainer modificado los desajustes por cada _Epoch_ con el fin de rastrear individualmente y capturar anomalías por muestra particular. Estas exclusiones o fallos terminan reflejadas dentro de listados CSV estáticos de la misma carpeta que sugerirán qué elementos excluir en nuevas iteraciones para potenciar rápidamente la red neuronal.

**(Consultar `tracking/README.md` para comandos e información detallada de la lógica de re-entrenamiento del área).**

## Dinámica de Estructura de un Experimento

Cada iteración de modelo configurada se aloja herméticamente en la ruta de su experimento de la siguiente manera:
```text
experiments/
└── exp_xx/
    ├── config_run.yml          # Topología congelada (Seed, hiperparámetros, muestreo y variables de procesamiento usadas)
    └── results/                # Resultados depositados post-entrenamiento
        ├── models/             # best_lstm_model.pth
        ├── tables/             # manifiestos (CSV), métricas finales (JSON), estadísticas Optuna, test predictions.
        └── plots/              # Snapshots visuales en PNG con reportes base (Learning Curves, Confusion Matrix).
```
*Nota: Los grandes volúmenes de datos en tensores (.npy) o recortes cortos de video (.mp4) jamás habitan el interior del experimento individual (se referencian centralmente a los pools reusables de `data/interim` y `data/processed`). Esto ahorra masivamente el consumo en disco si las estrategias comparten pipelines de datos.*

### Configuración

El archivo `config/params.yml` contiene todos los parámetros configurables para ejecutar un experimento:

- **Semilla**: Utilizada por todos los procesos que utilizan aleatoriedad para garantizar reproducibilidad del experimento
- **Experimento**: nombre y ubicación
- **Fuente**: Ubicación del dataset
- **Procesamiento de video**: longitud de clips, overlapping, etc.
- **Extracción de features**: tipo de extractor
- **Entrenamiento**: epochs, patience, etc.
- **Búsqueda de hiperparámetros**: número de trials de Optuna
- **Archivos resultantes**: Nombre

## Reglas de Control de Versión (Archivos Incluidos y Excluidos)

Por limitaciones estándares de almacenamiento en GitHub, extensos volúmenes de datasets están parametrizados en el `.gitignore`:

**Archivos Excluidos**:
- Videos pesados de origen (directorio `data/raw/` completo, exceptuando unas muestras pequeñas representativas).
- Todo el metraje de clips procesados y fragmentados residentes en `data/interim/` (`.mp4`).
- Matrices masivas extraídas de ResNet guardadas como agrupamiento binario denso dentro de las subcarpetas iterativas en `data/processed/` (`.npy`).

**Archivos Incluidos en Repositorio**:
- Código fuente y módulos `src/`.
- Notebooks Jupyter limpios formativos.
- Gráficas y reportes (HTMLs y PNGs visuales de validación) exportados hacia `results/`.
- Todos los `config_run.yml` in-corruptibles de toda la librería de todos los experimentos.
- Registros evaluativos consolidados, CSVs con la traza de los trials con Optuna, y reportes analíticos para reproducibilidad total cruzada (`experiments/*/tables/`).
- Binarios entrenados de cada uno de los experimentos `.pth`.
- Árboles esqueléticos que conservan las jerarquías complejas generadas (`.gitkeep` implementado estéticamente).

## Origen de Set de Datos Global (Videos)

Los archivos de entrenamiento reales (`.mp4`) no pueden enviarse vía la plataforma Git de manera arbitraria por su peso superior a decenas de gigabytes.

Para obtener el dataset completo empleado en el proyecto se debe:
1. Descargar el dataset completo desde [Robbery and Normal Videos for Classification](https://www.kaggle.com/datasets/ivang0nzalez/robbery-and-normal-videos-for-classification)
2. Descomprimir y colocar los directorios correspondientes dentro de `data/raw/`
3. Seguir la estructura indicada en `data/raw/README.md`


## Implementación en Aplicación Web

A partir de los resultados y el "mejor modelo" obtenido en este ciclo de vida de experimentación y MLOps, se ha desarrollado una aplicación completa y funcional (backend de inferencia + dashboard interactivo).

El código fuente interactivo y la arquitectura para despliegue de estos resultados se encuentra completamente desacoplado y disponible en el siguiente repositorio:
 **[IvanG0nzalez/RobberyDetection-App](https://github.com/IvanG0nzalez/RobberyDetection-App.git)**

## Autor
Iván Alejandro González Ortega

