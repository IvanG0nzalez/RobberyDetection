# Análisis de Clips Problemáticos

Flujo para replicar un experimento, trackear clips conflictivos y prepararlos para su revisión o exclusión.

## Estructura

```
tracking/
├── retrain_with_tracking.py              # Script principal: replica entrenamiento + tracking + análisis
├── tracked_datasets.py                   # Dataset con metadata de sample/clip                   
├── tracked_trainer.py                    # Entrenador con logging por sample             
└── output/
    └── exp_XX/
        ├── replication_config.json
        ├── tracking/                     # Archivos de traqueo de épocas
        └── analysis/                     # Listas de clips problemáticos
```

## Modo de uso

1) Replicar el entrenamiento de un experimento con tracking + análisis automático

```bash
cd analyze_clips
python retrain_with_tracking.py --exp exp_XX --gpu --yes
```

2) Analizar los archivos `.csv` resultantes

Tras ejecutar el script principal el código automáticamente generará archivos de seguimiento durante el entrenamiento y de análisis sobre los elementos que provocaron picos de pérdidas, estos se almacenan en `output/exp_xx/tracking/` y `output/exp_xx/analysis/`, para encontrar aquellos elementos que están introduciendo ruido al entrenamiento analizar el archivo llamado `exclusion_list.csv`.

3) Realizar un nuevo experimento

Con los resultados obtenidos en `exclusion_list.csv` se puede generar un nuevo experimento con la misma configuración del experimento analizado, ejecutar el notebook `1_data_preparation.ipynb`, eliminar manualmente los elementos problemáticos del archivo **manifest** y continuar con el flujo de trabajo común.



## Notas importantes

- Este flujo lee artefactos desde `experiments/exp_XX/`, pero **no** escribe resultados fuera de `tracking/output/`.
- El script `retrain_with_tracking.py` se encarga de todo el ciclo, no se necesita ejecutar otros scripts de Python para generar los reportes de anomalías.
