# Features Extraídas (Processed Data)

Este directorio, `data/processed/`, almacena las características (features) extraídas por la red neuronal convolucional 3D de los clips de video originados en `data/interim/`.

## Estructura del Directorio
Cada subdirectorio (ej. `features_original_len16_seg180_overlap_str8`) representa una versión de las características procesadas que se corresponden con la configuración de los recortes de clips.
A su vez, dentro de estos directorios, los archivos están categorizados en sus respectivas clases:
* `Normal/`
* `Robbery/`

## Archivos incluidos en el repositorio

Los archivos resultantes de la extracción de características son vectores numéricos (ej. tensores `*.npy`), que resultan muy pesados para ser almacenados y rastreados mediante Git, por esta razón simplemente se conserva la estructura de directorios.

Para reproducir o generar los tensores para el entrenamiento, se debe ejecutar el flujo de trabajo siguiendo los notebooks `1_data_preparation.ipynb` y `2_model_training.ipynb` con la configuración adecuada en `config/params.yml`.