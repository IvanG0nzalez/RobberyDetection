# Datos Intermedios - Clips

Este directorio contiene los clips de video generados durante el proceso de preparación de datos, así como también los aumentos de datos (oversample) y los recortes.

## Estructura

Los clips se organizan con nombres representativos que codifican los parámetros bajo los que fueron extraídos (longitud de clips, configuración de segmentación, sobrelapamiento y ratio de balanceo):

```
interim/
├── clips_original_len16_seg180_overlap_str8/
├── clips_original_len16_seg32_uniform/
├── clips_oversample_original_len16_seg180_overlap_str8_ratio10/
├── clips_recortados_len16_seg180_overlap_str8/
└── ...
```

## Contenido

Cada subdirectorio contiene pequeños clips de video en formato `.mp4` organizados por:
- **Clase**: `Normal/` y `Robbery/`
- **Video**: Cada subcarpeta contiene los extractos del video raíz.

Además, por cada proceso de generación se guarda un archivo unificado con las configuraciones:
- `generation_metadata.json`

## Archivos incluidos en el repositorio

Por limitaciones extremas de tamaño:
- **Todo el contenido de clips está excluido del repositorio** (`*.mp4`, subdirectorios extraídos, etc).
- Para los directorios normales y de recortes, se preserva únicamente la **estructura base de clases** con directorios vacíos (usando `.gitkeep` en `/Normal/` y `/Robbery/`).
- Para los directorios que aplican técnicas de `oversample`, se preserva únicamente el directorio `/Robbery/` (ya que la clase Normal no resulta aumentada).
- El archivo `generation_metadata.json` siempre está incluido para reflejar las configuraciones pasadas.

Para reproducir o generar los clips para entrenamiento, se debe ejecutar el notebook `1_data_preparation.ipynb` con la configuración adecuada en `config/params.yml`.
