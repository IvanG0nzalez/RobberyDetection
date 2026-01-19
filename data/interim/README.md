# Datos Intermedios - Clips

Este directorio contiene los clips de video generados durante el proceso de preparación de datos.

## Estructura

Los clips se organizan con nombres representativos del dataset del que salieron y con una enumeración incremental e incluyendo el método de balanceo aplicado (undersample u oversample):

```
interim/
├── clips_original/
├── clips_original_02/
├── clips_original_03/
├── ...
├── clips_original_XX_oversample/
└── clips_original_XX_undersample/
```

## Contenido

Cada subdirectorio contiene pequeños clips de video en formato `.mp4` organizados por:
- **Clase**: Normal/Robbery
- **Video**: Cada video original genera múltiples clips

## Archivos ignorados

Por limitaciones de tamaño, **todos los clips (.mp4) están excluidos del repositorio**.

Para generar los clips, se debe ejecutar el notebook `1_data_preparation.ipynb` con la configuración deseada en `config/params.yml`.
