# Dataset - Videos Originales y Recortados

Estos directorios contienen las clases `Normal` y `Robbery` con los videos originales del dataset y una versión con los videos de la clase Robbery cuyas escenas fueron recortadas manualmente para conservar únicamente los frames del robo con un margen de 1-2 segundos de antes y después del robo.

## Estructura

```
raw/
├──dataset_videos_original/
│   ├── Normal/          # Videos de comportamiento normal (193 videos)
│   │   └── Normal_Videos001_x264.mp4  # (ejemplo incluido en el repositorio)
│   └── Robbery/         # Videos de robos (147 videos)
│       └── Robbery001_x264.mp4        # (ejemplo incluido en el repositorio)
└──dataset_videos_recortados/
    ├── Normal/
    │   └── Normal_Videos001_x264.mp4
    └── Robbery/         
        └── Robbery001_x264.mp4
```

## Archivos incluidos en el repositorio

Por limitaciones de tamaño, este repositorio solo incluye **1 video de muestra** de cada clase en ambas versiones del dataset:
- `Normal/Normal_Videos001_x264.mp4`
- `Robbery/Robbery001_x264.mp4`

## Dataset completo

Para obtener el dataset completo:
1. Descargar los videos correspondientes desde [fuente del dataset]
2. Colocarlos en los directorios correspondientes
3. Asegurar de que los nombres de archivo sigan el patrón: `Normal_Videos*_x264.mp4` y `Robbery*_x264.mp4`

## Formato de videos

- **Formato**: MP4 (H.264)
- **Resolución**: Fija (320x240)
- **FPS**: Fija (30 fps)
- **Duración**: Variable según el video
