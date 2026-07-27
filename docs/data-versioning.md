# Versionado de datos y pipelines con DVC

Los datasets no se guardan en Git. Sus manifiestos, URLs, licencias, políticas de partición y fingerprints sí se versionan.

## Pipeline incluido

`dvc.yaml` contiene tres etapas:

1. `validate`: valida catálogo, configuraciones y material educativo.
2. `smoke`: ejecuta el laboratorio definido en `params.yaml`.
3. `leaderboard`: consolida las ejecuciones disponibles.

```bash
pip install -e ".[data-versioning]"
dvc repro validate
dvc repro smoke
```

## Remotos

El repositorio no incluye un remoto preconfigurado. Cada organización debe definir uno compatible con sus políticas de privacidad y retención. Nunca suba credenciales, datasets con redistribución prohibida ni datos personales sin una base legal y controles adecuados.

## Principio

Git versiona código y metadatos; DVC puede versionar punteros, caché y pipelines. La ficha del dataset sigue siendo obligatoria aun cuando los archivos estén en un remoto privado.
