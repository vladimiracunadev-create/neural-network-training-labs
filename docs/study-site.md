# Sitio de estudio y navegación

El repositorio publica, además de esta documentación técnica, un **sitio de estudio** en GitHub Pages pensado para recorrer los laboratorios en orden, como un curso.

- 🌐 <https://vladimiracunadev-create.github.io/neural-network-training-labs/>

## El orden es el número

Se estudia de la ruta **00** a la **30**, sin saltos. Ese orden es el mismo en las tres superficies (Markdown, sitio y HTML local) y es el que siguen todos los enlaces *anterior / siguiente*. Los bloques temáticos del `README.md` principal (00–02, 03–07, 08–12, 13–15, 16–20, 21–24, 25–30) son tramos **contiguos** de esa secuencia, nunca un orden alternativo.

## Tres superficies, una fuente

| Superficie | Qué se ve | Quién la genera | ¿Se versiona? |
|---|---|---|---|
| **GitHub** (Markdown) | Ficha del laboratorio, navegación superior con posición `Ruta N / 31`, barra de los cuatro documentos y bloque «Navegación del recorrido» al final de cada documento | `scripts/build_lab_docs.py` | Sí |
| **HTML local** (`<lab>/index.html`) | La clase completa como página autocontenida, con paginador, anclas por documento y enlaces relativos a cuadernos y configuraciones | `scripts/generate_lab_html.py` | Sí |
| **Sitio de estudio** (GitHub Pages) | Portada con las 31 rutas y una página por laboratorio con el paginador del sitio | `scripts/generate_site.py` | No (`site/` está en `.gitignore`) |

Las tres se construyen desde el mismo Markdown de `labs/` y `advanced_labs/`: esa es la única fuente de verdad.

## Qué enlaza con qué

Cada laboratorio publica cuatro documentos —`README.md`, `theory.md`, `experiments.md` y `assessment.md`— y todos comparten la misma capa navegable:

- **Arriba:** posición en el recorrido (`Ruta 4 / 31`), salto al laboratorio anterior y al siguiente *conservando el documento actual* (de `theory.md` se pasa a `theory.md`), enlace al índice y barra con los cuatro documentos, marcando el actual.
- **Abajo:** tabla anterior / índice / siguiente, enlaces a los otros documentos y a los tres cuadernos, y salidas hacia la portada del repositorio, el sitio de estudio y la página HTML local.

En la página `index.html` los enlaces se reescriben para funcionar sin conexión: los documentos del laboratorio pasan a ser anclas de la misma página, los saltos entre laboratorios apuntan al `index.html` vecino, y los cuadernos, configuraciones y fichas de dataset quedan como rutas relativas.

## Cómo se genera

```bash
# 1) Ficha y navegación en los 124 documentos Markdown (idempotente)
python scripts/build_lab_docs.py

# 2) Página HTML autocontenida por laboratorio + índice offline en la raíz
python -m pip install "markdown>=3.6"
python scripts/generate_lab_html.py

# 3) Sitio estático de GitHub Pages en site/
python scripts/generate_site.py
```

El orden importa: el HTML se construye a partir del Markdown ya actualizado.

Ambos generadores aceptan `--check`, que no escribe nada y falla si algo quedó desfasado. Es lo que ejecuta la integración continua:

```bash
python scripts/build_lab_docs.py --check
python scripts/generate_lab_html.py --check
```

## Publicación automática

El workflow [`deploy-pages.yml`](https://github.com/vladimiracunadev-create/neural-network-training-labs/blob/main/.github/workflows/deploy-pages.yml) regenera y publica el sitio en cada `push` a `main` que toque `labs/`, `advanced_labs/` o el generador. Comprueba que el sitio salió completo (portada, hoja de estilos y 32 páginas) antes de subir el artefacto.

## Mantenimiento

Si cambian los títulos, se añade un laboratorio o se reordena el recorrido:

1. Ejecuta `python scripts/build_lab_docs.py` para actualizar ficha y navegación en el Markdown.
2. Ejecuta `python scripts/generate_lab_html.py` para regenerar las páginas HTML versionadas.
3. Ejecuta `python scripts/generate_site.py` para revisar el sitio localmente.
4. Al hacer `push` a `main`, el sitio se publica solo.

El mapa de emojis por laboratorio se define de forma idéntica en `build_lab_docs.py` y `generate_site.py`; si añades un laboratorio, agrégalo a los dos.
