# Sitio de estudio y navegación

El repositorio publica, además de esta documentación técnica, un **sitio de estudio** en GitHub Pages pensado para recorrer los laboratorios en orden, como un curso.

- 🌐 <https://vladimiracunadev-create.github.io/neural-network-training-labs/>

## Qué ofrece

- Una **portada** con las 31 rutas agrupadas (25 centrales + 6 especializaciones avanzadas).
- Una **página por laboratorio** que renderiza su `README.md`, `theory.md`, `experiments.md` y `assessment.md` en un solo documento.
- Un **paginador anterior / siguiente** que crea un recorrido lineal de la neurona en NumPy (`00`) a SimCLR (`30`), incluyendo la transición entre `labs/` y `advanced_labs/`.
- Tema claro y oscuro automático.

## Doble navegación: repositorio y sitio

El flujo anterior/siguiente existe en dos lugares que se mantienen sincronizados:

| Dónde | Qué se ve | Quién lo genera |
|---|---|---|
| **GitHub** (Markdown del laboratorio) | Una línea de navegación bajo el título y un bloque «Navegación del curso» al final de cada `README.md` | `scripts/add_lab_nav.py` |
| **Sitio de estudio** (GitHub Pages) | El paginador propio del sitio | `scripts/generate_site.py` |

Para evitar duplicación, `generate_site.py` elimina los bloques de navegación del Markdown al renderizar: en el sitio la navegación la aporta su propio paginador.

## Cómo se genera

```bash
# 1) Insertar/actualizar la navegación anterior/siguiente en cada README (idempotente)
python scripts/add_lab_nav.py

# 2) Generar el sitio estático en site/
python -m pip install "markdown>=3.6"
python scripts/generate_site.py
```

El sitio se construye a partir del Markdown del repositorio: es su única fuente de verdad. El directorio `site/` no se versiona (está en `.gitignore`); lo regenera el flujo de integración continua.

## Publicación automática

El workflow [`deploy-pages.yml`](https://github.com/vladimiracunadev-create/neural-network-training-labs/blob/main/.github/workflows/deploy-pages.yml) regenera y publica el sitio en cada `push` a `main` que toque `labs/`, `advanced_labs/` o el generador. Comprueba que el sitio salió completo (portada, hoja de estilos y 32 páginas) antes de subir el artefacto.

## Mantenimiento

Si cambian los títulos, se añade un laboratorio o se reordena el recorrido:

1. Ejecuta `python scripts/add_lab_nav.py` para actualizar la navegación en los `README.md`.
2. Ejecuta `python scripts/generate_site.py` para regenerar el sitio localmente y revisarlo.
3. Al hacer `push` a `main`, el sitio se publica solo.

El mapa de emojis por laboratorio se define, de forma idéntica, en ambos scripts; si añades un laboratorio, agrégalo a los dos.
