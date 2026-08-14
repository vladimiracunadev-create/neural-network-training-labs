# Sitio de estudio y navegación

El repositorio publica, además de esta documentación técnica, un **sitio de estudio** en GitHub Pages pensado para recorrer los laboratorios en orden, como un curso.

- 🌐 <https://vladimiracunadev-create.github.io/neural-network-training-labs/>

## El orden es el número

Se estudia de la ruta **00** a la **30**, sin saltos. Ese orden es el mismo en las tres superficies (Markdown, sitio y HTML local) y es el que siguen todos los enlaces *anterior / siguiente*.

## Dos niveles: parte y clase

El recorrido se agrupa en **siete partes**, que son tramos **contiguos** de la secuencia 00 → 30 —nunca un orden alternativo—: 00–02, 03–07, 08–12, 13–15, 16–20, 21–24 y 25–30. Cada parte tiene su propia página, que evita aterrizar en el listado de archivos de una carpeta:

| Nivel | Markdown | HTML local | Sitio de Pages |
|---|---|---|---|
| Índice del recorrido | [`parts/README.md`](../parts/README.md) | `index.html` (raíz) | `index.html` |
| Parte | `parts/<NN-slug>.md` | `parts/<NN-slug>.html` | `parts/<NN-slug>.html` |
| Clase | `labs/<slug>/README.md` y sus tres documentos | `labs/<slug>/index.html` | `labs/<slug>/index.html` |

Por eso **todos los enlaces apuntan a archivos, no a carpetas**: desde la portada se entra a una parte, de la parte a una clase, y de una clase a la siguiente, sin ver nunca el árbol del repositorio.

## Tres superficies, una fuente

| Superficie | Qué se ve | Quién la genera | ¿Se versiona? |
|---|---|---|---|
| **GitHub** (Markdown) | La clase completa —teoría incrustada, comandos explicados y paso a paso—, con la posición `Ruta N / 31`, su parte, la barra de documentos y el bloque «Navegación del recorrido» | `scripts/build_lab_docs.py` | Sí |
| **HTML local** (`<lab>/index.html`) | La clase completa como página autocontenida, con paginador, anclas por documento y enlaces relativos a cuadernos y configuraciones | `scripts/generate_lab_html.py` | Sí |
| **Sitio de estudio** (GitHub Pages) | Portada agrupada por partes, página por parte y página por laboratorio con el paginador del sitio | `scripts/generate_site.py` | No (`site/` está en `.gitignore`) |

Las tres se construyen desde el mismo Markdown de `labs/` y `advanced_labs/`, y comparten la definición de las partes: la constante `PARTS` de `scripts/build_lab_docs.py`, de donde la importan los otros dos generadores.

## Qué enlaza con qué

Cada laboratorio publica cuatro documentos —`README.md`, `theory.md`, `experiments.md` y `assessment.md`— y todos comparten la misma capa navegable. La guía incrusta la explicación de `theory.md`, de modo que la clase se lee de corrido sin saltar de archivo; `theory.md` sigue siendo la fuente que se edita y la que aporta la bibliografía:

- **Arriba:** posición en el recorrido (`Ruta 4 / 31`) y parte a la que pertenece; salto al laboratorio anterior y al siguiente *conservando el documento actual* (de `theory.md` se pasa a `theory.md`); enlace al índice y barra con los cuatro documentos, marcando el actual.
- **Abajo:** tabla anterior / índice / siguiente, enlaces a los otros documentos y a los tres cuadernos, y salidas hacia la parte, la portada del repositorio, el sitio de estudio y la página HTML local.

En la página `index.html` los enlaces se reescriben para funcionar sin conexión: los documentos del laboratorio pasan a ser anclas de la misma página, los saltos entre laboratorios apuntan al `index.html` vecino, los enlaces a la parte apuntan a `parts/<slug>.html`, y los cuadernos, configuraciones y fichas de dataset quedan como rutas relativas.

## Cómo se genera

```bash
# 1) Guía, experimentos, evaluación y navegación, más las 7 partes y su índice
python scripts/build_lab_docs.py

# 2) Página HTML por laboratorio y por parte + índice offline en la raíz
python -m pip install "markdown>=3.6" "PyYAML>=6"
python scripts/generate_lab_html.py

# 3) Sitio estático de GitHub Pages en site/ (39 páginas)
python scripts/generate_site.py
```

El orden importa: el HTML se construye a partir del Markdown ya actualizado.

Ambos generadores aceptan `--check`, que no escribe nada y falla si algo quedó desfasado. Es lo que ejecuta la integración continua:

```bash
python scripts/build_lab_docs.py --check
python scripts/generate_lab_html.py --check
```

## Publicación automática

El workflow [`deploy-pages.yml`](https://github.com/vladimiracunadev-create/neural-network-training-labs/blob/main/.github/workflows/deploy-pages.yml) regenera y publica el sitio en cada `push` a `main` que toque `labs/`, `advanced_labs/` o el generador. Comprueba que el sitio salió completo (portada, hoja de estilos, las siete páginas de parte y 39 páginas en total) antes de subir el artefacto.

## Mantenimiento

Si cambian los títulos, se añade un laboratorio o se reordena el recorrido:

1. Ejecuta `python scripts/build_lab_docs.py` para regenerar guías, planes de experimentos, evaluaciones, páginas de parte y navegación.
2. Ejecuta `python scripts/generate_lab_html.py` para regenerar las páginas HTML versionadas.
3. Ejecuta `python scripts/generate_site.py` para revisar el sitio localmente.
4. Al hacer `push` a `main`, el sitio se publica solo.

Si añades un laboratorio, revisa dos sitios: el mapa de emojis, que se define de forma idéntica en `build_lab_docs.py` y `generate_site.py`, y la constante `PARTS`, que debe seguir cubriendo todo el rango numérico sin huecos ni solapes —si una ruta queda fuera de toda parte, los generadores fallan con un error explícito en vez de publicar una clase huérfana.
