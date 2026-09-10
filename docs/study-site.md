# Sitio de estudio y navegación

El repositorio publica un **sitio de estudio** en GitHub Pages organizado como curso: siete módulos y 31 clases independientes, numeradas del 01 al 31.

- 🌐 <https://vladimiracunadev-create.github.io/neural-network-training-labs/>

## El orden es el número

Se estudia de la clase **01** a la **31**. Los prefijos técnicos `00`–`30` se conservan por compatibilidad. El orden es el mismo en Markdown, el sitio y el HTML local.

## Dos niveles: módulo y clase

El recorrido se agrupa en **siete módulos**, que son tramos contiguos de las clases 01 → 31. Cada módulo tiene su propia página, pero cada clase conserva una URL, materiales y evaluación independientes:

| Nivel | Markdown | HTML local | Sitio de Pages |
|---|---|---|---|
| Índice del recorrido | [`parts/README.md`](https://github.com/vladimiracunadev-create/neural-network-training-labs/blob/main/parts/README.md) | `index.html` (raíz) | `index.html` |
| Módulo | `parts/<NN-slug>.md` | `parts/<NN-slug>.html` | `parts/<NN-slug>.html` |
| Clase | `labs/<slug>/README.md` y sus tres documentos | `labs/<slug>/index.html` | `labs/<slug>/index.html` |

Por eso **todos los enlaces apuntan a archivos, no a carpetas**: desde la portada se entra a un módulo, del módulo a una clase, y de una clase a la siguiente, sin ver nunca el árbol del repositorio.

## Tres superficies, una fuente

| Superficie | Qué se ve | Quién la genera | ¿Se versiona? |
|---|---|---|---|
| **GitHub** (Markdown) | La clase completa —pregunta esencial, mapa visual, teoría, práctica y cierre—, con la posición `Clase N / 31`, su módulo, la barra de documentos y el bloque «Navegación del programa» | `scripts/build_lab_docs.py` | Sí |
| **HTML local** (`<lab>/index.html`) | La clase completa como página autocontenida, con paginador, anclas por documento y enlaces relativos a cuadernos y configuraciones | `scripts/generate_lab_html.py` | Sí |
| **Sitio de estudio** (GitHub Pages) | Portada agrupada por módulos, página por módulo y página por clase con el paginador del sitio | `scripts/generate_site.py` | No (`site/` está en `.gitignore`) |

Las tres se construyen desde el mismo Markdown de `labs/` y `advanced_labs/`, y comparten la definición de los módulos: la constante `PARTS` de `scripts/build_lab_docs.py`, de donde la importan los otros dos generadores.

## Qué enlaza con qué

Cada clase publica cuatro documentos —`README.md`, `theory.md`, `experiments.md` y `assessment.md`— y todos comparten la misma capa navegable. La guía incrusta la explicación de `theory.md`, de modo que la clase se lee de corrido sin saltar de archivo; `theory.md` sigue siendo la fuente que se edita y la que aporta la bibliografía:

- **Arriba:** posición en el curso (`Clase 04 / 31`) y módulo al que pertenece; salto a la clase anterior y siguiente conservando el documento actual.
- **Abajo:** tabla anterior / índice / siguiente, enlaces a los otros documentos y a los tres cuadernos, y salidas hacia el módulo, la portada del repositorio, el sitio de estudio y la página HTML local.

En la página `index.html` los enlaces se reescriben para funcionar sin conexión: los documentos de la clase pasan a ser anclas de la misma página, los saltos entre clases apuntan al `index.html` vecino, los enlaces al módulo apuntan a `parts/<slug>.html`, y los cuadernos, configuraciones y fichas de dataset quedan como rutas relativas.

## Cómo se genera

```bash
# 1) Guía, experimentos, evaluación y navegación, más los 7 módulos y su índice
python scripts/build_lab_docs.py

# 2) Página HTML por clase y por módulo + índice offline en la raíz
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

El workflow [`deploy-pages.yml`](https://github.com/vladimiracunadev-create/neural-network-training-labs/blob/main/.github/workflows/deploy-pages.yml) regenera y publica el sitio en cada `push` a `main` que toque `labs/`, `advanced_labs/` o el generador. Comprueba que el sitio salió completo (portada, hoja de estilos, siete páginas de módulo y 39 páginas en total) antes de subir el artefacto.

## Mantenimiento

Si cambian los títulos, se añade una clase o se reordena el programa:

1. Ejecuta `python scripts/build_class_materials.py` para regenerar la identidad, guía docente y mapa visual de cada clase.
2. Ejecuta `python scripts/build_lab_docs.py` para regenerar guías, planes de experimentos, evaluaciones, páginas de módulo y navegación.
3. Ejecuta `python scripts/generate_lab_html.py` para regenerar las páginas HTML versionadas.
4. Ejecuta `python scripts/generate_site.py` para revisar el sitio localmente.
5. Al hacer `push` a `main`, el sitio se publica solo.

Si añades una clase, revisa dos sitios: el mapa de emojis, que se define de forma idéntica en `build_lab_docs.py` y `generate_site.py`, y la constante técnica `PARTS`, que debe seguir cubriendo todo el rango numérico sin huecos ni solapes —si una clase queda fuera de todo módulo, los generadores fallan con un error explícito en vez de publicarla huérfana.
