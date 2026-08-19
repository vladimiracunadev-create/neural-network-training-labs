# Registro de fuentes

`bibliography.json` es el registro de procedencia del programa. Toda afirmación que
una clase apoya en una obra externa —libro, artículo, norma, documentación oficial o
**dataset**— tiene aquí su entrada, con un localizador que se puede resolver.

Antes de este registro, el repositorio prometía «datasets públicos reales» y cuidaba la
cadena de suministro del código, pero no registraba la procedencia ni la licencia de sus
datos. Los datasets están ahora declarados como fuentes de primer orden, al mismo nivel
que los papers.

## Las tres formas de localizador

Se admiten exactamente tres, y el verificador comprueba que cada entrada use la que le
corresponde por su tipo:

| Tipo | Localizador | Forma canónica |
|---|---|---|
| `book` | ISBN-13 con dígito de control válido | `https://openlibrary.org/isbn/{isbn13}` |
| `paper` | DOI | `https://doi.org/{doi}` |
| `standard`, `reference`, `dataset` | URL https de la fuente primaria, con `accessed` | la URL misma |

Los DOI de artículos que solo existen como preprint son los que **arXiv** registra en
DataCite (`10.48550/arXiv.NNNN.NNNNN`): son DOI reales y resolubles, no una construcción
de este repositorio.

## Qué lleva una entrada de dataset

Además de lo común a todas las entradas, un `dataset` declara:

- `provenance` — dónde vive realmente el artefacto y qué colección lo publica;
- `license` y `license_short` — la licencia **tal como la declara la fuente**, no la que se
  le supone; cuando la fuente no declara ninguna, eso también se dice;
- `version` — la distribución concreta: la ficha de UCI, la revisión fijada de Hugging
  Face, la versión del paquete;
- `integrity` — el SHA-256 de cada artefacto, con su URL y su tamaño en bytes;
- `cite_as` — la cita académica que **pide su autor**, no una redactada aquí;
- `used_in` — las rutas que lo usan, derivado de la estructura de ficheros.

## Huecos declarados

Lo que no resuelve se marca `"status": "pendiente"` con su `pending_reason`, y **no se
elimina**. Un hueco declarado es información; un hueco rellenado por intuición es una
invención con formato de bibliografía. Hay dos huecos posibles y son independientes:

- una entrada sin localizador (`status: pendiente`) —por ejemplo, un artículo de actas
  que su editorial nunca registró en Crossref;
- un dataset con procedencia verificada pero sin checksum
  (`integrity.status: pendiente`) —por ejemplo, uno cuya descarga exige credenciales.

## Las dos capas de comprobación

Están separadas a propósito. Si la red entrara en el CI, el CI se volvería inestable y
acabaría ignorándose.

```bash
# offline, determinista, bloquea en CI
python scripts/verify-sources

# regenera `used_in` y las cifras del README a partir del registro
python scripts/verify-sources --sync

# en red, manual o programado, NO bloquea
python scripts/refresh-sources
python scripts/refresh-sources --write        # anota el resultado
python scripts/refresh-sources --checksums    # recalcula el SHA-256 de los datasets
```

`verify-sources` comprueba el esquema, los identificadores y sus formas canónicas, que
toda obra citada en una clase esté registrada, que ninguna entrada quede sin usar, que
`used_in` coincida con la estructura de ficheros, que ningún bloque de fuentes se repita
entre clases, que cada cita diga **qué uso hace esa clase** de la fuente, y que las cifras
del README coincidan con el recuento del registro.

`refresh-sources` resuelve los ISBN contra Open Library y los DOI contra Crossref o
DataCite comparando título y autores, consulta cada URL, y reporta lo que dejó de
resolver sin borrarlo.

## Cómo añadir una fuente

1. Añada la entrada a `bibliography.json` con su localizador resoluble.
2. Cítela en el `theory.md` que la usa, diciendo **para qué la usa esa clase**.
3. Ejecute `python scripts/verify-sources --sync` y luego `python scripts/verify-sources`.

Si el localizador no se puede resolver, márquela `pendiente` con el motivo. No invente
un ISBN, un DOI ni una fecha.
