# Ética, licencias y límites

## Código y datos

MIT cubre el código original del repositorio. No modifica la licencia ni las condiciones de los datasets. Cada manifiesto declara la fuente y, cuando es conocida, la licencia.

## Datos sensibles

Varios laboratorios usan datos de salud, empleo, fraude o clientes. Son apropiados para aprendizaje técnico, pero no para desplegar decisiones reales sin:

- base jurídica y autorización;
- análisis de sesgo y representatividad;
- protección de privacidad;
- evaluación por subgrupos;
- monitoreo y revisión humana;
- documentación de impacto y mecanismos de apelación.

## Adult Census

El laboratorio de explicabilidad no convierte correlaciones históricas en reglas legítimas para empleo o crédito. Las atribuciones explican el modelo, no justifican decisiones ni prueban causalidad.

## Breast Cancer Wisconsin

El laboratorio no es una herramienta diagnóstica. Una métrica de test educativa no sustituye validación clínica, evaluación prospectiva ni aprobación regulatoria.

## Fraude y churn

Las clases minoritarias, el cambio temporal y los costos asimétricos exigen evaluación adicional. No reequilibre el test para aparentar mejor desempeño.

## Model cards

Toda ejecución genera una ficha con uso previsto, protocolo, resultados y limitaciones. Complétela antes de publicar un modelo.
