# Benchmarking y comparación

Una sola semilla no permite evaluar estabilidad. El comando `benchmark` repite un laboratorio con semillas declaradas:

```bash
neural-labs benchmark --lab 02_mlp_nonlinear --seeds 41 42 43
```

Después:

```bash
neural-labs leaderboard
neural-labs compare runs/02_mlp_nonlinear/RUN_A runs/02_mlp_nonlinear/RUN_B
```

## Reglas

- Mantenga el mismo dataset y política de partición.
- Compare presupuestos equivalentes.
- Elija arquitectura e hiperparámetros con validation.
- Informe promedio, dispersión e intervalos, no solo el mejor número.
- Compare precisión, tiempo, parámetros, tamaño y latencia.
- No mezcle runs rápidos con resultados completos en una conclusión final.

El leaderboard sirve para navegación; no sustituye una revisión del reporte y la model card.
