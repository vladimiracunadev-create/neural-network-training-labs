# Seguridad

## Reporte responsable

No publique vulnerabilidades explotables, tokens, credenciales ni datos privados en issues. Utilice un canal privado del mantenedor y entregue pasos mínimos de reproducción, impacto y versión afectada.

## Secretos

- No versionar `kaggle.json`, tokens de Hugging Face, variables MLflow ni claves de almacenamiento DVC.
- Use variables de entorno o gestores de secretos.
- Revise artefactos antes de subirlos: predicciones y muestras pueden contener información sensible.

## Datos y modelos

La seguridad incluye privacidad, membresía inferible, sesgo, envenenamiento, artefactos maliciosos y uso fuera de alcance. No cargue checkpoints no confiables mediante mecanismos que ejecuten código arbitrario. Prefiera formatos y APIs de carga seguros, verifique procedencia y limite permisos.

## Dependencias

```bash
pip install -e ".[security]"
make security
```

La CI ejecuta análisis estático y auditoría de dependencias de forma separada para que una caída de red no oculte el estado de las pruebas unitarias.
