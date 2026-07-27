# Uso de KaggleHub

## Laboratorios

- `06_autoencoder_anomaly`: Credit Card Fraud Detection.

## Autenticación

Instale el extra y autentique una vez:

```bash
pip install -e ".[kaggle]"
python -c "import kagglehub; kagglehub.login()"
```

KaggleHub también admite configuración mediante token y credenciales según su documentación oficial. No suba credenciales, `kaggle.json`, tokens ni `.env` al repositorio.

## Condiciones

Antes de descargar:

1. abra la ficha del dataset;
2. revise licencia y reglas de uso;
3. acepte las condiciones cuando Kaggle lo solicite;
4. confirme que su reutilización es compatible con el objetivo del proyecto.

## Ejecución

```bash
neural-labs dataset --lab 06_autoencoder_anomaly --quick
neural-labs train --lab 06_autoencoder_anomaly --quick
```

Si la descarga falla, el mensaje no debe reemplazarse con datos inventados. Corrija la autenticación o use otro dataset real autorizado.
