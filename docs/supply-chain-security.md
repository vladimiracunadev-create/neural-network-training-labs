# Seguridad de cadena de suministro

El proyecto genera hashes, SBOM y procedencia local.

```bash
neural-labs supply-chain
```

## Release firmado

El workflow de release puede ejecutar:

```bash
cosign sign-blob --yes --bundle artifact.zip.sigstore.json artifact.zip
cosign verify-blob \
  --bundle artifact.zip.sigstore.json \
  --certificate-identity <identidad> \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com \
  artifact.zip
```

La verificación debe comprobar identidad esperada, emisor y transparencia. Un hash por sí solo detecta cambios, pero no prueba quién produjo el artefacto.
