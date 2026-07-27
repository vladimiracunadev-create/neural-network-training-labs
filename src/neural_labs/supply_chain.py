from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

from .runtime import save_json


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_checksum_manifest(root: Path, output: Path, *, exclude: set[str] | None = None) -> Path:
    excluded = exclude or {"SHA256SUMS", output.name}
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in excluded and ".git" not in path.parts:
            lines.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def generate_sbom(output: Path) -> Path:
    packages = []
    for distribution in sorted(distributions(), key=lambda item: item.metadata.get("Name", "").lower()):
        name = distribution.metadata.get("Name")
        if not name:
            continue
        packages.append({"name": name, "version": distribution.version, "type": "python"})
    payload = {
        "bomFormat": "CycloneDX-compatible-minimal",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:neural-labs-{datetime.now(timezone.utc).timestamp()}",
        "components": packages,
    }
    save_json(output, payload)
    return output


def generate_provenance(root: Path, output: Path) -> Path:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = "unavailable"
    payload = {
        "predicateType": "https://slsa.dev/provenance/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {"repository": os.environ.get("GITHUB_REPOSITORY", "local"), "commit": commit},
        "builder": {"python": sys.version, "platform": platform.platform()},
        "materials": {"pyproject_sha256": sha256_file(root / "pyproject.toml")},
    }
    save_json(output, payload)
    return output


def cosign_blob_command(artifact: Path) -> list[str]:
    return ["cosign", "sign-blob", "--yes", "--bundle", f"{artifact}.sigstore.json", str(artifact)]


def verify_blob_command(artifact: Path, identity: str, issuer: str) -> list[str]:
    return [
        "cosign",
        "verify-blob",
        "--bundle",
        f"{artifact}.sigstore.json",
        "--certificate-identity",
        identity,
        "--certificate-oidc-issuer",
        issuer,
        str(artifact),
    ]
