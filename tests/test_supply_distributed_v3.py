from pathlib import Path

import torch

from neural_labs.distributed import DistributedContext, cleanup_distributed, distributed_diagnostics, wrap_distributed
from neural_labs.supply_chain import build_checksum_manifest, cosign_blob_command, generate_provenance, generate_sbom, sha256_file, verify_blob_command


def test_supply_chain_artifacts(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    data = tmp_path / "file.txt"
    data.write_text("hello")
    assert len(sha256_file(data)) == 64
    checksums = build_checksum_manifest(tmp_path, tmp_path / "SHA256SUMS")
    assert "file.txt" in checksums.read_text()
    assert generate_sbom(tmp_path / "sbom.json").exists()
    assert generate_provenance(tmp_path, tmp_path / "provenance.json").exists()
    assert cosign_blob_command(data)[0] == "cosign"
    assert "verify-blob" in verify_blob_command(data, "id", "issuer")


def test_distributed_single_process(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    payload = distributed_diagnostics(tmp_path / "dist.json")
    assert payload["world_size"] == 1
    model = torch.nn.Linear(2, 1)
    wrapped = wrap_distributed(model, DistributedContext(0, 1, 0, "gloo", "cpu"), "ddp")
    assert wrapped is model
    cleanup_distributed()
