#!/usr/bin/env python3
from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def check(url: str, timeout: float = 15.0) -> dict[str, object]:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "neural-network-training-labs-source-check/2.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=ssl.create_default_context()) as response:
            return {"url": url, "ok": 200 <= response.status < 400, "status": response.status, "final_url": response.url}
    except urllib.error.HTTPError as exc:
        if exc.code in {403, 405}:
            try:
                fallback = urllib.request.Request(url, method="GET", headers={"User-Agent": "neural-network-training-labs-source-check/2.0", "Range": "bytes=0-0"})
                with urllib.request.urlopen(fallback, timeout=timeout, context=ssl.create_default_context()) as response:
                    return {"url": url, "ok": 200 <= response.status < 400, "status": response.status, "final_url": response.url, "fallback": "GET"}
            except Exception as fallback_exc:
                return {"url": url, "ok": False, "error": str(fallback_exc)}
        return {"url": url, "ok": False, "status": exc.code, "error": str(exc)}
    except Exception as exc:
        return {"url": url, "ok": False, "error": str(exc)}


def main() -> None:
    catalog = yaml.safe_load((ROOT / "configs" / "datasets.yaml").read_text(encoding="utf-8"))["datasets"]
    results = {name: check(item["source_ref"]) for name, item in catalog.items()}
    output = ROOT / "reports" / "dataset-source-check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(output)
    if not all(item["ok"] for item in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
