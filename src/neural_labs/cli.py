from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .advanced.catalog import list_tracks as list_advanced_tracks, tracks as advanced_tracks
from .advanced.training import train_advanced
from .batch_inference import batch_predict
from .benchmarking import build_leaderboard, compare_runs, write_benchmark_report
from .catalog import ROOT, get_dataset, get_lab, list_labs
from .core.registry import MODEL_REGISTRY
from .cross_validation import cross_validate_lab
from .data_quality import quality_report
from .datasets import audit_bundle, describe_bundle, prepare_dataset
from .doctor import environment_doctor
from .drift import drift_report
from .experiments import run_lab
from .exporting import benchmark_formats, export_executorch, export_onnx, quantize_dynamic
from .inference import load_external_input, load_inference_package
from .model_registry import LocalModelRegistry
from .monitoring import monitoring_report
from .runtime import resolve_run, save_json
from .scaffold import create_lab_scaffold
from .schema import validate_repository
from .supply_chain import build_checksum_manifest, generate_provenance, generate_sbom


def _json(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def _add_seeds(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split-seed", type=int, default=42, help="Semilla usada solo para crear particiones de datos.")
    parser.add_argument("--training-seed", type=int, default=42, help="Semilla usada solo para inicialización y entrenamiento.")
    parser.add_argument("--seed", type=int, default=None, help=argparse.SUPPRESS)


def build_parser(fixed_lab: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Neural Network Training Labs 3: ingeniería de IA con datos públicos reales")
    subparsers = parser.add_subparsers(dest="command", required=fixed_lab is None)

    def add_common(command_parser: argparse.ArgumentParser, *, require_lab: bool = True) -> None:
        if fixed_lab is None and require_lab:
            command_parser.add_argument("--lab", required=True, choices=list_labs())
        command_parser.add_argument("--quick", action="store_true", help="Usa una fracción real del dataset y pocas épocas.")
        _add_seeds(command_parser)

    train = subparsers.add_parser("train", help="Entrena, selecciona por validación, congela y evalúa test una vez.")
    add_common(train)
    train.add_argument("--config", default="baseline", choices=["baseline", "improved"])
    train.add_argument("--device", default=None, choices=["auto", "cpu", "cuda", "mps"])
    train.add_argument("--output-dir", default="runs")
    train.add_argument("--tracker", default="json", choices=["json", "mlflow", "none"])
    train.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    train.add_argument("--compile", dest="compile_model", action=argparse.BooleanOptionalAction, default=None)
    train.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--num-workers", type=int, default=None)

    dataset = subparsers.add_parser("dataset", help="Descarga, prepara y audita un dataset real.")
    add_common(dataset)
    dataset.add_argument("--force", action="store_true")

    audit = subparsers.add_parser("audit", help="Comprueba separación train/validation/test.")
    add_common(audit)

    quality = subparsers.add_parser("quality", help="Genera calidad y deriva de los datos.")
    add_common(quality)
    quality.add_argument("--output", default=None)

    predict = subparsers.add_parser("predict", help="Ejecuta inferencia con una entrada externa.")
    if fixed_lab is None:
        predict.add_argument("--lab", required=True, choices=list_labs())
    predict.add_argument("--run", default="latest")
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])

    batch = subparsers.add_parser("batch-predict", help="Ejecuta inferencia por lotes desde CSV, JSON o NPY.")
    if fixed_lab is None:
        batch.add_argument("--lab", required=True, choices=list_labs())
    batch.add_argument("--run", default="latest")
    batch.add_argument("--input", type=Path, required=True)
    batch.add_argument("--output", type=Path, required=True)
    batch.add_argument("--batch-size", type=int, default=128)
    batch.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])

    export = subparsers.add_parser("export", help="Exporta, cuantiza o compara formatos de inferencia.")
    if fixed_lab is None:
        export.add_argument("--lab", required=True, choices=list_labs())
    export.add_argument("--run", default="latest")
    export.add_argument("--format", default="onnx", choices=["onnx", "int8", "executorch", "benchmark"])
    export.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True)

    catalog = subparsers.add_parser("catalog", help="Muestra laboratorios y datasets.")
    catalog.add_argument("--format", choices=["table", "json"], default="table")

    advanced_catalog = subparsers.add_parser("advanced", help="Muestra los laboratorios avanzados de especialización.")
    advanced_catalog.add_argument("--format", choices=["table", "json"], default="table")

    advanced_train = subparsers.add_parser("train-advanced", help="Entrena segmentación, audio, WGAN-GP, DDPM, SimCLR o transformer preentrenado.")
    advanced_train.add_argument("--track", required=True, choices=list_advanced_tracks())
    advanced_train.add_argument("--quick", action="store_true")
    advanced_train.add_argument("--split-seed", type=int, default=42)
    advanced_train.add_argument("--training-seed", type=int, default=42)
    advanced_train.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    advanced_train.add_argument("--output-dir", default="runs-advanced")
    advanced_train.add_argument("--lora", action=argparse.BooleanOptionalAction, default=False)

    subparsers.add_parser("models", help="Muestra el registro extensible de modelos.")
    subparsers.add_parser("doctor", help="Comprueba Python, hardware, extras y credenciales.")
    validate = subparsers.add_parser("validate", help="Valida catálogo, manifiestos y material educativo.")
    validate.add_argument("--warnings-as-errors", action="store_true")

    leaderboard = subparsers.add_parser("leaderboard", help="Consolida las mejores ejecuciones.")
    leaderboard.add_argument("--output", default="reports/leaderboard")

    compare = subparsers.add_parser("compare", help="Compara directorios de ejecución.")
    compare.add_argument("runs", nargs="+", type=Path)
    compare.add_argument("--output", type=Path, default=Path("reports/comparison.csv"))

    benchmark = subparsers.add_parser("benchmark", help="Repite entrenamiento con partición fija y varias semillas de entrenamiento.")
    if fixed_lab is None:
        benchmark.add_argument("--lab", required=True, choices=list_labs())
    benchmark.add_argument("--split-seed", type=int, default=42)
    benchmark.add_argument("--training-seeds", "--seeds", dest="training_seeds", type=int, nargs="+", default=[41, 42, 43])
    benchmark.add_argument("--config", default="baseline", choices=["baseline", "improved"])
    benchmark.add_argument("--quick", action="store_true")
    benchmark.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    benchmark.add_argument("--tracker", default="json", choices=["json", "mlflow", "none"])
    benchmark.add_argument("--output-dir", default="runs")

    cross_validate = subparsers.add_parser("cross-validate", help="Valida sobre desarrollo sin tocar el conjunto test.")
    if fixed_lab is None:
        cross_validate.add_argument("--lab", required=True, choices=list_labs())
    cross_validate.add_argument("--folds", type=int, default=5)
    cross_validate.add_argument("--split-seed", type=int, default=42)
    cross_validate.add_argument("--training-seeds", type=int, nargs="+", default=[41, 42, 43])
    cross_validate.add_argument("--quick", action="store_true")
    cross_validate.add_argument("--config", default="baseline", choices=["baseline", "improved"])
    cross_validate.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])

    registry = subparsers.add_parser("registry", help="Administra versiones y alias locales de modelos.")
    registry.add_argument("action", choices=["list", "register", "alias", "resolve", "promote"])
    registry.add_argument("--path", type=Path, default=Path("model-registry.json"))
    registry.add_argument("--name", default="default")
    registry.add_argument("--run", type=Path)
    registry.add_argument("--version", type=int)
    registry.add_argument("--alias", default="champion")
    registry.add_argument("--backend", choices=["local", "mlflow"], default="local")
    registry.add_argument("--tracking-uri", default=None)
    registry.add_argument("--metric", default="accuracy")
    registry.add_argument("--minimum", type=float, default=None)
    registry.add_argument("--maximum", type=float, default=None)
    registry.add_argument("--max-latency-ms", type=float, default=None)

    serve = subparsers.add_parser("serve", help="Inicia la API de inferencia FastAPI.")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")

    dashboard = subparsers.add_parser("dashboard", help="Inicia el panel Streamlit de experimentos y registro.")
    dashboard.add_argument("--port", type=int, default=8501)

    monitor = subparsers.add_parser("monitor", help="Resume predicciones recientes y deriva respecto de referencia.")
    monitor.add_argument("--log", type=Path, default=Path("monitoring/predictions.jsonl"))
    monitor.add_argument("--reference", type=Path, default=Path("monitoring/reference_stats.json"))
    monitor.add_argument("--limit", type=int, default=1000)

    distributed_train = subparsers.add_parser("train-distributed", help="Entrena con torchrun usando DDP o FSDP2.")
    if fixed_lab is None:
        distributed_train.add_argument("--lab", required=True, choices=list_labs())
    distributed_train.add_argument("--strategy", choices=["ddp", "fsdp2"], default="ddp")
    distributed_train.add_argument("--split-seed", type=int, default=42)
    distributed_train.add_argument("--training-seed", type=int, default=42)
    distributed_train.add_argument("--quick", action="store_true")
    distributed_train.add_argument("--config", default="baseline", choices=["baseline", "improved"])
    distributed_train.add_argument("--output-dir", default="runs-distributed")

    distributed = subparsers.add_parser("distributed", help="Diagnostica el entorno torchrun/DDP/FSDP2.")
    distributed.add_argument("--output", type=Path, default=None)

    supply = subparsers.add_parser("supply-chain", help="Genera SBOM, procedencia y hashes.")
    supply.add_argument("--output-dir", type=Path, default=Path("dist/security"))

    new_lab = subparsers.add_parser("new-lab", help="Crea el esqueleto de un laboratorio nuevo.")
    new_lab.add_argument("name")
    new_lab.add_argument("--title", default=None)

    card = subparsers.add_parser("dataset-card", help="Muestra la ficha estructurada del dataset.")
    if fixed_lab is None:
        card.add_argument("--lab", required=True, choices=list_labs())

    return parser


def _lab(args: argparse.Namespace, fixed_lab: str | None) -> str:
    return fixed_lab or args.lab


def execute(args: argparse.Namespace, fixed_lab: str | None = None) -> None:
    command = args.command
    if command == "catalog":
        records = [
            {
                "lab": lab_id,
                "title": get_lab(lab_id)["title"],
                "dataset": get_lab(lab_id)["dataset"],
                "source": get_lab(lab_id)["source"],
                "level": get_lab(lab_id)["level"],
            }
            for lab_id in list_labs()
        ]
        if args.format == "json":
            _json(records)
        else:
            for item in records:
                print(f"{item['lab']:36} {item['dataset']:28} {item['level']:12} {item['source']}")
        return
    if command == "advanced":
        records = advanced_tracks()
        if args.format == "json":
            _json(records)
        else:
            for item in records:
                print(f"{item['id']:36} {item['domain']:14} {item['dataset']:30} {item['title']}")
        return
    if command == "train-advanced":
        _json(train_advanced(
            args.track, quick=args.quick, split_seed=args.split_seed, training_seed=args.training_seed,
            device=args.device, output_dir=args.output_dir, use_lora=args.lora,
        ))
        return
    if command == "models":
        _json(MODEL_REGISTRY.describe())
        return
    if command == "doctor":
        _json(environment_doctor())
        return
    if command == "validate":
        issues = validate_repository(ROOT)
        payload = {
            "issues": [issue.as_dict() for issue in issues],
            "errors": sum(issue.severity == "error" for issue in issues),
            "warnings": sum(issue.severity == "warning" for issue in issues),
        }
        _json(payload)
        if payload["errors"] or (args.warnings_as_errors and payload["warnings"]):
            raise SystemExit(1)
        return
    if command == "leaderboard":
        _json(build_leaderboard(ROOT, ROOT / args.output))
        return
    if command == "compare":
        print(compare_runs([path.resolve() for path in args.runs], args.output.resolve()))
        return
    if command == "new-lab":
        print(create_lab_scaffold(args.name, title=args.title))
        return
    if command == "serve":
        import uvicorn

        uvicorn.run("neural_labs.deployment.api:app", host=args.host, port=args.port, reload=args.reload)
        return
    if command == "dashboard":
        import subprocess

        subprocess.run([sys.executable, "-m", "streamlit", "run", str(ROOT / "dashboard" / "app.py"), "--server.port", str(args.port)], check=True)
        return
    if command == "monitor":
        log_path = args.log if args.log.is_absolute() else ROOT / args.log
        reference_path = args.reference if args.reference.is_absolute() else ROOT / args.reference
        _json(monitoring_report(log_path, reference_path, limit=args.limit))
        return
    if command == "distributed":
        from .distributed import distributed_diagnostics

        _json(distributed_diagnostics(args.output))
        return
    if command == "supply-chain":
        output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
        output.mkdir(parents=True, exist_ok=True)
        _json(
            {
                "sbom": str(generate_sbom(output / "sbom.cdx.json")),
                "provenance": str(generate_provenance(ROOT, output / "provenance.json")),
                "checksums": str(build_checksum_manifest(ROOT, output / "SHA256SUMS")),
            }
        )
        return
    if command == "registry":
        if args.backend == "mlflow":
            if args.action != "register" or args.run is None:
                raise SystemExit("El backend MLflow admite aquí registry register --run ...")
            from .mlflow_registry import register_run_with_mlflow

            _json(register_run_with_mlflow(args.run, model_name=args.name, alias=args.alias, tracking_uri=args.tracking_uri))
            return
        registry = LocalModelRegistry(args.path if args.path.is_absolute() else ROOT / args.path)
        if args.action == "list":
            _json(registry.list())
        elif args.action == "register":
            if args.run is None:
                raise SystemExit("registry register requiere --run")
            _json(registry.register(args.name, args.run, alias=args.alias).__dict__)
        elif args.action == "alias":
            if args.version is None:
                raise SystemExit("registry alias requiere --version")
            _json(registry.set_alias(args.name, args.version, args.alias).__dict__)
        elif args.action == "promote":
            if args.version is None:
                raise SystemExit("registry promote requiere --version")
            _json(registry.promote(
                args.name, args.version, alias=args.alias, metric=args.metric, minimum=args.minimum,
                maximum=args.maximum, max_latency_ms=args.max_latency_ms,
            ).__dict__)
        else:
            _json(registry.resolve(args.name, args.alias).__dict__)
        return

    if command == "train-distributed":
        from .distributed_training import train_distributed

        lab_id = _lab(args, fixed_lab)
        _json(train_distributed(
            lab_id, strategy=args.strategy, split_seed=args.split_seed, training_seed=args.training_seed,
            quick=args.quick, config_name=args.config, output_dir=args.output_dir,
        ))
        return

    lab_id = _lab(args, fixed_lab)
    if command == "dataset-card":
        _json(get_dataset(lab_id))
        return
    if command in {"dataset", "audit", "quality"}:
        split_seed = args.seed if args.seed is not None else args.split_seed
        bundle = prepare_dataset(lab_id, quick=args.quick, seed=split_seed, force=getattr(args, "force", False))
        if command == "dataset":
            describe_bundle(bundle)
        elif command == "audit":
            _json(audit_bundle(bundle))
        else:
            report = {"quality": quality_report(bundle), "drift": drift_report(bundle)}
            if args.output:
                path = Path(args.output)
                if not path.is_absolute():
                    path = ROOT / path
                save_json(path, report)
                print(path)
            else:
                _json(report)
        return
    if command == "train":
        split_seed = args.seed if args.seed is not None else args.split_seed
        training_seed = args.seed if args.seed is not None else args.training_seed
        result = run_lab(
            lab_id,
            quick=args.quick,
            config_name=args.config,
            output_dir=args.output_dir,
            device=args.device,
            split_seed=split_seed,
            training_seed=training_seed,
            tracker=args.tracker,
            amp=args.amp,
            compile_model=args.compile_model,
            deterministic=args.deterministic,
            num_workers=args.num_workers,
        )
        _json({"lab": lab_id, "run_dir": str(result.run_dir), "metrics": result.metrics})
        return
    if command == "benchmark":
        results = []
        for training_seed in args.training_seeds:
            result = run_lab(
                lab_id,
                quick=args.quick,
                config_name=args.config,
                output_dir=args.output_dir,
                device=args.device,
                split_seed=args.split_seed,
                training_seed=training_seed,
                tracker=args.tracker,
            )
            results.append({"split_seed": args.split_seed, "training_seed": training_seed, "run_dir": str(result.run_dir), "metrics": result.metrics})
        report_dir = write_benchmark_report(lab_id, results, ROOT)
        _json({"lab": lab_id, "report_dir": str(report_dir), "runs": results})
        return
    if command == "cross-validate":
        _json(
            cross_validate_lab(
                lab_id,
                folds=args.folds,
                split_seed=args.split_seed,
                training_seeds=args.training_seeds,
                quick=args.quick,
                config_name=args.config,
                device=args.device,
            )
        )
        return
    if command == "predict":
        run_dir = resolve_run(lab_id, args.run)
        package = load_inference_package(run_dir, args.device)
        tensor = load_external_input(args.input, package.contract, package.run_dir)
        _json(package.predict_tensor(tensor))
        return
    if command == "batch-predict":
        run_dir = resolve_run(lab_id, args.run)
        output = args.output if args.output.is_absolute() else ROOT / args.output
        print(batch_predict(run_dir, args.input, output_path=output, batch_size=args.batch_size, device=args.device))
        return
    if command == "export":
        run_dir = resolve_run(lab_id, args.run)
        if args.format == "onnx":
            _json(export_onnx(run_dir, verify=args.verify).__dict__)
        elif args.format == "int8":
            _json(quantize_dynamic(run_dir).__dict__)
        elif args.format == "executorch":
            _json(export_executorch(run_dir).__dict__)
        else:
            _json(benchmark_formats(run_dir))
        return
    raise ValueError(command)


def main() -> None:
    execute(build_parser().parse_args())


def run_fixed_lab(lab_id: str) -> None:
    arguments = sys.argv[1:]
    known_commands = {
        "train", "dataset", "audit", "quality", "predict", "export", "catalog", "models", "doctor", "validate",
        "leaderboard", "compare", "benchmark", "cross-validate", "registry", "serve", "train-distributed", "distributed", "supply-chain",
        "advanced", "train-advanced", "batch-predict", "dashboard", "monitor",
        "new-lab", "dataset-card",
    }
    if not arguments or arguments[0] not in known_commands:
        arguments = ["train", *arguments]
    execute(build_parser(fixed_lab=lab_id).parse_args(arguments), fixed_lab=lab_id)


if __name__ == "__main__":
    main()
