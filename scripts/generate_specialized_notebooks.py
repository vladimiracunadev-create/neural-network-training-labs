from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import nbformat
import yaml
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
LABS = yaml.safe_load((ROOT / "configs" / "labs.yaml").read_text(encoding="utf-8"))["labs"]

DOMAIN_CONTENT = {
    "vision": {
        "question": "¿Qué patrones locales aprende cada bloque convolucional y cómo cambia el campo receptivo?",
        "visual": """# Visualización sugerida: imágenes, activaciones y errores por clase\n# Descomenta después de preparar bundle.\n# from neural_labs.visualization import plot_class_distribution\n# plot_class_distribution(bundle)""",
        "analysis": "Compara mapas de activación, matriz de confusión, robustez a ruido, costo de inferencia y ejemplos mal clasificados.",
        "exercise": "Modifica el ancho de la CNN y mide el intercambio entre macro-F1, parámetros y latencia.",
    },
    "text": {
        "question": "¿Cómo influyen padding, vocabulario, longitud máxima y atención en la decisión?",
        "visual": """# Inspección de tokens y longitudes sin abrir test\n# texts = bundle.raw['train_texts'][:5]\n# print(texts)\n# print(bundle.metadata.get('vocabulary', {}) and list(bundle.metadata['vocabulary'])[:20])""",
        "analysis": "Analiza errores por longitud, palabras desconocidas, negación, matriz de atención y sensibilidad al truncamiento.",
        "exercise": "Compara RNN/transformer desde cero con una línea base TF-IDF y documenta cuándo la red justifica su costo.",
    },
    "time_series": {
        "question": "¿Cómo cambia el error cuando aumenta el horizonte y cómo evita el protocolo mirar el futuro?",
        "visual": """# Comprobación del orden cronológico y ventanas\n# print(bundle.summary)\n# assert bundle.summary.get('chronological_split', True)""",
        "analysis": "Realiza backtesting walk-forward, evalúa por horizonte y compara LSTM, TCN, persistencia y media móvil.",
        "exercise": "Agrega una TCN y compara MAE/RMSE por estación, hora y horizonte de predicción.",
    },
    "graph": {
        "question": "¿Cuánto aporta la estructura de aristas frente a usar solo atributos de nodos?",
        "visual": """# Resumen del grafo real\n# graph = bundle.train\n# print(graph.num_nodes, graph.num_edges, graph.num_node_features)""",
        "analysis": "Compara GCN, GraphSAGE, GAT y MLP sin aristas; inspecciona vecindarios y embeddings.",
        "exercise": "Evalúa sensibilidad al retirar aristas y explica qué clases dependen más de la topología.",
    },
    "generative": {
        "question": "¿El generador cubre la diversidad real o colapsa hacia pocas prendas?",
        "visual": """# Visualiza un lote real antes de entrenar la DCGAN\n# images, labels = next(iter(torch.utils.data.DataLoader(bundle.train, batch_size=16)))\n# print(images.shape, labels[:8])""",
        "analysis": "Sigue pérdidas, diversidad, MMD, vecinos reales más cercanos, interpolaciones latentes y señales de mode collapse.",
        "exercise": "Compara DCGAN y WGAN-GP con igual presupuesto de cómputo.",
    },
    "reinforcement": {
        "question": "¿La política mejora servicio y retorno sin incrementar excesivamente inventario?",
        "visual": """# Demanda histórica real usada por el entorno\n# demand = bundle.raw['train_demand']\n# print(demand[:20], demand.mean(), demand.max())""",
        "analysis": "Compara política periódica, DQN y Double Dueling DQN con varias semillas y episodios de evaluación sin exploración.",
        "exercise": "Añade prioritized replay y reporta retorno, stockouts, service level y estabilidad.",
    },
    "tabular": {
        "question": "¿La red supera una línea base simple después de controlar desbalance, calibración y fuga?",
        "visual": """# Inspección de atributos y distribución de clases\n# print(bundle.feature_names[:20])\n# print(bundle.summary)""",
        "analysis": "Compara contra regresión logística/árboles, revisa calibración, subgrupos, importancia de variables y estabilidad.",
        "exercise": "Ejecuta validación cruzada con split_seed fijo y al menos tres training_seeds.",
    },
    "multimodal": {
        "question": "¿Cada modalidad aporta información complementaria o una domina la predicción?",
        "visual": """# Revisa las formas de acelerómetro y giroscopio\n# sample, target = bundle.train[0]\n# print(sample.shape, target)""",
        "analysis": "Compara cada modalidad aislada, fusión temprana, fusión tardía y degradación cuando falta un sensor.",
        "exercise": "Implementa modality dropout y evalúa robustez ante la pérdida de un sensor.",
    },
}


def domain_for(lab: dict[str, Any]) -> str:
    architecture = lab["architecture"]
    if architecture in {"cnn", "transfer_resnet18", "distillation_cnn", "augmentation_comparison", "cnn_export", "regularization_comparison"}:
        return "vision"
    if architecture in {"rnn_text", "transformer_text"}:
        return "text"
    if architecture in {"lstm_regression"}:
        return "time_series"
    if architecture == "gcn":
        return "graph"
    if architecture == "dcgan":
        return "generative"
    if architecture == "dqn_inventory":
        return "reinforcement"
    if architecture == "sensor_fusion":
        return "multimodal"
    return "tabular"


def graded_metadata(grade_id: str, points: int, solution: bool) -> dict[str, Any]:
    return {
        "nbgrader": {
            "grade": False,
            "grade_id": grade_id,
            "locked": solution,
            "points": points,
            "schema_version": 3,
            "solution": True,
            "task": False,
        }
    }


def test_metadata(grade_id: str, points: int) -> dict[str, Any]:
    return {
        "nbgrader": {
            "grade": True,
            "grade_id": grade_id,
            "locked": True,
            "points": points,
            "schema_version": 3,
            "solution": False,
            "task": False,
        }
    }


def build_notebook(lab: dict[str, Any], *, solution: bool) -> nbformat.NotebookNode:
    domain = domain_for(lab)
    content = DOMAIN_CONTENT[domain]
    lab_id = lab["id"]
    cells = [
        new_markdown_cell(f"# {lab['title']}\n\n**Laboratorio:** `{lab_id}`  \n**Dataset real:** {lab['dataset']}  \n**Fuente:** {lab['source']}  \n**Versión:** {'solución' if solution else 'estudiante'}"),
        new_markdown_cell(f"## Objetivo\n\n{lab['objective']}\n\n**Pregunta guía:** {content['question']}"),
        new_markdown_cell("## Contrato científico\n\n1. `split_seed` define las particiones.  \n2. `training_seed` define inicialización y batches.  \n3. Toda selección ocurre con validation.  \n4. `experiment.lock.json` congela decisiones antes de abrir test.  \n5. Test se informa una sola vez."),
        new_code_cell("from pathlib import Path\nimport json\nimport yaml\nimport torch\n\nfrom neural_labs.catalog import ROOT, get_lab, get_dataset\nfrom neural_labs.datasets import prepare_dataset, audit_bundle\nfrom neural_labs.baselines import run_baseline\nfrom neural_labs.experiments import run_lab\n\nLAB_ID = '" + lab_id + "'\nSPLIT_SEED = 42\nTRAINING_SEED = 42\nQUICK = True"),
        new_code_cell("lab = get_lab(LAB_ID)\ndataset_card = get_dataset(LAB_ID)\nlab, dataset_card"),
        new_markdown_cell("## Procedencia, licencia y limitaciones\n\nLee la ficha antes de descargar. No redistribuyas datos si la licencia no lo permite."),
        new_code_cell("manifest_path = ROOT / 'labs' / LAB_ID / 'data' / 'dataset.yaml'\nmanifest = yaml.safe_load(manifest_path.read_text(encoding='utf-8'))\nmanifest"),
        new_markdown_cell("## Preparación y auditoría\n\nLa siguiente celda descarga datos reales. Puede requerir Internet, credenciales de Kaggle o aceptación de condiciones."),
        new_code_cell("# bundle = prepare_dataset(LAB_ID, quick=QUICK, seed=SPLIT_SEED)\n# audit_bundle(bundle)\nprint('Descomenta para descargar y preparar el dataset real.')"),
        new_markdown_cell(f"## Exploración específica del dominio: {domain}\n\n{content['analysis']}"),
        new_code_cell(content["visual"]),
        new_markdown_cell("## Línea base sobre validation\n\nLa línea base no debe mirar test antes de congelar el experimento."),
        new_code_cell("# baseline_validation = run_baseline(LAB_ID, bundle, quick=QUICK, evaluation_split='validation')\n# baseline_validation"),
        new_markdown_cell(f"## Modelo y teoría\n\n**Arquitectura:** `{lab['architecture']}`  \n**Fundamento:** {lab['math']}"),
        new_code_cell("# sample, target = bundle.train[0]\n# print('input:', sample.shape, 'target:', target)\n# print('selection metric:', lab['selection_metric'])"),
        new_markdown_cell("## Entrenamiento reproducible\n\nEl comando guarda contrato de inferencia, lock experimental, métricas, checkpoint y tarjetas."),
        new_code_cell("# result = run_lab(\n#     LAB_ID, quick=QUICK, config_name='baseline',\n#     split_seed=SPLIT_SEED, training_seed=TRAINING_SEED, device='auto',\n# )\n# result.run_dir, result.metrics"),
        new_markdown_cell("## Inspección de artefactos"),
        new_code_cell("# sorted(path.name for path in result.run_dir.iterdir())"),
        new_markdown_cell("## Ejercicio evaluable"),
        new_code_cell(
            ("# SOLUCIÓN DE REFERENCIA\nEXPERIMENT_DESCRIPTION = " + repr(content["exercise"]) + "\nEXPECTED_SEEDS = [41, 42, 43]\nassert len(EXPECTED_SEEDS) >= 3")
            if solution
            else ("# YOUR CODE HERE\nEXPERIMENT_DESCRIPTION = ''\nEXPECTED_SEEDS = []"),
            metadata=graded_metadata("experiment_design", 5, solution),
        ),
        new_code_cell(
            "assert isinstance(EXPERIMENT_DESCRIPTION, str) and len(EXPERIMENT_DESCRIPTION) >= 20\nassert len(EXPECTED_SEEDS) >= 3",
            metadata=test_metadata("experiment_design_test", 5),
        ),
        new_markdown_cell("## Interpretación y responsabilidad\n\nDocumenta resultados negativos, sesgos, grupos con peor desempeño, costo computacional y usos no recomendados."),
        new_code_cell(
            ("CONCLUSION = 'El modelo debe compararse con la línea base, varias semillas y subgrupos antes de recomendar su uso.'")
            if solution
            else "# YOUR CODE HERE\nCONCLUSION = ''",
            metadata=graded_metadata("conclusion", 5, solution),
        ),
        new_code_cell("assert len(CONCLUSION) >= 40", metadata=test_metadata("conclusion_test", 5)),
        new_markdown_cell("## Próximos pasos\n\nEjecuta primero con `--quick`; después usa el dataset completo, varias semillas y el pipeline de benchmark."),
    ]
    notebook = new_notebook(cells=cells)
    notebook.metadata.update(
        {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "neural_labs": {"lab_id": lab_id, "domain": domain, "variant": "solution" if solution else "student"},
        }
    )
    return notebook


def main() -> None:
    source_root = ROOT / "assignments" / "source"
    release_root = ROOT / "assignments" / "release"
    source_root.mkdir(parents=True, exist_ok=True)
    release_root.mkdir(parents=True, exist_ok=True)
    for lab in LABS:
        folder = ROOT / "labs" / lab["id"]
        solution = build_notebook(lab, solution=True)
        student = build_notebook(lab, solution=False)
        nbformat.write(solution, folder / "notebook_solution.ipynb")
        nbformat.write(solution, folder / "notebook.ipynb")
        nbformat.write(student, folder / "notebook_student.ipynb")
        assignment_source = source_root / lab["id"]
        assignment_release = release_root / lab["id"]
        assignment_source.mkdir(parents=True, exist_ok=True)
        assignment_release.mkdir(parents=True, exist_ok=True)
        shutil.copy2(folder / "notebook_solution.ipynb", assignment_source / "notebook.ipynb")
        shutil.copy2(folder / "notebook_student.ipynb", assignment_release / "notebook.ipynb")
    print(json.dumps({"labs": len(LABS), "notebooks": len(LABS) * 3}, indent=2))


if __name__ == "__main__":
    main()
