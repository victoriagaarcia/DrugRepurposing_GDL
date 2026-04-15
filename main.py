"""
=============================================================================
MAIN.PY - Punto de Entrada Principal para Drug Repurposing GNN
=============================================================================

Orquesta el pipeline completo:
1. Carga de datos
2. Entrenamiento
3. Evaluación
4. Ablación
5. Análisis de predicciones
=============================================================================
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from config import Config, ENCODER_TYPES, DECODER_TYPES, get_config
from data_loader import HetionetDataLoader
from train import train_model
from evaluate import LinkPredictionEvaluator, format_metrics
from ablation import AblationStudy
from models.full_model import create_model
from utils import (
    set_seed,
    setup_logging,
    get_device,
    Timer,
    compute_graph_statistics,
    print_graph_statistics,
    save_results_json,
    analyze_predictions,
    format_predictions_report,
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

ENCODER_CHOICES = sorted(set(ENCODER_TYPES + ["sage"]))
DECODER_CHOICES = sorted(set(DECODER_TYPES + ["dot"]))


def normalize_encoder_name(name: str) -> str:
    mapping = {
        "rgcn": "rgcn",
        "han": "han",
        "graphsage": "sage",
        "sage": "sage",
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Encoder no soportado: {name}")
    return mapping[key]


def normalize_decoder_name(name: str) -> str:
    mapping = {
        "distmult": "distmult",
        "dotproduct": "dot",
        "dot": "dot",
        "mlp": "mlp",
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Decoder no soportado: {name}")
    return mapping[key]


def apply_config_dict(config: Config, cfg_dict: Dict) -> Config:
    """Sobrescribe un Config con un dict serializado."""
    if not isinstance(cfg_dict, dict):
        return config

    for section_name in ["data", "model", "training", "evaluation", "ablation"]:
        section_values = cfg_dict.get(section_name)
        if not isinstance(section_values, dict):
            continue

        section_obj = getattr(config, section_name, None)
        if section_obj is None:
            continue

        for key, value in section_values.items():
            if hasattr(section_obj, key):
                setattr(section_obj, key, value)

    for key in ["experiment_name", "seed"]:
        if key in cfg_dict and hasattr(config, key):
            setattr(config, key, cfg_dict[key])

    return config


def build_config(
    use_synthetic: bool,
    seed: int,
    checkpoint_dir: Path,
) -> Config:
    """Construye una configuración coherente para el experimento."""
    config = get_config()
    device = get_device()

    config.seed = seed
    config.data.random_seed = seed
    config.training.device = device.type
    config.training.checkpoint_dir = str(checkpoint_dir)

    if use_synthetic:
        # Con el data_loader actual, poner la URL a None fuerza el fallback
        # al dataset sintético dentro del except.
        config.data.hetionet_url = None
        config.training.num_epochs = min(config.training.num_epochs, 50)
        config.training.patience = min(config.training.patience, 10)

    return config


def resolve_target_edge_type(
    data,
    default_target: Tuple[str, str, str],
) -> Tuple[str, str, str]:
    """
    Busca en el HeteroData el edge type real equivalente al target lógico.
    Prioriza mismo src/dst y misma relación; si no existe, usa el primero
    con mismo src/dst.
    """
    if default_target in data.edge_types:
        return default_target

    exact_src_dst = None
    for et in data.edge_types:
        if et[0] == default_target[0] and et[2] == default_target[2]:
            if et[1] == default_target[1]:
                return et
            if exact_src_dst is None:
                exact_src_dst = et

    if exact_src_dst is not None:
        return exact_src_dst

    if len(data.edge_types) == 0:
        raise ValueError("El grafo no contiene tipos de arista.")

    return list(data.edge_types)[0]


def sample_negative_edges(
    data,
    edge_type: Tuple[str, str, str],
    num_samples: int,
    device: torch.device,
) -> torch.Tensor:
    """Muestrea negativos sencillos para evaluación cuando no hay edge_label."""
    src_type, _, dst_type = edge_type
    num_src = data[src_type].num_nodes
    num_dst = data[dst_type].num_nodes

    existing_edges = set()
    if hasattr(data[edge_type], "edge_index"):
        for src, dst in data[edge_type].edge_index.t().cpu().tolist():
            existing_edges.add((src, dst))

    neg_src: List[int] = []
    neg_dst: List[int] = []

    while len(neg_src) < num_samples:
        batch = max(2 * (num_samples - len(neg_src)), 32)
        src_samples = torch.randint(0, num_src, (batch,))
        dst_samples = torch.randint(0, num_dst, (batch,))

        for src, dst in zip(src_samples.tolist(), dst_samples.tolist()):
            if (src, dst) not in existing_edges:
                neg_src.append(src)
                neg_dst.append(dst)
                if len(neg_src) >= num_samples:
                    break

    return torch.tensor(
        [neg_src[:num_samples], neg_dst[:num_samples]],
        dtype=torch.long,
        device=device,
    )


def get_eval_edges_and_labels(
    data,
    edge_type: Tuple[str, str, str],
    device: torch.device,
    negative_ratio: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extrae edge_label_index/edge_label o los construye si no existen."""
    if hasattr(data[edge_type], "edge_label_index") and hasattr(data[edge_type], "edge_label"):
        return data[edge_type].edge_label_index.to(device), data[edge_type].edge_label.to(device)

    if not hasattr(data[edge_type], "edge_index"):
        raise ValueError(f"No se encontraron edge_index ni edge_label_index para {edge_type}")

    pos_edge_index = data[edge_type].edge_index.to(device)
    num_pos = pos_edge_index.size(1)
    num_neg = max(num_pos * negative_ratio, 1)

    neg_edge_index = sample_negative_edges(data, edge_type, num_neg, device)

    edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    edge_label = torch.cat(
        [
            torch.ones(num_pos, device=device),
            torch.zeros(num_neg, device=device),
        ]
    )
    return edge_label_index, edge_label


def save_final_model(
    path: Path,
    model,
    config: Config,
    encoder_type: str,
    decoder_type: str,
    target_edge_type: Tuple[str, str, str],
    history: Dict,
    test_metrics: Dict,
) -> None:
    """Guarda un checkpoint simple y coherente para inferencia/análisis."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "encoder_type": encoder_type,
        "decoder_type": decoder_type,
        "target_edge_type": list(target_edge_type),
        "history": history,
        "test_metrics": test_metrics,
        "timestamp": datetime.now().isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


# -----------------------------------------------------------------------------
# Experimentos
# -----------------------------------------------------------------------------

def run_single_experiment(
    encoder_type: str = "rgcn",
    decoder_type: str = "distmult",
    use_synthetic: bool = False,
    seed: int = 42,
    output_dir: str = "results",
) -> Dict:
    """
    Ejecuta un experimento único completo:
    datos -> entrenamiento -> evaluación -> guardado.
    """
    set_seed(seed)

    encoder_cli = encoder_type
    decoder_cli = decoder_type
    encoder_type = normalize_encoder_name(encoder_type)
    decoder_type = normalize_decoder_name(decoder_type)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{encoder_cli}_{decoder_cli}_{timestamp}"
    exp_dir = Path(output_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(log_dir=str(exp_dir), log_file="experiment.log")
    device = get_device()

    logger.info(f"Iniciando experimento: {exp_name}")
    logger.info(f"Encoder CLI: {encoder_cli} -> interno: {encoder_type}")
    logger.info(f"Decoder CLI: {decoder_cli} -> interno: {decoder_type}")
    logger.info(f"Device: {device}")

    config = build_config(
        use_synthetic=use_synthetic,
        seed=seed,
        checkpoint_dir=exp_dir / "checkpoints",
    )

    logger.info("Cargando datos...")
    with Timer("Carga de datos", logger):
        data_loader = HetionetDataLoader(config)
        data, train_data, val_data, test_data = data_loader.load_data()

    stats = compute_graph_statistics(train_data)
    print_graph_statistics(stats)
    save_results_json(stats, str(exp_dir / "graph_statistics.json"))

    logger.info(
        f"Datos cargados: {stats['total_nodes']:,} nodos, {stats['total_edges']:,} aristas"
    )

    target_edge_type = resolve_target_edge_type(train_data, config.data.target_edge_type)
    logger.info(f"Target edge type resuelto: {target_edge_type}")

    logger.info("Iniciando entrenamiento...")
    with Timer("Entrenamiento", logger):
        trained_model, history = train_model(
            config=config,
            train_data=train_data,
            val_data=val_data,
            encoder_type=encoder_type,
            decoder_type=decoder_type,
            target_edge_type=target_edge_type,
        )

    logger.info("Evaluación en test set...")
    evaluator = LinkPredictionEvaluator(
        hits_k_values=config.evaluation.hits_k_values,
        filtered=config.evaluation.filtered,
    )

    edge_label_index, edge_label = get_eval_edges_and_labels(
        test_data,
        target_edge_type,
        device=torch.device(config.training.device),
        negative_ratio=max(1, config.training.negative_sampling_ratio),
    )

    with Timer("Evaluación", logger):
        test_metrics = evaluator.evaluate(
            model=trained_model,
            data=test_data,
            edge_label_index=edge_label_index,
            edge_label=edge_label,
            src_type=target_edge_type[0],
            dst_type=target_edge_type[2],
            batch_size=config.training.batch_size,
        )

    logger.info("\n" + "=" * 60)
    logger.info("RESULTADOS FINALES (TEST)")
    logger.info("=" * 60)
    logger.info("\n" + format_metrics(test_metrics))
    logger.info("=" * 60)

    results = {
        "experiment_name": exp_name,
        "encoder_type_cli": encoder_cli,
        "decoder_type_cli": decoder_cli,
        "encoder_type": encoder_type,
        "decoder_type": decoder_type,
        "seed": seed,
        "use_synthetic": use_synthetic,
        "target_edge_type": list(target_edge_type),
        "config": asdict(config),
        "graph_statistics": stats,
        "training_history": history,
        "test_metrics": test_metrics,
        "timestamp": timestamp,
    }

    save_results_json(results, str(exp_dir / "results.json"))
    save_final_model(
        path=exp_dir / "final_model.pt",
        model=trained_model,
        config=config,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        target_edge_type=target_edge_type,
        history=history,
        test_metrics=test_metrics,
    )

    logger.info(f"Resultados guardados en: {exp_dir}")
    return results


def run_quick_test(output_dir: str = "results/quick_test") -> Dict:
    """Ejecuta una prueba rápida con datos sintéticos."""
    logger = setup_logging()
    logger.info("Ejecutando test rápido con datos sintéticos...")

    results = run_single_experiment(
        encoder_type="rgcn",
        decoder_type="distmult",
        use_synthetic=True,
        seed=42,
        output_dir=output_dir,
    )

    logger.info("Test rápido completado.")
    logger.info(f"MRR: {results['test_metrics'].get('MRR', 0.0):.4f}")
    logger.info(f"Hits@10: {results['test_metrics'].get('Hits@10', 0.0):.4f}")
    return results


def run_ablation_study(
    seeds: Optional[List[int]] = None,
    use_synthetic: bool = False,
    output_dir: str = "results/ablation",
) -> Dict:
    """
    Ejecuta el estudio de ablación usando la API actual de AblationStudy.
    """
    seeds = seeds or [42, 43, 44]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(output_dir) / f"ablation_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(log_dir=str(exp_dir), log_file="ablation.log")
    logger.info("Iniciando estudio de ablación...")

    config = build_config(
        use_synthetic=use_synthetic,
        seed=seeds[0],
        checkpoint_dir=exp_dir / "checkpoints",
    )
    config.ablation.num_runs = len(seeds)

    if use_synthetic:
        config.training.num_epochs = min(config.training.num_epochs, 30)
        config.training.patience = min(config.training.patience, 5)

    study = AblationStudy(config)

    # AblationStudy actual espera "sage", no "graphsage".
    encoder_types = ["rgcn", "han", "sage"]

    with Timer("Estudio de ablación", logger):
        results = study.run_full_study(
            encoder_types=encoder_types,
            decoder_type="distmult",
        )

    analysis_text = study.analyze_results()

    analysis_path = exp_dir / "ablation_analysis.txt"
    analysis_path.write_text(analysis_text, encoding="utf-8")
    save_results_json(results, str(exp_dir / "ablation_results_copy.json"))

    logger.info("Análisis de ablación generado.")
    logger.info(f"Guardado en: {analysis_path}")

    return {
        "results": results,
        "analysis_path": str(analysis_path),
        "output_dir": str(exp_dir),
    }


def analyze_model_predictions(
    checkpoint_path: str,
    output_dir: str = "results/analysis",
) -> Dict:
    """
    Carga un modelo guardado y genera análisis de predicciones.
    """
    output_dir = str(Path(output_dir))
    logger = setup_logging(log_dir=output_dir, log_file="analysis.log")
    device = get_device()

    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = get_config()
    config = apply_config_dict(config, checkpoint.get("config", {}))
    config.training.device = device.type

    encoder_type = normalize_encoder_name(checkpoint.get("encoder_type", "rgcn"))
    decoder_type = normalize_decoder_name(checkpoint.get("decoder_type", "distmult"))

    logger.info(f"Analizando checkpoint: {checkpoint_path}")
    logger.info(f"Encoder: {encoder_type} | Decoder: {decoder_type}")

    data_loader = HetionetDataLoader(config)
    data, train_data, val_data, test_data = data_loader.load_data()

    model = create_model(
        data=data,
        config=config,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    target_edge_type = resolve_target_edge_type(data, config.data.target_edge_type)

    logger.info("Generando scores para todos los pares posibles...")
    with torch.no_grad():
        all_scores = model.predict_all_pairs(
            data.to(device),
            src_type=target_edge_type[0],
            dst_type=target_edge_type[2],
        ).detach().cpu()

    # Filtrar pares ya conocidos en el grafo completo.
    known_edges = set()
    if hasattr(data[target_edge_type], "edge_index"):
        for src, dst in data[target_edge_type].edge_index.t().cpu().tolist():
            known_edges.add((src, dst))

    src_type, _, dst_type = target_edge_type
    src_names = data_loader.idx_to_node.get(src_type, {})
    dst_names = data_loader.idx_to_node.get(dst_type, {})

    predictions = []
    num_src, num_dst = all_scores.shape
    for i in range(num_src):
        for j in range(num_dst):
            if (i, j) in known_edges:
                continue
            src_id = src_names.get(i, f"{src_type}_{i}")
            dst_id = dst_names.get(j, f"{dst_type}_{j}")
            predictions.append((src_id, dst_id, float(all_scores[i, j].item())))

    analysis = analyze_predictions(predictions, top_k=50)
    report = format_predictions_report(analysis)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions_report.txt").write_text(report, encoding="utf-8")

    save_results_json(
        {
            "checkpoint_path": checkpoint_path,
            "encoder_type": encoder_type,
            "decoder_type": decoder_type,
            "target_edge_type": list(target_edge_type),
            "top_predictions": analysis["top_predictions"][:100],
            "score_distribution": analysis["score_distribution"],
            "num_predictions": analysis["num_total_predictions"],
        },
        str(out_dir / "predictions_analysis.json"),
    )

    logger.info(f"Análisis guardado en: {out_dir}")
    print(report)
    return analysis


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Drug Repurposing mediante Graph Neural Networks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python main.py --mode quick
  python main.py --mode single --encoder rgcn --decoder distmult
  python main.py --mode single --encoder graphsage --decoder mlp --synthetic
  python main.py --mode ablation --seeds 42 43 44
  python main.py --mode analyze --checkpoint results/exp/final_model.pt
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["quick", "single", "ablation", "analyze"],
        default="quick",
        help="Modo de ejecución",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        choices=ENCODER_CHOICES,
        default="rgcn",
        help="Tipo de encoder",
    )
    parser.add_argument(
        "--decoder",
        type=str,
        choices=DECODER_CHOICES,
        default="distmult",
        help="Tipo de decoder",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Usar dataset sintético",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
        help="Semillas para ablación",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Ruta a checkpoint para modo analyze",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directorio de salida",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print(" DRUG REPURPOSING mediante GRAPH NEURAL NETWORKS")
    print(" Proyecto de Geometric Deep Learning")
    print("=" * 60 + "\n")

    if args.mode == "quick":
        logger = setup_logging()
        logger.info("Modo: Test rápido")
        run_quick_test(output_dir=str(Path(args.output_dir) / "quick_test"))

    elif args.mode == "single":
        logger = setup_logging()
        logger.info("Modo: Experimento único")
        run_single_experiment(
            encoder_type=args.encoder,
            decoder_type=args.decoder,
            use_synthetic=args.synthetic,
            seed=args.seed,
            output_dir=args.output_dir,
        )

    elif args.mode == "ablation":
        logger = setup_logging()
        logger.info("Modo: Estudio de ablación")
        run_ablation_study(
            seeds=args.seeds,
            use_synthetic=args.synthetic,
            output_dir=str(Path(args.output_dir) / "ablation"),
        )

    elif args.mode == "analyze":
        logger = setup_logging()
        logger.info("Modo: Análisis de predicciones")
        if not args.checkpoint:
            raise ValueError("Debes proporcionar --checkpoint en modo analyze")
        analyze_model_predictions(
            checkpoint_path=args.checkpoint,
            output_dir=str(Path(args.output_dir) / "analysis"),
        )

    else:
        raise ValueError(f"Modo desconocido: {args.mode}")


if __name__ == "__main__":
    main()