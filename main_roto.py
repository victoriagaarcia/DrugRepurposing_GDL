"""
=============================================================================
MAIN.PY - Punto de Entrada Principal para Drug Repurposing GNN
=============================================================================

Este script orquesta todo el pipeline de Drug Repurposing mediante GNNs:

1. CARGA DE DATOS
   - Descarga Hetionet o crea dataset sintético
   - Preprocesa en formato PyG HeteroData
   - Divide en train/val/test

2. ENTRENAMIENTO
   - Configura modelo (encoder + decoder)
   - Entrena con negative sampling
   - Guarda checkpoints

3. EVALUACIÓN
   - Métricas de ranking (MRR, Hits@K)
   - Métricas de clasificación (AUC-ROC)

4. ESTUDIO DE ABLACIÓN
   - Compara arquitecturas (R-GCN, HAN, GraphSAGE)
   - Evalúa importancia de entidades intermedias

5. ANÁLISIS DE RESULTADOS
   - Genera reportes
   - Visualiza predicciones

USO:
----
    # Modo rápido (datos sintéticos)
    python main.py --mode quick
    
    # Experimento único
    python main.py --mode single --encoder rgcn --decoder distmult
    
    # Estudio de ablación completo
    python main.py --mode ablation
    
    # Análisis de predicciones
    python main.py --mode analyze --checkpoint path/to/model.pt

=============================================================================
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch

# Importar módulos del proyecto
from config import (
    Config, DataConfig, ModelConfig, TrainingConfig, 
    EvaluationConfig, AblationConfig, 
    ENCODER_TYPES, DECODER_TYPES
)
from data_loader import HetionetDataLoader
from models import create_model, LinkPredictionLoss
from DrugRepurposing_GDL.train_roto import Trainer, train_model
from evaluate import LinkPredictionEvaluator
from ablation import AblationStudy
from utils import (
    set_seed, setup_logging, get_device, Timer,
    save_checkpoint, load_checkpoint,
    plot_training_curves, plot_ablation_results,
    compute_graph_statistics, print_graph_statistics,
    save_results_json, analyze_predictions, format_predictions_report
)


# =============================================================================
# FUNCIONES PRINCIPALES
# =============================================================================

def run_single_experiment(
    encoder_type: str = 'rgcn',
    decoder_type: str = 'distmult',
    use_synthetic: bool = False,
    seed: int = 42,
    output_dir: str = 'results'
) -> dict:
    """
    Ejecuta un experimento único con configuración específica.
    
    PIPELINE:
    ---------
    1. Configurar reproducibilidad
    2. Cargar y preprocesar datos
    3. Crear modelo (encoder + decoder)
    4. Entrenar con early stopping
    5. Evaluar en test set
    6. Guardar resultados
    
    Args:
        encoder_type: Tipo de encoder ('rgcn', 'han', 'graphsage')
        decoder_type: Tipo de decoder ('distmult', 'dotproduct', 'mlp')
        use_synthetic: Si True, usa datos sintéticos (más rápido para debugging)
        seed: Semilla para reproducibilidad
        output_dir: Directorio para guardar resultados
        
    Returns:
        Diccionario con métricas y resultados
    """
    # -------------------------------------------------------------------------
    # SETUP
    # -------------------------------------------------------------------------
    set_seed(seed)
    device = get_device()
    
    # Crear directorio de salida
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{encoder_type}_{decoder_type}_{timestamp}"
    exp_dir = Path(output_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Configurar logging
    logger = setup_logging(
        log_dir=str(exp_dir),
        log_file='experiment.log'
    )
    
    logger.info(f"Iniciando experimento: {exp_name}")
    logger.info(f"Encoder: {encoder_type}, Decoder: {decoder_type}")
    logger.info(f"Device: {device}")
    
    # -------------------------------------------------------------------------
    # CONFIGURACIÓN
    # -------------------------------------------------------------------------
    config = Config()
    data_config = DataConfig()
    model_config = ModelConfig()
    training_config = TrainingConfig()
    eval_config = EvaluationConfig()
    
    # Ajustar para datos sintéticos (más pequeños)
    if use_synthetic:
        config.training.epochs = 50
        config.training.patience = 10
        config.data.hetionet_url = None  # Forzar sintético
        # training_config.epochs = 50
        # training_config.patience = 10
        # data_config.hetionet_url = None  # Forzar sintético
    
    # -------------------------------------------------------------------------
    # CARGA DE DATOS
    # -------------------------------------------------------------------------
    logger.info("Cargando datos...")
    
    with Timer("Carga de datos", logger):
        # data_loader = HetionetDataLoader(data_config)
        data_loader = HetionetDataLoader(config)
        # train_data, val_data, test_data = data_loader.prepare_data()
        data, train_data, val_data, test_data = data_loader.load_data()
    
    # Estadísticas del grafo
    stats = compute_graph_statistics(train_data)
    print_graph_statistics(stats)
    save_results_json(stats, str(exp_dir / 'graph_statistics.json'))
    
    # Mover a dispositivo
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)
    
    logger.info(f"Datos cargados: {stats['total_nodes']:,} nodos, {stats['total_edges']:,} aristas")
    
    # -------------------------------------------------------------------------
    # CREAR MODELO
    # -------------------------------------------------------------------------
    logger.info(f"Creando modelo {encoder_type} + {decoder_type}...")
    
    model = create_model(
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        data=train_data,
        # model_config=model_config
        config = config
    )
    model = model.to(device)

    # Contar parámetros
    # total_params = sum(p.numel() for p in model.parameters())
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # logger.info(f"Parámetros totales: {total_params:,}")
    # logger.info(f"Parámetros entrenables: {trainable_params:,}")
    
    # -------------------------------------------------------------------------
    # ENTRENAMIENTO
    # -------------------------------------------------------------------------
    logger.info("Iniciando entrenamiento...")
    
    with Timer("Entrenamiento", logger):
        trained_model, history = train_model(
            config = config,
            train_data=train_data,
            val_data=val_data,
            encoder_type=encoder_type,
            decoder_type=decoder_type,
            # training_config=training_config,
            # eval_config=eval_config,
            # device=device,
            # checkpoint_dir=str(exp_dir / 'checkpoints')
        )
    
    # Guardar curvas de entrenamiento
    if history.get('train_losses') and history.get('val_losses'):
        plot_training_curves(
            train_losses=history['train_losses'],
            val_losses=history['val_losses'],
            val_metrics={'MRR': history.get('val_mrr', [])},
            save_path=str(exp_dir / 'training_curves.png'),
            title=f'Training Progress - {encoder_type} + {decoder_type}'
        )
    
    # -------------------------------------------------------------------------
    # EVALUACIÓN FINAL
    # -------------------------------------------------------------------------
    logger.info("Evaluación en test set...")
    
    evaluator = LinkPredictionEvaluator(eval_config)
    
    with Timer("Evaluación", logger):
        test_metrics = evaluator.evaluate(
            model=trained_model,
            data=test_data,
            edge_type=data_config.target_edge_type
        )
    
    # Mostrar resultados
    logger.info("\n" + "=" * 50)
    logger.info("RESULTADOS FINALES (TEST SET)")
    logger.info("=" * 50)
    for metric, value in test_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    logger.info("=" * 50)
    
    # -------------------------------------------------------------------------
    # GUARDAR RESULTADOS
    # -------------------------------------------------------------------------
    results = {
        'experiment_name': exp_name,
        'encoder_type': encoder_type,
        'decoder_type': decoder_type,
        'seed': seed,
        'config': {
            'model': model_config.__dict__,
            'training': training_config.__dict__,
        },
        'graph_statistics': stats,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'training_history': history,
        'test_metrics': test_metrics,
        'timestamp': timestamp
    }
    
    save_results_json(results, str(exp_dir / 'results.json'))
    
    # Guardar modelo final
    save_checkpoint(
        model=trained_model,
        optimizer=None,  # No necesario para inferencia
        epoch=history.get('best_epoch', -1),
        metrics=test_metrics,
        path=str(exp_dir / 'final_model.pt'),
        config=results['config']
    )
    
    logger.info(f"Resultados guardados en: {exp_dir}")
    
    return results


def run_ablation_study(
    seeds: list = [42, 123, 456],
    use_synthetic: bool = False,
    output_dir: str = 'results/ablation'
) -> dict:
    """
    Ejecuta el estudio de ablación completo.
    
    DISEÑO DEL ESTUDIO:
    -------------------
    El estudio evalúa dos dimensiones:
    
    1. ARQUITECTURAS GNN (encoder):
       - R-GCN: Una convolución por tipo de relación
       - HAN: Attention heterogéneo
       - GraphSAGE: Sampling + aggregation inductivo
    
    2. CONTRIBUCIÓN DE ENTIDADES INTERMEDIAS (ablación):
       - full: Grafo completo (Compound, Disease, Gene, Anatomy)
       - no_anatomy: Sin nodos de anatomía
       - no_gene: Sin nodos de genes
       - no_intermediate: Solo Compound-Disease (baseline)
    
    La hipótesis central es que los genes son cruciales porque
    los fármacos actúan a través de vecindarios de proteínas,
    validando el framework de network medicine de Barabási.
    
    Args:
        seeds: Lista de semillas para múltiples runs
        use_synthetic: Si True, usa datos sintéticos
        output_dir: Directorio de salida
        
    Returns:
        Diccionario con todos los resultados del estudio
    """
    logger = logging.getLogger('DrugRepurposingGNN')
    logger.info("Iniciando estudio de ablación...")
    
    # Crear estudio
    ablation_config = AblationConfig()
    
    # Configuraciones
    data_config = DataConfig()
    model_config = ModelConfig()
    training_config = TrainingConfig()
    eval_config = EvaluationConfig()
    
    # Ajustar para datos sintéticos
    if use_synthetic:
        training_config.epochs = 30
        training_config.patience = 5
    
    # Ejecutar estudio
    study = AblationStudy(
        data_config=data_config,
        model_config=model_config,
        training_config=training_config,
        eval_config=eval_config,
        ablation_config=ablation_config,
        output_dir=output_dir
    )
    
    with Timer("Estudio de ablación completo", logger):
        results = study.run_full_study(
            seeds=seeds,
            encoder_types=['rgcn', 'han', 'graphsage']
        )
    
    # Analizar y visualizar
    analysis = study.analyze_results(results)
    
    # Visualizar resultados
    for metric in ['MRR', 'Hits@10']:
        plot_ablation_results(
            results={k: v['mean'] for k, v in analysis.items()},
            metric=metric,
            save_path=str(Path(output_dir) / f'ablation_{metric}.png')
        )
    
    logger.info("Estudio de ablación completado")
    
    return results


def run_quick_test(output_dir: str = 'results/quick_test'):
    """
    Ejecuta un test rápido con datos sintéticos para verificar el pipeline.
    
    Útil para:
    - Verificar que todo está instalado correctamente
    - Debug rápido del código
    - Entender el flujo del pipeline
    """
    logger = logging.getLogger('DrugRepurposingGNN')
    logger.info("Ejecutando test rápido con datos sintéticos...")
    
    # Ejecutar un experimento simple con datos sintéticos
    results = run_single_experiment(
        encoder_type='rgcn',
        decoder_type='distmult',
        use_synthetic=True,
        seed=42,
        output_dir=output_dir
    )
    
    logger.info("\nTest rápido completado!")
    logger.info(f"MRR: {results['test_metrics'].get('MRR', 'N/A'):.4f}")
    logger.info(f"Hits@10: {results['test_metrics'].get('Hits@10', 'N/A'):.4f}")
    
    return results


def analyze_model_predictions(
    checkpoint_path: str,
    output_dir: str = 'results/analysis'
):
    """
    Analiza las predicciones de un modelo entrenado.
    
    ANÁLISIS INCLUIDO:
    ------------------
    1. Top-K predicciones nuevas
    2. Distribución de scores
    3. Predicciones por enfermedad
    4. Predicciones por fármaco
    5. Validación retrospectiva (si hay ground truth)
    
    Args:
        checkpoint_path: Ruta al checkpoint del modelo
        output_dir: Directorio para guardar análisis
    """
    logger = logging.getLogger('DrugRepurposingGNN')
    logger.info(f"Analizando predicciones del modelo: {checkpoint_path}")
    
    device = get_device()
    
    # Cargar datos
    data_config = DataConfig()
    data_loader = HetionetDataLoader(data_config)
    train_data, val_data, test_data = data_loader.prepare_data()
    
    # Reconstruir modelo (necesitamos la misma arquitectura)
    # Cargar checkpoint para obtener config
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})
    
    # Crear modelo con la configuración guardada
    model_config = ModelConfig()
    if 'model' in config:
        for key, value in config['model'].items():
            if hasattr(model_config, key):
                setattr(model_config, key, value)
    
    # Inferir tipo de encoder/decoder del nombre del archivo si no está en config
    encoder_type = 'rgcn'  # default
    decoder_type = 'distmult'  # default
    
    model = create_model(
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        data=train_data,
        model_config=model_config
    )
    
    # Cargar pesos
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Generar predicciones para todos los pares posibles
    logger.info("Generando predicciones...")
    
    test_data = test_data.to(device)
    
    with torch.no_grad():
        # Obtener embeddings
        embeddings = model.encode(test_data)
        
        # Predecir todos los pares Compound-Disease
        compound_emb = embeddings['Compound']
        disease_emb = embeddings['Disease']
        
        # Scores para todos los pares
        all_scores = model.predict_all_pairs(
            compound_emb, disease_emb, 
            data_config.target_edge_type
        )
    
    # Crear lista de predicciones
    predictions = []
    n_compounds = compound_emb.size(0)
    n_diseases = disease_emb.size(0)
    
    for i in range(n_compounds):
        for j in range(n_diseases):
            score = all_scores[i, j].item()
            predictions.append((f"Compound_{i}", f"Disease_{j}", score))
    
    # Filtrar predicciones conocidas (edges en train/val)
    # (En una implementación real, usaríamos los IDs reales)
    
    # Analizar predicciones
    analysis = analyze_predictions(predictions, top_k=50)
    
    # Generar reporte
    report = format_predictions_report(analysis)
    print(report)
    
    # Guardar
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    with open(Path(output_dir) / 'predictions_report.txt', 'w') as f:
        f.write(report)
    
    save_results_json(
        {
            'top_predictions': analysis['top_predictions'][:100],
            'score_distribution': analysis['score_distribution'],
            'num_predictions': analysis['num_total_predictions']
        },
        str(Path(output_dir) / 'predictions_analysis.json')
    )
    
    logger.info(f"Análisis guardado en: {output_dir}")
    
    return analysis


# =============================================================================
# INTERFAZ DE LÍNEA DE COMANDOS
# =============================================================================

def parse_args():
    """Parser de argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description='Drug Repurposing mediante Graph Neural Networks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
----------------
  # Test rápido con datos sintéticos
  python main.py --mode quick
  
  # Experimento único con R-GCN + DistMult
  python main.py --mode single --encoder rgcn --decoder distmult
  
  # Estudio de ablación completo
  python main.py --mode ablation --seeds 42 123 456
  
  # Análisis de predicciones de un modelo guardado
  python main.py --mode analyze --checkpoint results/exp/final_model.pt
        """
    )
    
    # Modo de ejecución
    parser.add_argument(
        '--mode',
        type=str,
        choices=['quick', 'single', 'ablation', 'analyze'],
        default='quick',
        help='Modo de ejecución (default: quick)'
    )
    
    # Configuración del modelo
    parser.add_argument(
        '--encoder',
        type=str,
        choices=ENCODER_TYPES,
        default='rgcn',
        help='Tipo de encoder GNN (default: rgcn)'
    )
    
    parser.add_argument(
        '--decoder',
        type=str,
        choices=DECODER_TYPES,
        default='distmult',
        help='Tipo de decoder (default: distmult)'
    )
    
    # Datos
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Usar datos sintéticos (más rápido para testing)'
    )
    
    # Reproducibilidad
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Semilla para reproducibilidad (default: 42)'
    )
    
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=[42, 123, 456],
        help='Semillas para estudio de ablación (default: 42 123 456)'
    )
    
    # Paths
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directorio de salida (default: results)'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path al checkpoint para modo analyze'
    )
    
    # Logging
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Mostrar logs de debug'
    )
    
    return parser.parse_args()


def main():
    """Función principal."""
    args = parse_args()
    
    # Configurar logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level=log_level)
    
    logger = logging.getLogger('DrugRepurposingGNN')
    
    # Banner
    print("\n" + "=" * 60)
    print(" DRUG REPURPOSING mediante GRAPH NEURAL NETWORKS")
    print(" Proyecto de Geometric Deep Learning")
    print("=" * 60 + "\n")
    
    # Ejecutar según modo
    if args.mode == 'quick':
        logger.info("Modo: Test rápido")
        results = run_quick_test(
            output_dir=str(Path(args.output_dir) / 'quick_test')
        )
        
    elif args.mode == 'single':
        logger.info("Modo: Experimento único")
        results = run_single_experiment(
            encoder_type=args.encoder,
            decoder_type=args.decoder,
            use_synthetic=args.synthetic,
            seed=args.seed,
            output_dir=args.output_dir
        )
        
    elif args.mode == 'ablation':
        logger.info("Modo: Estudio de ablación")
        results = run_ablation_study(
            seeds=args.seeds,
            use_synthetic=args.synthetic,
            output_dir=str(Path(args.output_dir) / 'ablation')
        )
        
    elif args.mode == 'analyze':
        logger.info("Modo: Análisis de predicciones")
        if not args.checkpoint:
            logger.error("Se requiere --checkpoint para modo analyze")
            sys.exit(1)
        results = analyze_model_predictions(
            checkpoint_path=args.checkpoint,
            output_dir=str(Path(args.output_dir) / 'analysis')
        )
    
    print("\n" + "=" * 60)
    print(" Ejecución completada!")
    print("=" * 60 + "\n")
    
    return results


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    main()
