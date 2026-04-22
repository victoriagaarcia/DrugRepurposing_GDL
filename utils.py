"""
=============================================================================
UTILS.PY - Utilidades Generales para Drug Repurposing GNN
=============================================================================

Este módulo contiene funciones de utilidad que son usadas a través de todo
el proyecto:

1. LOGGING Y REPRODUCIBILIDAD
   - Configuración de seeds para reproducibilidad
   - Sistema de logging configurable
   - Métricas de tiempo de ejecución

2. MANEJO DE DISPOSITIVOS
   - Detección automática de GPU/CPU
   - Movimiento de datos entre dispositivos

3. CHECKPOINTING
   - Guardar y cargar modelos
   - Guardar estados de entrenamiento completos

4. VISUALIZACIÓN
   - Curvas de aprendizaje
   - Distribución de embeddings
   - Visualización de grafos

5. ANÁLISIS DE RESULTADOS
   - Análisis estadístico de predicciones
   - Validación retrospectiva

BASE TEÓRICA:
-------------
La reproducibilidad es fundamental en machine learning científico. 
Según Pineau et al. (2021) "A Checklist for Responsible ML Research",
un experimento reproducible debe controlar:
- Seeds de generadores aleatorios (Python, NumPy, PyTorch)
- Orden de operaciones en GPU (determinismo)
- Versiones de librerías
- Hiperparámetros exactos

El proyecto sigue estas prácticas para garantizar que los resultados
del estudio comparativo sean replicables.

=============================================================================
"""

import os
import json
import random
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

# Imports opcionales para visualización
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


# =============================================================================
# LOGGING Y REPRODUCIBILIDAD
# =============================================================================

def set_seed(seed: int = 42) -> None:
    """
    Establece seeds para reproducibilidad completa.
    
    TEORÍA:
    -------
    Los generadores de números pseudoaleatorios (PRNGs) son fundamentales
    en ML para:
    - Inicialización de pesos (Xavier, Kaiming)
    - Dropout y otras regularizaciones
    - Shuffling de datos
    - Negative sampling
    
    Para reproducibilidad completa, debemos sincronizar:
    1. random: Módulo estándar de Python
    2. numpy: Operaciones numéricas
    3. torch: Operaciones de PyTorch en CPU
    4. torch.cuda: Operaciones en GPU
    
    NOTA: El determinismo completo en GPU puede afectar el rendimiento.
    torch.backends.cudnn.benchmark=False desactiva la auto-tuning de cuDNN.
    
    Args:
        seed: Semilla para todos los generadores
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Para multi-GPU
        
        # Determinismo completo (puede reducir rendimiento)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Para operaciones determinísticas en PyTorch 1.8+
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logging.info(f"Seeds establecidas: {seed}")


def setup_logging(
    log_dir: Optional[str] = None,
    log_level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Configura el sistema de logging.
    
    Args:
        log_dir: Directorio para archivos de log
        log_level: Nivel de logging (DEBUG, INFO, WARNING, ERROR)
        log_file: Nombre del archivo de log (opcional)
        
    Returns:
        Logger configurado
    """
    # Crear directorio si no existe
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Formato del log
    log_format = '%(asctime)s | %(levelname)8s | %(name)s | %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Configurar handlers
    handlers = [logging.StreamHandler()]  # Console
    
    if log_file and log_dir:
        log_path = Path(log_dir) / log_file
        handlers.append(logging.FileHandler(log_path))
    
    # Configurar logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force=True  # Sobrescribe configuraciones previas
    )
    
    logger = logging.getLogger('DrugRepurposingGNN')
    logger.setLevel(log_level)
    
    return logger


class Timer:
    """
    Context manager para medir tiempo de ejecución.
    
    Uso:
        with Timer("Entrenamiento"):
            train_model(...)
        # Imprime: "Entrenamiento: 123.45s"
    """
    
    def __init__(self, name: str = "Operation", logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or logging.getLogger('DrugRepurposingGNN')
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        self.logger.info(f"{self.name}: {elapsed:.2f}s")
    
    @property
    def elapsed(self) -> float:
        """Tiempo transcurrido en segundos."""
        if self.end_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return time.time() - self.start_time
        return 0.0


# =============================================================================
# MANEJO DE DISPOSITIVOS
# =============================================================================

def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Detecta y retorna el dispositivo óptimo para computación.
    
    TEORÍA:
    -------
    Las GPUs aceleran el entrenamiento de GNNs significativamente debido a:
    - Paralelismo masivo para multiplicación de matrices
    - Memoria de alto ancho de banda (HBM)
    - Operaciones optimizadas en cuDNN
    
    Para grafos grandes, la memoria GPU puede ser limitante.
    PyG soporta mini-batching para grafos que no caben en memoria.
    
    Args:
        prefer_gpu: Si True, usa GPU cuando esté disponible
        
    Returns:
        torch.device para CPU o CUDA
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        
        # Información de la GPU
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        logging.info(f"Usando GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        device = torch.device('cpu')
        logging.info("Usando CPU")
    
    return device


def move_to_device(data: Any, device: torch.device) -> Any:
    """
    Mueve datos (tensores, modelos, HeteroData) al dispositivo especificado.
    
    Args:
        data: Datos a mover (tensor, modelo, HeteroData, o diccionario)
        device: Dispositivo destino
        
    Returns:
        Datos en el dispositivo especificado
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, nn.Module):
        return data.to(device)
    elif hasattr(data, 'to'):  # HeteroData, Data, etc.
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(move_to_device(item, device) for item in data)
    else:
        return data


def get_memory_usage() -> Dict[str, float]:
    """
    Retorna el uso actual de memoria (CPU y GPU si está disponible).
    
    Returns:
        Diccionario con uso de memoria en GB
    """
    import psutil
    
    memory_info = {
        'cpu_used_gb': psutil.Process().memory_info().rss / 1e9,
        'cpu_available_gb': psutil.virtual_memory().available / 1e9
    }
    
    if torch.cuda.is_available():
        memory_info['gpu_allocated_gb'] = torch.cuda.memory_allocated() / 1e9
        memory_info['gpu_cached_gb'] = torch.cuda.memory_reserved() / 1e9
        memory_info['gpu_total_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    return memory_info


# =============================================================================
# CHECKPOINTING
# =============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    metrics: Dict[str, float],
    path: str,
    scheduler: Optional[Any] = None,
    config: Optional[Dict] = None
) -> None:
    """
    Guarda un checkpoint completo del entrenamiento.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    # Guardar solo config serializable
    if config is not None:
        if isinstance(config, dict):
            checkpoint["config"] = config
        else:
            try:
                from dataclasses import asdict, is_dataclass
                if is_dataclass(config):
                    checkpoint["config"] = asdict(config)
                else:
                    checkpoint["config"] = str(config)
            except Exception:
                checkpoint["config"] = str(config)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    logging.info(f"Checkpoint guardado: {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Carga un checkpoint y restaura el estado.
    """
    if device is None:
        device = get_device()

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    logging.info(f"Checkpoint cargado: {path} (época {checkpoint['epoch']})")

    return {
        "epoch": checkpoint["epoch"],
        "metrics": checkpoint.get("metrics", {}),
        "config": checkpoint.get("config", {}),
        "timestamp": checkpoint.get("timestamp"),
    }


# =============================================================================
# VISUALIZACIÓN
# =============================================================================

def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    val_metrics: Optional[Dict[str, List[float]]] = None,
    save_path: Optional[str] = None,
    title: str = "Training Progress"
) -> None:
    """
    Visualiza las curvas de entrenamiento.
    
    TEORÍA:
    -------
    Las curvas de aprendizaje revelan:
    - Convergencia: ¿El modelo está aprendiendo?
    - Overfitting: Gap creciente entre train y val loss
    - Underfitting: Loss alta que no baja
    - Inestabilidad: Oscilaciones grandes
    
    Para link prediction, además monitoreamos:
    - MRR: Mean Reciprocal Rank (debe subir)
    - Hits@K: Proporción de aciertos en top-K (debe subir)
    
    Args:
        train_losses: Lista de pérdidas de entrenamiento
        val_losses: Lista de pérdidas de validación
        val_metrics: Diccionario con métricas adicionales
        save_path: Ruta para guardar la figura
        title: Título del gráfico
    """
    if not HAS_PLOTTING:
        logging.warning("matplotlib/seaborn no disponibles para visualización")
        return
    
    n_epochs = len(train_losses)
    epochs = range(1, n_epochs + 1)
    
    # Configurar estilo
    sns.set_style("whitegrid")
    
    # Número de subplots
    n_metrics = len(val_metrics) if val_metrics else 0
    n_plots = 1 + n_metrics
    
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    
    if n_plots == 1:
        axes = [axes]
    
    # Plot 1: Pérdidas
    axes[0].plot(epochs, train_losses, label='Train Loss', color='blue', alpha=0.8)
    axes[0].plot(epochs, val_losses, label='Val Loss', color='orange', alpha=0.8)
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Curvas de Pérdida')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plots adicionales para métricas
    if val_metrics:
        colors = ['green', 'red', 'purple', 'brown']
        for idx, (metric_name, values) in enumerate(val_metrics.items()):
            ax = axes[idx + 1]
            color = colors[idx % len(colors)]
            ax.plot(epochs[:len(values)], values, label=metric_name, color=color, alpha=0.8)
            ax.set_xlabel('Época')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} vs Época')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"Figura guardada: {save_path}")
    
    plt.show()


def plot_embedding_distribution(
    embeddings: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    method: str = 'tsne',
    save_path: Optional[str] = None,
    title: str = "Embedding Distribution"
) -> None:
    """
    Visualiza la distribución de embeddings en 2D.
    
    TEORÍA:
    -------
    La visualización de embeddings ayuda a entender:
    - Separabilidad: ¿Se agrupan nodos similares?
    - Estructura: ¿Hay clusters claros?
    - Calidad: ¿Los embeddings capturan relaciones semánticas?
    
    Métodos de reducción de dimensionalidad:
    - t-SNE: Preserva estructura local, bueno para clusters
    - UMAP: Preserva estructura global y local
    - PCA: Rápido pero puede perder estructura no lineal
    
    Args:
        embeddings: Tensor de embeddings [N, D]
        labels: Etiquetas opcionales para colorear
        method: 'tsne', 'umap', o 'pca'
        save_path: Ruta para guardar
        title: Título del gráfico
    """
    if not HAS_PLOTTING:
        logging.warning("matplotlib no disponible")
        return
    
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    
    # Convertir a numpy
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    
    # Reducir dimensionalidad
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    else:
        raise ValueError(f"Método no soportado: {method}")
    
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Visualizar
    plt.figure(figsize=(10, 8))
    
    if labels is not None:
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        scatter = plt.scatter(
            embeddings_2d[:, 0], 
            embeddings_2d[:, 1], 
            c=labels, 
            cmap='tab10',
            alpha=0.6,
            s=10
        )
        plt.colorbar(scatter, label='Tipo de nodo')
    else:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=10)
    
    plt.xlabel(f'{method.upper()} 1')
    plt.ylabel(f'{method.upper()} 2')
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_ablation_results(
    results: Dict[str, Dict[str, float]],
    metric: str = 'MRR',
    save_path: Optional[str] = None
) -> None:
    """
    Visualiza resultados del estudio de ablación.
    
    Args:
        results: Diccionario {config_name: {metric: value}}
        metric: Métrica a visualizar
        save_path: Ruta para guardar
    """
    if not HAS_PLOTTING:
        return
    
    configs = list(results.keys())
    values = [results[c].get(metric, 0) for c in configs]
    
    # Calcular cambios respecto al modelo completo
    full_value = results.get('full', {}).get(metric, values[0])
    changes = [(v - full_value) / full_value * 100 if full_value != 0 else 0 for v in values]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Valores absolutos
    colors = ['green' if c == 'full' else 'steelblue' for c in configs]
    bars1 = ax1.bar(configs, values, color=colors, alpha=0.8)
    ax1.set_ylabel(metric)
    ax1.set_title(f'{metric} por Configuración')
    ax1.set_xticklabels(configs, rotation=45, ha='right')
    
    # Añadir valores encima de las barras
    for bar, val in zip(bars1, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Cambios porcentuales
    colors2 = ['green' if c >= 0 else 'red' for c in changes]
    bars2 = ax2.bar(configs, changes, color=colors2, alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('Cambio (%)')
    ax2.set_title(f'Cambio en {metric} vs Modelo Completo')
    ax2.set_xticklabels(configs, rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


# =============================================================================
# ANÁLISIS DE PREDICCIONES
# =============================================================================

def analyze_predictions(
    predictions: List[Tuple[str, str, float]],
    ground_truth: Optional[List[Tuple[str, str]]] = None,
    top_k: int = 20
) -> Dict[str, Any]:
    """
    Analiza las predicciones del modelo.
    
    TEORÍA:
    -------
    Para drug repurposing, el análisis de predicciones incluye:
    1. Top-K predictions: ¿Qué pares fármaco-enfermedad predicen?
    2. Validación retrospectiva: ¿Alguno ya ha sido probado?
    3. Análisis por enfermedad: ¿Qué fármacos se predicen para cada enfermedad?
    4. Análisis por fármaco: ¿Para qué enfermedades podría servir cada fármaco?
    
    Args:
        predictions: Lista de (fármaco, enfermedad, score)
        ground_truth: Pares conocidos para validación
        top_k: Número de predicciones top a analizar
        
    Returns:
        Diccionario con análisis detallado
    """
    # Ordenar por score
    sorted_preds = sorted(predictions, key=lambda x: x[2], reverse=True)
    top_predictions = sorted_preds[:top_k]
    
    analysis = {
        'top_predictions': top_predictions,
        'num_total_predictions': len(predictions),
        'score_distribution': {
            'mean': np.mean([p[2] for p in predictions]),
            'std': np.std([p[2] for p in predictions]),
            'min': min(p[2] for p in predictions),
            'max': max(p[2] for p in predictions)
        }
    }
    
    # Análisis por enfermedad
    by_disease = {}
    for drug, disease, score in sorted_preds:
        if disease not in by_disease:
            by_disease[disease] = []
        by_disease[disease].append((drug, score))
    
    analysis['predictions_by_disease'] = {
        d: preds[:5] for d, preds in by_disease.items()  # Top 5 por enfermedad
    }
    
    # Análisis por fármaco
    by_drug = {}
    for drug, disease, score in sorted_preds:
        if drug not in by_drug:
            by_drug[drug] = []
        by_drug[drug].append((disease, score))
    
    analysis['predictions_by_drug'] = {
        d: preds[:5] for d, preds in by_drug.items()  # Top 5 por fármaco
    }
    
    # Validación retrospectiva si hay ground truth
    if ground_truth:
        gt_set = set(ground_truth)
        hits_in_top_k = sum(1 for d, dis, s in top_predictions if (d, dis) in gt_set)
        analysis['retrospective_validation'] = {
            'hits_in_top_k': hits_in_top_k,
            'precision_at_k': hits_in_top_k / top_k if top_k > 0 else 0
        }
    
    return analysis


def format_predictions_report(
    analysis: Dict[str, Any],
    drug_names: Optional[Dict[str, str]] = None,
    disease_names: Optional[Dict[str, str]] = None
) -> str:
    """
    Formatea el análisis de predicciones como reporte legible.
    
    Args:
        analysis: Resultado de analyze_predictions()
        drug_names: Mapeo ID -> nombre de fármaco
        disease_names: Mapeo ID -> nombre de enfermedad
        
    Returns:
        Reporte formateado como string
    """
    lines = [
        "=" * 60,
        "REPORTE DE PREDICCIONES - DRUG REPURPOSING",
        "=" * 60,
        "",
        f"Total de predicciones: {analysis['num_total_predictions']:,}",
        "",
        "Distribución de scores:",
        f"  Mean: {analysis['score_distribution']['mean']:.4f}",
        f"  Std:  {analysis['score_distribution']['std']:.4f}",
        f"  Min:  {analysis['score_distribution']['min']:.4f}",
        f"  Max:  {analysis['score_distribution']['max']:.4f}",
        "",
        "-" * 60,
        "TOP PREDICCIONES:",
        "-" * 60
    ]
    
    for i, (drug, disease, score) in enumerate(analysis['top_predictions'], 1):
        drug_name = drug_names.get(drug, drug) if drug_names else drug
        disease_name = disease_names.get(disease, disease) if disease_names else disease
        lines.append(f"{i:3d}. {drug_name} -> {disease_name} (score: {score:.4f})")
    
    if 'retrospective_validation' in analysis:
        lines.extend([
            "",
            "-" * 60,
            "VALIDACIÓN RETROSPECTIVA:",
            "-" * 60,
            f"Hits en Top-K: {analysis['retrospective_validation']['hits_in_top_k']}",
            f"Precision@K:   {analysis['retrospective_validation']['precision_at_k']:.4f}"
        ])
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


# =============================================================================
# UTILIDADES PARA GRAFOS
# =============================================================================

def compute_graph_statistics(data) -> Dict[str, Any]:
    """
    Calcula estadísticas del grafo heterogéneo.
    
    TEORÍA:
    -------
    Las estadísticas del grafo ayudan a entender:
    - Tamaño: ¿Es computacionalmente manejable?
    - Densidad: ¿Qué tan conectado está?
    - Heterogeneidad: ¿Cuántos tipos de nodos/aristas?
    - Balance: ¿Están balanceados los tipos?
    
    Estas estadísticas informan decisiones de diseño:
    - Grafos muy densos pueden requerir sampling
    - Grafos muy sparse pueden necesitar más capas
    - Desbalance extremo puede requerir oversampling
    
    Args:
        data: HeteroData de PyTorch Geometric
        
    Returns:
        Diccionario con estadísticas
    """
    stats = {
        'node_types': {},
        'edge_types': {},
        'total_nodes': 0,
        'total_edges': 0
    }
    
    # Estadísticas por tipo de nodo
    for node_type in data.node_types:
        num_nodes = data[node_type].num_nodes
        stats['node_types'][node_type] = num_nodes
        stats['total_nodes'] += num_nodes
    
    # Estadísticas por tipo de arista
    for edge_type in data.edge_types:
        edge_index = data[edge_type].edge_index
        num_edges = edge_index.size(1)
        stats['edge_types'][str(edge_type)] = num_edges
        stats['total_edges'] += num_edges
    
    # Métricas derivadas
    stats['num_node_types'] = len(data.node_types)
    stats['num_edge_types'] = len(data.edge_types)
    
    # Densidad aproximada
    max_edges = stats['total_nodes'] ** 2
    stats['density'] = stats['total_edges'] / max_edges if max_edges > 0 else 0
    
    return stats


def print_graph_statistics(stats: Dict[str, Any]) -> None:
    """Imprime estadísticas del grafo de forma legible."""
    print("\n" + "=" * 50)
    print("ESTADÍSTICAS DEL GRAFO")
    print("=" * 50)
    
    print(f"\nNodos totales: {stats['total_nodes']:,}")
    print(f"Aristas totales: {stats['total_edges']:,}")
    print(f"Tipos de nodos: {stats['num_node_types']}")
    print(f"Tipos de aristas: {stats['num_edge_types']}")
    print(f"Densidad: {stats['density']:.6f}")
    
    print("\nNodos por tipo:")
    for node_type, count in stats['node_types'].items():
        pct = count / stats['total_nodes'] * 100
        print(f"  {node_type}: {count:,} ({pct:.1f}%)")
    
    print("\nAristas por tipo:")
    for edge_type, count in stats['edge_types'].items():
        pct = count / stats['total_edges'] * 100
        print(f"  {edge_type}: {count:,} ({pct:.1f}%)")
    
    print("=" * 50 + "\n")


# =============================================================================
# EXPORT DE RESULTADOS
# =============================================================================

def save_results_json(
    results: Dict[str, Any],
    path: str,
    indent: int = 2
) -> None:
    """
    Guarda resultados en formato JSON.
    
    Args:
        results: Diccionario de resultados
        path: Ruta del archivo
        indent: Indentación para legibilidad
    """
    # Crear directorio si no existe
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    # Convertir tipos no serializables
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        return obj
    
    # Serializar recursivamente
    def serialize(d):
        if isinstance(d, dict):
            return {k: serialize(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [serialize(item) for item in d]
        else:
            return convert(d)
    
    serialized = serialize(results)
    
    with open(path, 'w') as f:
        json.dump(serialized, f, indent=indent)
    
    logging.info(f"Resultados guardados: {path}")


def load_results_json(path: str) -> Dict[str, Any]:
    """Carga resultados desde JSON."""
    with open(path, 'r') as f:
        return json.load(f)


# =============================================================================
# MAIN - TESTS DE UTILIDADES
# =============================================================================

if __name__ == "__main__":
    # Test básico de utilidades
    print("Testing Drug Repurposing GNN Utilities")
    print("=" * 50)
    
    # Test set_seed
    set_seed(42)
    print("✓ set_seed()")
    
    # Test logging
    logger = setup_logging(log_level=logging.INFO)
    logger.info("Test log message")
    print("✓ setup_logging()")
    
    # Test device
    device = get_device()
    print(f"✓ get_device() -> {device}")
    
    # Test Timer
    with Timer("Test operation"):
        time.sleep(0.1)
    print("✓ Timer")
    
    # Test memory
    mem = get_memory_usage()
    print(f"✓ get_memory_usage() -> CPU: {mem['cpu_used_gb']:.2f} GB")
    
    print("\n" + "=" * 50)
    print("Todas las utilidades funcionan correctamente!")
