"""
================================================================================
CONFIG.PY - Configuración Central del Proyecto
================================================================================

BASE TEÓRICA:
-------------
Este archivo centraliza todos los hiperparámetros del experimento. La elección
de estos valores está fundamentada en:

1. DIMENSIÓN DE EMBEDDINGS (128-256):
   - Basado en Decagon (Zitnik et al., 2018) que usa 64-128 dimensiones
   - Dimensiones mayores capturan más información pero aumentan el overfitting
   - 128 es un buen balance entre expresividad y regularización

2. NÚMERO DE CAPAS (2-3):
   - En GNNs, cada capa expande el campo receptivo en 1-hop
   - Con 2 capas: información de vecinos a distancia 2
   - Más de 3 capas suele causar over-smoothing (todos los nodos convergen
     a representaciones similares)

3. LEARNING RATE (1e-3 a 1e-4):
   - Adam con lr=1e-3 es estándar para GNNs
   - Usamos scheduler para reducir gradualmente

4. NEGATIVE SAMPLING:
   - Por cada arista positiva (drug→disease que existe), generamos k negativos
   - Ratio típico: 1:5 a 1:10 (1 positivo por cada 5-10 negativos)
   - Esto balancea el dataset que naturalmente tiene muy pocas aristas positivas

================================================================================
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import torch


@dataclass
class DataConfig:
    """
    Configuración para la carga y preprocesamiento de datos.
    
    DATASET HETIONET:
    - ~47,000 nodos de 11 tipos diferentes
    - ~2.25M aristas de 24 tipos de relación
    - Incluye: Drug, Disease, Gene, Anatomy, Compound, Side Effect, etc.
    
    Para este proyecto nos enfocamos en los 4 tipos principales:
    - Drug (Compound): Fármacos/compuestos químicos
    - Disease: Enfermedades
    - Gene: Genes/proteínas target
    - Anatomy: Localizaciones anatómicas
    """
    # Ruta para guardar los datos descargados
    data_dir: str = "./data"
    
    # URL del dataset Hetionet (formato JSON)
    hetionet_url: str = "https://github.com/hetio/hetionet/raw/main/hetnet/json/hetionet-v1.0.json.bz2"
    
    # Tipos de nodos a incluir en el grafo
    # La hipótesis de network medicine sugiere que estos son los más relevantes
    node_types: List[str] = field(default_factory=lambda: [
        "Compound",   # Fármacos
        "Disease",    # Enfermedades  
        "Gene",       # Genes/proteínas
        "Anatomy"     # Localizaciones anatómicas
    ])
    
    # Relación objetivo para link prediction
    # "treats" conecta Compound → Disease (indicaciones terapéuticas)
    target_edge_type: tuple = ("Compound", "treats", "Disease")
    
    # Proporción del dataset para train/val/test
    # Usamos 80/10/10 siguiendo convenciones estándar
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Semilla para reproducibilidad
    random_seed: int = 42


@dataclass
class ModelConfig:
    """
    Configuración de las arquitecturas GNN.
    
    BASE TEÓRICA DE LAS ARQUITECTURAS:
    
    1. R-GCN (Relational Graph Convolutional Network):
       - Extiende GCN para grafos con múltiples tipos de relación
       - Usa matrices de pesos diferentes para cada tipo de arista
       - Fórmula: h_i^(l+1) = σ(Σ_r Σ_{j∈N_i^r} (1/c_{i,r}) W_r^(l) h_j^(l) + W_0^(l) h_i^(l))
       - Donde r indexa el tipo de relación y c_{i,r} es factor de normalización
    
    2. HAN (Heterogeneous Attention Network):
       - Aplica atención a nivel de nodo Y a nivel de metapath
       - Permite ponderar la importancia de diferentes tipos de vecinos
       - Útil cuando no todas las relaciones son igual de informativas
    
    3. GraphSAGE Heterogéneo:
       - Sampling de vecinos para escalabilidad
       - Agregación por muestreo en lugar de usar todos los vecinos
       - Permite inducción: generar embeddings para nodos no vistos
    """
    # Dimensión de los embeddings latentes
    # 128 es balance entre expresividad y evitar overfitting
    hidden_dim: int = 128
    
    # Dimensión de salida del encoder
    out_dim: int = 64
    
    # Número de capas GNN
    # 2 capas = información de 2-hop neighborhood
    num_layers: int = 2
    
    # Dropout para regularización
    # 0.3-0.5 es típico para GNNs
    dropout: float = 0.3
    
    # Número de cabezas de atención (para GAT/HAN)
    # Multi-head attention captura diferentes patrones de relación
    num_heads: int = 4
    
    # Usar basis decomposition en R-GCN para reducir parámetros
    # Importante cuando hay muchos tipos de relación
    use_basis: bool = True
    num_bases: int = 30  # Número de matrices base
    
    # Tipo de agregación para GraphSAGE
    sage_aggregator: str = "mean"  # Opciones: mean, max, lstm


@dataclass
class TrainingConfig:
    """
    Configuración del entrenamiento.
    
    NEGATIVE SAMPLING:
    -----------------
    En link prediction, tenemos aristas positivas (existen) y necesitamos
    generar negativas (no existen). El ratio negativo:positivo afecta:
    - Muy pocos negativos: modelo muy optimista, muchos falsos positivos
    - Muchos negativos: entrenamiento más lento, mejor calibración
    
    EARLY STOPPING:
    ---------------
    Detenemos si la métrica de validación no mejora en N épocas.
    Previene overfitting al dataset de entrenamiento.
    """
    # Número de épocas
    num_epochs: int = 200
    
    # Tamaño del batch para entrenamiento
    batch_size: int = 1024
    
    # Learning rate inicial
    learning_rate: float = 1e-3
    
    # Weight decay (L2 regularization)
    weight_decay: float = 1e-5
    
    # Ratio de ejemplos negativos por cada positivo
    # 5 negativos por cada positivo es un buen balance
    negative_sampling_ratio: int = 5
    
    # Early stopping: paciencia (épocas sin mejora antes de parar)
    patience: int = 20
    
    # Factor de reducción del learning rate
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 10
    
    # Device para entrenamiento
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Checkpointing
    save_best_model: bool = True
    checkpoint_dir: str = "./checkpoints"


@dataclass
class EvaluationConfig:
    """
    Configuración de métricas de evaluación.
    
    MÉTRICAS PARA LINK PREDICTION:
    ------------------------------
    
    1. Hits@K: Proporción de aristas positivas rankeadas en top-K
       - Hits@10: ¿Está la arista correcta en los 10 mejores candidatos?
       - Interpretación práctica: Si un médico revisa los top-10 fármacos
         sugeridos, ¿cuántas veces encontrará uno efectivo?
    
    2. MRR (Mean Reciprocal Rank): Media de 1/rank de la arista correcta
       - Si el fármaco correcto está en posición 1 → 1/1 = 1
       - Si está en posición 5 → 1/5 = 0.2
       - Premia más a las predicciones que están muy arriba
    
    3. AUC-ROC: Área bajo la curva ROC
       - Mide la capacidad de distinguir positivos de negativos
       - 0.5 = aleatorio, 1.0 = perfecto
       - Útil para ver calibración general del modelo
    """
    # Valores de K para Hits@K
    hits_k_values: List[int] = field(default_factory=lambda: [1, 3, 10, 50, 100])
    
    # Número de negativos a considerar para ranking
    # En evaluación usamos más negativos para ranking más realista
    num_negatives_eval: int = 100
    
    # Si hacer filtered evaluation
    # Filtered: excluir aristas que existen en train del ranking
    # Esto evita penalizar al modelo por rankear alto aristas verdaderas
    filtered: bool = True


@dataclass
class AblationConfig:
    """
    Configuración para el estudio de ablación.
    
    PROPÓSITO DEL ESTUDIO DE ABLACIÓN:
    -----------------------------------
    Queremos entender qué tipos de entidades intermedias son más importantes
    para la predicción drug→disease.
    
    Hipótesis de Network Medicine (Barabási, cubierto en L15):
    - Los fármacos actúan a través de proteínas/genes target
    - Las enfermedades tienen "módulos" de proteínas asociadas
    - La proximidad en la red PPI predice eficacia terapéutica
    
    Si esta hipótesis es correcta:
    - Quitar genes debería degradar mucho el rendimiento
    - Quitar anatomías podría tener menor impacto
    - Quitar ambos debería ser catastrófico
    
    Esto validaría que el modelo está usando la estructura biológica
    correctamente y no solo memorizando patrones superficiales.
    """
    # Configuraciones de ablación a probar
    # Cada configuración especifica qué tipos de nodos mantener
    ablation_configs: List[Dict] = field(default_factory=lambda: [
        {"name": "full", "node_types": ["Compound", "Disease", "Gene", "Anatomy"]},
        {"name": "no_anatomy", "node_types": ["Compound", "Disease", "Gene"]},
        {"name": "no_gene", "node_types": ["Compound", "Disease", "Anatomy"]},
        {"name": "no_intermediate", "node_types": ["Compound", "Disease"]},
    ])
    
    # Número de repeticiones por configuración (para intervalos de confianza)
    num_runs: int = 3


@dataclass
class Config:
    """Configuración global que agrupa todas las sub-configuraciones."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    
    # Nombre del experimento para logging
    experiment_name: str = "drug_repurposing_gnn"
    
    # Semilla global
    seed: int = 42


def get_config() -> Config:
    """Retorna la configuración por defecto."""
    return Config()

# Tipos válidos de encoder y decoder para CLI
ENCODER_TYPES = ["rgcn", "han", "sage"]
DECODER_TYPES = ["distmult", "dotproduct", "mlp"]

if __name__ == "__main__":
    # Test de la configuración
    config = get_config()
    print(f"Experiment: {config.experiment_name}")
    print(f"Device: {config.training.device}")
    print(f"Node types: {config.data.node_types}")
    print(f"Hidden dim: {config.model.hidden_dim}")
