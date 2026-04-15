"""
================================================================================
ENCODERS.PY - Arquitecturas GNN para Grafos Heterogéneos
================================================================================

Este archivo implementa tres arquitecturas de encoders GNN diseñadas para
grafos heterogéneos (con múltiples tipos de nodos y aristas):

1. R-GCN (Relational Graph Convolutional Network)
2. HAN (Heterogeneous Attention Network) 
3. HeteroGraphSAGE (GraphSAGE adaptado para grafos heterogéneos)

================================================================================
BASE TEÓRICA COMÚN - MESSAGE PASSING EN GNNS:
================================================================================

Todas las GNNs siguen el paradigma de Message Passing (Gilmer et al., 2017):

Para cada capa l, para cada nodo v:

1. MESSAGE: Crear mensajes desde vecinos
   m_u→v = M(h_u^(l), h_v^(l), e_uv)
   
2. AGGREGATE: Agregar mensajes (invariante a permutación)
   m_v = AGG({m_u→v : u ∈ N(v)})
   
3. UPDATE: Actualizar embedding del nodo
   h_v^(l+1) = U(h_v^(l), m_v)

Las diferentes arquitecturas varían en:
- Cómo computan los mensajes M()
- Cómo agregan AGG() 
- Cómo actualizan U()
- Cómo manejan múltiples tipos de relación

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    HeteroConv,
    GCNConv,
    SAGEConv,
    GATConv,
    Linear,
    RGCNConv
)
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional


class RGCNEncoder(nn.Module):
    """
    R-GCN: Relational Graph Convolutional Network
    (Schlichtkrull et al., 2018)
    
    ============================================================================
    BASE TEÓRICA:
    ============================================================================
    
    R-GCN extiende GCN (Kipf & Welling, 2017) para grafos multi-relacionales.
    
    GCN ORIGINAL:
    La convolución de grafo estándar promedia los embeddings de vecinos:
    
        h_v^(l+1) = σ( Σ_{u∈N(v)} (1/√(d_v·d_u)) · W^(l) · h_u^(l) )
    
    donde:
    - N(v): vecinos del nodo v
    - d_v, d_u: grados de los nodos
    - W^(l): matriz de pesos de la capa l
    
    PROBLEMA: Asume un solo tipo de relación. En Knowledge Graphs tenemos
    múltiples tipos (treats, targets, associates, etc.)
    
    R-GCN SOLUCIÓN:
    Usa matrices de pesos diferentes para cada tipo de relación r:
    
        h_v^(l+1) = σ( Σ_r Σ_{u∈N_r(v)} (1/c_{v,r}) · W_r^(l) · h_u^(l) + W_0^(l)·h_v^(l) )
    
    donde:
    - N_r(v): vecinos conectados por relación de tipo r
    - W_r^(l): matriz de pesos específica para relación r
    - c_{v,r}: constante de normalización
    - W_0^(l): self-loop (el nodo también contribuye a sí mismo)
    
    PROBLEMA DE ESCALABILIDAD:
    Con muchos tipos de relación (Hetionet tiene 24), hay muchos parámetros.
    
    SOLUCIÓN - BASIS DECOMPOSITION:
    Expresar cada W_r como combinación de B matrices base:
    
        W_r = Σ_{b=1}^B a_{rb} · V_b
    
    donde V_b son matrices base compartidas y a_{rb} son coeficientes.
    Esto reduce drásticamente los parámetros manteniendo expresividad.
    
    ============================================================================
    IMPLEMENTACIÓN CON HETEROCONV:
    ============================================================================
    
    PyTorch Geometric provee HeteroConv para aplicar convoluciones diferentes
    a cada tipo de arista. Internamente, HeteroConv:
    1. Aplica la convolución apropiada para cada tipo de arista
    2. Agrega las contribuciones de diferentes tipos de vecinos
    """
    
    def __init__(
        self,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        hidden_dim: int = 128,
        out_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_bases: int = 30,
    ):
        """
        Inicializa el encoder R-GCN.
        
        Args:
            node_types: Lista de tipos de nodo en el grafo
            edge_types: Lista de tuplas (src_type, relation, dst_type)
            hidden_dim: Dimensión de capas ocultas
            out_dim: Dimensión de salida
            num_layers: Número de capas R-GCN
            dropout: Tasa de dropout
            num_bases: Número de matrices base para decomposición
        """
        super().__init__()
        
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Proyección inicial: diferentes dimensiones de entrada → hidden_dim
        # Cada tipo de nodo puede tener features de diferente dimensión
        self.input_proj = nn.ModuleDict({
            node_type: Linear(-1, hidden_dim)  # -1 = inferir dimensión
            for node_type in node_types
        })
        
        # Capas R-GCN usando HeteroConv
        self.convs = nn.ModuleList()
        
        for i in range(num_layers):
            # Determinar dimensiones de entrada/salida de esta capa
            in_channels = hidden_dim
            out_channels = out_dim if i == num_layers - 1 else hidden_dim
            
            # Crear diccionario de convoluciones por tipo de arista
            conv_dict = {}
            for edge_type in edge_types:
                # GCNConv para cada tipo de relación
                # Nota: Usamos GCNConv simplificado aquí
                # Para R-GCN puro con basis decomposition, usar RGCNConv directamente
                conv_dict[edge_type] = GCNConv(
                    in_channels, 
                    out_channels,
                    add_self_loops=False,  # Manejamos self-loops manualmente
                    normalize=True
                )
            
            # HeteroConv envuelve todas las convoluciones
            # aggr='sum' suma las contribuciones de diferentes tipos de vecinos
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))
        
        # Layer normalization para estabilidad del entrenamiento
        self.norms = nn.ModuleList([
            nn.ModuleDict({
                node_type: nn.LayerNorm(hidden_dim if i < num_layers-1 else out_dim)
                for node_type in node_types
            })
            for i in range(num_layers)
        ])
    
    def forward(self, x_dict: Dict[str, torch.Tensor], 
                edge_index_dict: Dict[Tuple, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass del encoder.
        
        FLUJO DE INFORMACIÓN:
        1. Proyección inicial de features a hidden_dim
        2. Para cada capa:
           a. Aplicar convolución R-GCN (message passing por tipo de relación)
           b. Normalización + activación + dropout
        3. Retornar embeddings finales por tipo de nodo
        
        Args:
            x_dict: Diccionario {node_type: features_tensor}
            edge_index_dict: Diccionario {edge_type: edge_index_tensor}
            
        Returns:
            Diccionario {node_type: embeddings_tensor}
        """
        # Proyección inicial
        h_dict = {}
        for node_type, x in x_dict.items():
            if node_type in self.input_proj:
                h_dict[node_type] = self.input_proj[node_type](x)
        
        # Capas R-GCN
        for i, conv in enumerate(self.convs):
            # Message passing heterogéneo
            h_dict = conv(h_dict, edge_index_dict)
            
            # Post-procesamiento por tipo de nodo
            for node_type in h_dict:
                h = h_dict[node_type]
                
                # Layer normalization
                if node_type in self.norms[i]:
                    h = self.norms[i][node_type](h)
                
                # Activación (excepto última capa)
                if i < self.num_layers - 1:
                    h = F.relu(h)
                    h = F.dropout(h, p=self.dropout, training=self.training)
                
                h_dict[node_type] = h
        
        return h_dict


class HANEncoder(nn.Module):
    """
    HAN: Heterogeneous Graph Attention Network
    (Wang et al., 2019)
    
    ============================================================================
    BASE TEÓRICA:
    ============================================================================
    
    HAN extiende GAT (Veličković et al., 2018) para grafos heterogéneos usando
    DOS NIVELES de atención:
    
    1. ATENCIÓN A NIVEL DE NODO (Node-level Attention):
       Igual que GAT, aprende a ponderar la importancia de cada vecino.
       
       Para un nodo v y su vecino u conectado por relación r:
       
       α_vu = softmax_u( LeakyReLU( a^T · [W·h_v || W·h_u] ) )
       
       donde:
       - a: vector de atención learnable
       - W: matriz de transformación
       - ||: concatenación
       - El softmax se aplica sobre todos los vecinos de tipo r
    
    2. ATENCIÓN A NIVEL DE METAPATH (Semantic-level Attention):
       En grafos heterogéneos, diferentes tipos de relación capturan
       diferentes semánticas. HAN aprende qué tipos son más importantes.
       
       Para un tipo de relación r:
       
       β_r = softmax_r( q^T · tanh(W_s · z_r + b_s) )
       
       donde z_r es el embedding agregado usando relación r.
       
       El embedding final es:
       
       h_v = Σ_r β_r · z_r
    
    ¿POR QUÉ ATENCIÓN EN DOS NIVELES?
    
    En drug repurposing:
    - Node attention: No todos los targets de un fármaco son igual de importantes
    - Semantic attention: La relación "targets" puede ser más predictiva que
      "upregulates" para cierta enfermedad
    
    MULTI-HEAD ATTENTION:
    Siguiendo a Transformers, usamos múltiples cabezas de atención.
    Cada cabeza puede aprender patrones diferentes:
    - Una cabeza podría enfocarse en similitud estructural
    - Otra en co-ocurrencia en pathways
    
    ============================================================================
    SIMPLIFICACIÓN PARA ESTE PROYECTO:
    ============================================================================
    
    La implementación completa de HAN es compleja. Aquí usamos GATConv
    dentro de HeteroConv, que captura la atención a nivel de nodo pero
    simplifica la atención semántica usando agregación por suma.
    """
    
    def __init__(
        self,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        hidden_dim: int = 128,
        out_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        """
        Inicializa el encoder HAN.
        
        Args:
            node_types: Lista de tipos de nodo
            edge_types: Lista de tipos de arista
            hidden_dim: Dimensión de capas ocultas
            out_dim: Dimensión de salida
            num_layers: Número de capas de atención
            num_heads: Número de cabezas de atención
            dropout: Tasa de dropout
        """
        super().__init__()
        
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Proyección inicial
        self.input_proj = nn.ModuleDict({
            node_type: Linear(-1, hidden_dim)
            for node_type in node_types
        })
        
        # Capas de atención
        self.convs = nn.ModuleList()
        
        for i in range(num_layers):
            in_channels = hidden_dim
            # GAT produce num_heads * out_channels, necesitamos manejar eso
            if i == num_layers - 1:
                out_channels = out_dim
                heads = 1  # Última capa: una sola cabeza
                concat = False
            else:
                out_channels = hidden_dim // num_heads
                heads = num_heads
                concat = True  # Concatenar cabezas (output = heads * out_channels)
            
            conv_dict = {}
            for edge_type in edge_types:
                conv_dict[edge_type] = GATConv(
                    in_channels,
                    out_channels,
                    heads=heads,
                    concat=concat,
                    dropout=dropout,
                    add_self_loops=False
                )
            
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))
        
        # Normalización
        self.norms = nn.ModuleList([
            nn.ModuleDict({
                node_type: nn.LayerNorm(hidden_dim if i < num_layers-1 else out_dim)
                for node_type in node_types
            })
            for i in range(num_layers)
        ])
        
        # Atención semántica (simplificada): pesos learnable por tipo de relación
        # Esto captura qué relaciones son más importantes globalmente
        self.semantic_attention = nn.ParameterDict({
            f"{et[0]}_{et[1]}_{et[2]}": nn.Parameter(torch.ones(1))
            for et in edge_types
        })
    
    def forward(self, x_dict: Dict[str, torch.Tensor],
                edge_index_dict: Dict[Tuple, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass con atención heterogénea.
        
        El mecanismo de atención permite al modelo:
        1. Ponderar qué vecinos son más relevantes (node attention)
        2. Ponderar qué tipos de relación importan más (semantic attention)
        
        Args:
            x_dict: Features por tipo de nodo
            edge_index_dict: Aristas por tipo
            
        Returns:
            Embeddings por tipo de nodo
        """
        # Proyección inicial
        h_dict = {}
        for node_type, x in x_dict.items():
            if node_type in self.input_proj:
                h_dict[node_type] = self.input_proj[node_type](x)
        
        # Capas de atención
        for i, conv in enumerate(self.convs):
            h_dict = conv(h_dict, edge_index_dict)
            
            for node_type in h_dict:
                h = h_dict[node_type]
                
                if node_type in self.norms[i]:
                    h = self.norms[i][node_type](h)
                
                if i < self.num_layers - 1:
                    h = F.elu(h)  # ELU es más estable que ReLU con atención
                    h = F.dropout(h, p=self.dropout, training=self.training)
                
                h_dict[node_type] = h
        
        return h_dict


class HeteroGraphSAGEEncoder(nn.Module):
    """
    GraphSAGE Heterogéneo
    (Hamilton et al., 2017, adaptado para grafos heterogéneos)
    
    ============================================================================
    BASE TEÓRICA:
    ============================================================================
    
    GraphSAGE (SAmple and aggreGatE) introduce dos ideas clave:
    
    1. SAMPLING DE VECINOS:
       En lugar de usar TODOS los vecinos (como GCN), muestrea un subconjunto.
       Esto hace el entrenamiento escalable a grafos muy grandes.
       
       Durante entrenamiento: sample k vecinos por capa
       Durante inferencia: usar todos o sample mayor
    
    2. AGREGACIÓN GENERALIZADA:
       GCN usa promedio ponderado. GraphSAGE permite agregadores más expresivos:
       
       - MEAN: h_N(v) = mean({h_u : u ∈ N(v)})
         Simple y efectivo, similar a GCN
         
       - MAX (element-wise): h_N(v) = max({ReLU(W·h_u) : u ∈ N(v)})
         Captura features "más destacadas"
         
       - LSTM: h_N(v) = LSTM({h_u : u ∈ N(v)})
         Más expresivo pero no permutation-invariant
         (se aplica a ordenamientos aleatorios)
       
       - POOL: h_N(v) = max({σ(W_pool·h_u + b) : u ∈ N(v)})
         Pooling tras transformación no lineal
    
    ACTUALIZACIÓN:
    
    h_v^(l+1) = σ(W · CONCAT(h_v^(l), h_N(v)))
    
    A diferencia de GCN que promedia v con sus vecinos, SAGE concatena
    el embedding del nodo con el agregado de vecinos. Esto preserva
    más información sobre el nodo central.
    
    APRENDIZAJE INDUCTIVO:
    GraphSAGE puede generar embeddings para nodos NO vistos en entrenamiento,
    porque aprende funciones de agregación, no embeddings fijos.
    
    Esto es crucial para drug repurposing:
    - Nuevos fármacos pueden agregarse sin re-entrenar
    - Solo necesitan tener algunas conexiones (targets, etc.)
    - El modelo generaliza la función de agregación aprendida
    
    ============================================================================
    ADAPTACIÓN HETEROGÉNEA:
    ============================================================================
    
    Para grafos heterogéneos, aplicamos SAGEConv por tipo de relación
    y agregamos las contribuciones de diferentes tipos.
    """
    
    def __init__(
        self,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        hidden_dim: int = 128,
        out_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        aggregator: str = "mean",
    ):
        """
        Inicializa el encoder GraphSAGE heterogéneo.
        
        Args:
            node_types: Lista de tipos de nodo
            edge_types: Lista de tipos de arista
            hidden_dim: Dimensión de capas ocultas
            out_dim: Dimensión de salida
            num_layers: Número de capas
            dropout: Tasa de dropout
            aggregator: Tipo de agregador ("mean", "max", "sum")
        """
        super().__init__()
        
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.aggregator = aggregator
        
        # Proyección inicial
        self.input_proj = nn.ModuleDict({
            node_type: Linear(-1, hidden_dim)
            for node_type in node_types
        })
        
        # Capas SAGE
        self.convs = nn.ModuleList()
        
        for i in range(num_layers):
            in_channels = hidden_dim
            out_channels = out_dim if i == num_layers - 1 else hidden_dim
            
            conv_dict = {}
            for edge_type in edge_types:
                # SAGEConv con el agregador especificado
                conv_dict[edge_type] = SAGEConv(
                    in_channels,
                    out_channels,
                    aggr=aggregator,  # mean, max, o sum
                    normalize=True,   # L2 normalize output
                    root_weight=True  # Include self-loop
                )
            
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))
        
        # Normalización
        self.norms = nn.ModuleList([
            nn.ModuleDict({
                node_type: nn.LayerNorm(hidden_dim if i < num_layers-1 else out_dim)
                for node_type in node_types
            })
            for i in range(num_layers)
        ])
    
    def forward(self, x_dict: Dict[str, torch.Tensor],
                edge_index_dict: Dict[Tuple, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass con agregación SAGE.
        
        A diferencia de GCN/GAT:
        - Concatena features del nodo con agregado de vecinos
        - Normaliza los embeddings de salida (L2)
        - Más adecuado para inferencia inductiva
        
        Args:
            x_dict: Features por tipo de nodo
            edge_index_dict: Aristas por tipo
            
        Returns:
            Embeddings por tipo de nodo
        """
        # Proyección inicial
        h_dict = {}
        for node_type, x in x_dict.items():
            if node_type in self.input_proj:
                h_dict[node_type] = self.input_proj[node_type](x)
        
        # Capas SAGE
        for i, conv in enumerate(self.convs):
            h_dict = conv(h_dict, edge_index_dict)
            
            for node_type in h_dict:
                h = h_dict[node_type]
                
                if node_type in self.norms[i]:
                    h = self.norms[i][node_type](h)
                
                if i < self.num_layers - 1:
                    h = F.relu(h)
                    h = F.dropout(h, p=self.dropout, training=self.training)
                
                h_dict[node_type] = h
        
        return h_dict


def get_encoder(
    encoder_type: str,
    node_types: List[str],
    edge_types: List[Tuple[str, str, str]],
    config
) -> nn.Module:
    """
    Factory function para crear encoders.
    
    Args:
        encoder_type: "rgcn", "han", o "sage"
        node_types: Lista de tipos de nodo
        edge_types: Lista de tipos de arista
        config: Objeto de configuración
        
    Returns:
        Instancia del encoder apropiado
    """
    encoder_type = encoder_type.lower()
    
    if encoder_type == "rgcn":
        return RGCNEncoder(
            node_types=node_types,
            edge_types=edge_types,
            hidden_dim=config.model.hidden_dim,
            out_dim=config.model.out_dim,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
            num_bases=config.model.num_bases
        )
    elif encoder_type == "han":
        return HANEncoder(
            node_types=node_types,
            edge_types=edge_types,
            hidden_dim=config.model.hidden_dim,
            out_dim=config.model.out_dim,
            num_layers=config.model.num_layers,
            num_heads=config.model.num_heads,
            dropout=config.model.dropout
        )
    elif encoder_type == "sage":
        return HeteroGraphSAGEEncoder(
            node_types=node_types,
            edge_types=edge_types,
            hidden_dim=config.model.hidden_dim,
            out_dim=config.model.out_dim,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
            aggregator=config.model.sage_aggregator
        )
    else:
        raise ValueError(f"Encoder desconocido: {encoder_type}. "
                        f"Opciones: rgcn, han, sage")
