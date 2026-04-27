"""
================================================================================
FULL_MODEL.PY - Modelo Completo Encoder-Decoder para Drug Repurposing
================================================================================

Este módulo combina:
1. Un encoder GNN heterogéneo (R-GCN, HAN, o GraphSAGE)
2. Un decoder para link prediction (DistMult, Dot Product, o MLP)

El pipeline completo para predecir si un fármaco trata una enfermedad:

    1. Encoder: Grafo → Embeddings de nodos
       h_drug, h_disease, h_gene, h_anatomy = Encoder(G)
    
    2. Decoder: Embeddings → Score de arista
       score(drug_i, disease_j) = Decoder(h_drug[i], h_disease[j])
    
    3. Loss: Comparar scores positivos vs negativos
       L = BCE(scores_positivos, 1) + BCE(scores_negativos, 0)

================================================================================
BASE TEÓRICA - ARQUITECTURA ENCODER-DECODER:
================================================================================

Esta arquitectura es estándar en aprendizaje de representaciones de grafos:

1. ENCODER (Representación):
   - Transforma el grafo en un espacio latente
   - Captura la estructura local y global
   - Los embeddings codifican información sobre cada nodo y su contexto
   
2. DECODER (Predicción):
   - Opera sobre embeddings, no sobre el grafo original
   - Predice propiedades específicas (existencia de aristas, en este caso)
   - Más eficiente que trabajar con el grafo directamente

VENTAJAS DE SEPARAR ENCODER Y DECODER:
- Modularidad: podemos cambiar uno sin afectar el otro
- Los embeddings del encoder pueden usarse para otras tareas
- Permite pre-entrenamiento del encoder en tareas auxiliares

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional

from models.encoders import get_encoder
from models.decoders import get_decoder


class DrugRepurposingModel(nn.Module):
    """
    Modelo completo para Drug Repurposing como Link Prediction.
    
    ARQUITECTURA:
    
    Input: Grafo heterogéneo G = (V, E) con tipos {Drug, Disease, Gene, Anatomy}
    
    1. Encoder GNN:
       - Múltiples capas de message passing heterogéneo
       - Cada capa: Aggregate(neighbors) → Transform → Update
       - Output: embeddings h_v para cada nodo v
    
    2. Decoder:
       - Input: pares (h_drug, h_disease)
       - Output: probabilidad de que exista arista "treats"
    
    ENTRENAMIENTO:
    - Aristas positivas: pares (drug, disease) que existen en train
    - Aristas negativas: pares (drug, disease) muestreados que NO existen
    - Loss: Binary Cross-Entropy entre scores y labels (1/0)
    
    INFERENCIA:
    - Para cada drug, rankear todas las diseases por score
    - Top-k diseases son las predicciones de nuevas indicaciones
    """
    
    def __init__(
        self,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        node_counts: Dict[str, int] = None,
        encoder_type: str = "rgcn",
        decoder_type: str = "distmult",
        hidden_dim: int = 128,
        out_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_heads: int = 4,
        num_bases: int = 30,
        sage_aggregator: str = "mean",
    ):
        """
        Inicializa el modelo completo.
        
        Args:
            node_types: Lista de tipos de nodo
            edge_types: Lista de tipos de arista
            encoder_type: Tipo de encoder ("rgcn", "han", "sage")
            decoder_type: Tipo de decoder ("distmult", "dot", "mlp")
            hidden_dim: Dimensión de capas ocultas
            out_dim: Dimensión de embeddings finales
            num_layers: Número de capas GNN
            dropout: Tasa de dropout
            num_heads: Número de cabezas de atención (para HAN)
            num_bases: Número de matrices base (para R-GCN)
            sage_aggregator: Tipo de agregador (para SAGE)
        """
        super().__init__()
        
        self.node_types = node_types
        self.edge_types = edge_types
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.out_dim = out_dim

        # Crear configuración mock para el encoder
        class MockConfig:
            class model:
                pass
        config = MockConfig()
        config.model.hidden_dim = hidden_dim
        config.model.out_dim = out_dim
        config.model.num_layers = num_layers
        config.model.dropout = dropout
        config.model.num_heads = num_heads
        config.model.num_bases = num_bases
        config.model.sage_aggregator = sage_aggregator

        # Crear encoder
        self.encoder = get_encoder(
            encoder_type=encoder_type,
            node_types=node_types,
            edge_types=edge_types,
            config=config,
            node_counts=node_counts,
        )
        
        # Crear decoder
        self.decoder = get_decoder(
            decoder_type=decoder_type,
            embedding_dim=out_dim
        )
    
    def encode(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Genera embeddings para todos los nodos.
        
        Este método ejecuta el encoder GNN para producir representaciones
        latentes de cada nodo, incorporando información de:
        - Features propias del nodo
        - Estructura del grafo (conexiones)
        - Tipos de relación (en grafos heterogéneos)
        
        Args:
            x_dict: Features iniciales por tipo de nodo
            edge_index_dict: Aristas por tipo
            
        Returns:
            Embeddings por tipo de nodo
        """
        return self.encoder(x_dict, edge_index_dict)
    
    def decode(
        self,
        h_src: torch.Tensor,
        h_dst: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcula scores para pares de nodos.
        
        Args:
            h_src: Embeddings de nodos fuente
            h_dst: Embeddings de nodos destino
            
        Returns:
            Scores (logits antes de sigmoid)
        """
        return self.decoder(h_src, h_dst)
    
    def forward(
        self,
        data: HeteroData,
        edge_label_index: torch.Tensor,
        src_type: str = "Compound",
        dst_type: str = "Disease"
    ) -> torch.Tensor:
        """
        Forward pass completo: grafo → scores de aristas.
        
        FLUJO:
        1. Extraer features y aristas del HeteroData
        2. Encoder: generar embeddings de todos los nodos
        3. Seleccionar embeddings de los nodos en edge_label_index
        4. Decoder: calcular scores para cada par
        
        Args:
            data: Objeto HeteroData con el grafo
            edge_label_index: Índices de aristas a evaluar [2, num_edges]
                             edge_label_index[0]: índices de nodos source
                             edge_label_index[1]: índices de nodos target
            src_type: Tipo de nodo fuente
            dst_type: Tipo de nodo destino
            
        Returns:
            Scores para cada arista [num_edges]
        """
        # Extraer features de nodos
        x_dict = {}
        for node_type in self.node_types:
            if hasattr(data[node_type], 'x') and data[node_type].x is not None:
                x_dict[node_type] = data[node_type].x
        
        # Extraer aristas (excluyendo las de entrenamiento si es necesario)
        edge_index_dict = {}
        for edge_type in self.edge_types:
            if hasattr(data[edge_type], 'edge_index'):
                edge_index_dict[edge_type] = data[edge_type].edge_index
        
        # Encode: generar embeddings
        h_dict = self.encode(x_dict, edge_index_dict)
        
        # Obtener embeddings de los nodos en el edge_label_index
        h_src = h_dict[src_type][edge_label_index[0]]  # [num_edges, out_dim]
        h_dst = h_dict[dst_type][edge_label_index[1]]  # [num_edges, out_dim]
        
        # Decode: calcular scores
        scores = self.decode(h_src, h_dst)  # [num_edges]
        
        return scores
    
    def get_embeddings(
        self,
        data: HeteroData
    ) -> Dict[str, torch.Tensor]:
        """
        Obtiene embeddings de todos los nodos (útil para análisis).
        
        Args:
            data: Objeto HeteroData
            
        Returns:
            Diccionario de embeddings por tipo de nodo
        """
        x_dict = {}
        for node_type in self.node_types:
            if hasattr(data[node_type], 'x') and data[node_type].x is not None:
                x_dict[node_type] = data[node_type].x
        
        edge_index_dict = {}
        for edge_type in self.edge_types:
            if hasattr(data[edge_type], 'edge_index'):
                edge_index_dict[edge_type] = data[edge_type].edge_index
        
        with torch.no_grad():
            h_dict = self.encode(x_dict, edge_index_dict)
        
        return h_dict
    
    def predict_all_pairs(
        self,
        data: HeteroData,
        src_type: str = "Compound",
        dst_type: str = "Disease"
    ) -> torch.Tensor:
        """
        Calcula scores para TODOS los pares posibles (src × dst).
        
        Útil para:
        - Ranking completo de predicciones
        - Encontrar nuevas indicaciones para todos los fármacos
        
        ADVERTENCIA: Costoso en memoria para grafos grandes.
        Para grafos grandes, usar predicción por batches.
        
        Args:
            data: Objeto HeteroData
            src_type: Tipo de nodo fuente
            dst_type: Tipo de nodo destino
            
        Returns:
            Matriz de scores [num_src, num_dst]
        """
        h_dict = self.get_embeddings(data)
        
        h_src = h_dict[src_type]  # [num_src, out_dim]
        h_dst = h_dict[dst_type]  # [num_dst, out_dim]
        
        # Usar forward_all del decoder
        scores = self.decoder.forward_all(h_src, h_dst)  # [num_src, num_dst]
        
        return scores


class LinkPredictionLoss(nn.Module):
    """
    Loss function para link prediction.
    
    ============================================================================
    BASE TEÓRICA:
    ============================================================================
    
    Para link prediction, típicamente usamos Binary Cross-Entropy (BCE):
    
        L = -1/N * Σ [y * log(σ(s)) + (1-y) * log(1 - σ(s))]
    
    donde:
    - y ∈ {0, 1}: label (1 = arista existe, 0 = no existe)
    - s: score del modelo
    - σ: sigmoid function
    
    NEGATIVE SAMPLING:
    El grafo tiene pocas aristas positivas relativo al total posible.
    Por ejemplo, en Hetionet hay ~47k nodos y ~2.25M aristas,
    pero el máximo posible sería ~2 billones de aristas.
    
    Técnica: por cada positivo, muestrear k negativos aleatorios.
    Esto balancea el dataset y hace el entrenamiento eficiente.
    
    MARGIN RANKING LOSS (alternativa):
        L = max(0, margin - s_pos + s_neg)
    
    Empuja a que positivos tengan score mayor que negativos por un margen.
    Usada en TransE y otros modelos de KG embedding.
    
    ============================================================================
    """
    
    def __init__(self, margin: float = 1.0, use_margin_loss: bool = False):
        """
        Inicializa la loss function.
        
        Args:
            margin: Margen para margin ranking loss
            use_margin_loss: Si True, usa margin loss en vez de BCE
        """
        super().__init__()
        self.margin = margin
        self.use_margin_loss = use_margin_loss
        
        if use_margin_loss:
            self.loss_fn = nn.MarginRankingLoss(margin=margin)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcula la loss.
        
        Args:
            scores: Scores del modelo [batch_size]
            labels: Labels verdaderos [batch_size] (1=positivo, 0=negativo)
            
        Returns:
            Valor escalar de la loss
        """
        if self.use_margin_loss:
            # Separar positivos y negativos
            pos_mask = labels == 1
            neg_mask = labels == 0
            
            pos_scores = scores[pos_mask]
            neg_scores = scores[neg_mask]
            
            # Margin loss requiere igual número de pos y neg
            min_len = min(len(pos_scores), len(neg_scores))
            if min_len == 0:
                return torch.tensor(0.0, device=scores.device)
            
            pos_scores = pos_scores[:min_len]
            neg_scores = neg_scores[:min_len]
            
            # Target: positivos > negativos
            target = torch.ones(min_len, device=scores.device)
            return self.loss_fn(pos_scores, neg_scores, target)
        else:
            # BCE con logits
            return self.loss_fn(scores, labels.float())


def create_model(
    data: HeteroData,
    config,
    encoder_type: str = "rgcn",
    decoder_type: str = "distmult"
) -> DrugRepurposingModel:
    """
    Factory function para crear el modelo a partir de los datos.
    
    Extrae automáticamente los tipos de nodo y arista del HeteroData.
    
    Args:
        data: Objeto HeteroData con el grafo
        config: Configuración del modelo
        encoder_type: Tipo de encoder
        decoder_type: Tipo de decoder
        
    Returns:
        Modelo inicializado
    """
    node_types = data.node_types
    edge_types = list(data.edge_types)
    node_counts = {nt: data[nt].num_nodes for nt in node_types}

    model = DrugRepurposingModel(
        node_types=node_types,
        edge_types=edge_types,
        node_counts=node_counts,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        hidden_dim=config.model.hidden_dim,
        out_dim=config.model.out_dim,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        num_heads=config.model.num_heads,
        num_bases=config.model.num_bases,
        sage_aggregator=config.model.sage_aggregator
    )
    
    return model


if __name__ == "__main__":
    # Test del modelo
    import sys
    sys.path.append("..")
    from config import get_config
    
    config = get_config()
    
    # Crear datos dummy para testing
    data = HeteroData()
    # OLD: data["Compound"].x = torch.randn(100, 128)
    # OLD: data["Disease"].x  = torch.randn(50, 128)
    # OLD: data["Gene"].x     = torch.randn(500, 128)
    # NEW: indices for learnable nn.Embedding lookup
    data["Compound"].x = torch.arange(100)
    data["Disease"].x  = torch.arange(50)
    data["Gene"].x     = torch.arange(500)
    
    data["Compound", "treats", "Disease"].edge_index = torch.randint(0, 50, (2, 200))
    data["Compound", "targets", "Gene"].edge_index = torch.randint(0, 100, (2, 500))
    data["Gene", "interacts", "Gene"].edge_index = torch.randint(0, 500, (2, 1000))
    
    # Crear modelo
    model = create_model(data, config, encoder_type="rgcn", decoder_type="distmult")
    print(f"Modelo creado: {type(model).__name__}")
    print(f"Encoder: {model.encoder_type}")
    print(f"Decoder: {model.decoder_type}")
    
    # Forward pass
    edge_label_index = torch.randint(0, 50, (2, 20))
    scores = model(data, edge_label_index, src_type="Compound", dst_type="Disease")
    print(f"Scores shape: {scores.shape}")
    
    # Loss
    labels = torch.randint(0, 2, (20,))
    loss_fn = LinkPredictionLoss()
    loss = loss_fn(scores, labels)
    print(f"Loss: {loss.item():.4f}")
