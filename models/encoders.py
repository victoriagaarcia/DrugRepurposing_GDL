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

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, RGCNConv, SAGEConv


class RGCNEncoder(nn.Module):
    """
    R-GCN: Relational Graph Convolutional Network
    (Schlichtkrull et al., 2018)

    IMPLEMENTACIÓN:
    ---------------
    Para preservar la idea de R-GCN real, convertimos el grafo heterogéneo
    (x_dict, edge_index_dict) en una representación homogénea temporal:
    - concatenamos todos los nodos tras proyectarlos a hidden_dim
    - convertimos cada tipo de arista en un relation_id
    - aplicamos RGCNConv sobre el grafo homogéneo con edge_type
    - separamos de nuevo el tensor resultante en un h_dict por tipo de nodo

    Así mantenemos la interfaz del resto del proyecto:
        encoder(x_dict, edge_index_dict) -> h_dict
    """

    def __init__(
        self,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        node_counts: Dict[str, int] = None,
        hidden_dim: int = 128,
        out_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_bases: int = 30,
    ):
        super().__init__()

        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.relation_to_id = {edge_type: i for i, edge_type in enumerate(edge_types)}
        self.num_relations = len(edge_types)

        # OLD: Fixed random feature projection (expects 2D float tensors)
        # self.input_proj = nn.ModuleDict(
        #     {node_type: Linear(-1, hidden_dim) for node_type in node_types}
        # )

        # NEW: Learnable node embeddings (expects 1D index tensors produced by data_loader)
        self.node_embeddings = nn.ModuleDict(
            {node_type: nn.Embedding(node_counts[node_type], hidden_dim)
             for node_type in node_types if node_type in node_counts}
        ) if node_counts is not None else None

        # Capas R-GCN reales
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = hidden_dim
            out_channels = out_dim if i == num_layers - 1 else hidden_dim

            effective_num_bases = min(num_bases, self.num_relations) if self.num_relations > 0 else None

            self.convs.append(
                RGCNConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    num_relations=self.num_relations,
                    num_bases=effective_num_bases,
                )
            )

        # Normalización por tipo de nodo y capa
        self.norms = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        node_type: nn.LayerNorm(hidden_dim if i < num_layers - 1 else out_dim)
                        for node_type in node_types
                    }
                )
                for i in range(num_layers)
            ]
        )

    # OLD: Fixed input projection (expects 2D float tensors)
    # def _project_inputs(self, x_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    #     h_dict = {}
    #     for node_type, x in x_dict.items():
    #         if node_type in self.input_proj:
    #             h_dict[node_type] = self.input_proj[node_type](x)
    #     return h_dict

    # NEW: Learnable embeddings — x is a 1D LongTensor of node indices
    def _project_inputs(self, x_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        h_dict = {}
        for node_type, x in x_dict.items():
            if self.node_embeddings is not None and node_type in self.node_embeddings:
                h_dict[node_type] = self.node_embeddings[node_type](x)
        return h_dict

    def _to_homogeneous(
        self,
        h_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ):
        """
        Convierte h_dict + edge_index_dict a:
        - x_all: [N_total, d]
        - edge_index_all: [2, E_total]
        - edge_type_all: [E_total]
        - offsets/counts para reconstruir por tipo
        """
        present_node_types = [nt for nt in self.node_types if nt in h_dict]
        if len(present_node_types) == 0:
            raise ValueError("No hay tipos de nodo presentes en h_dict.")

        offsets = {}
        counts = {}
        x_parts = []

        start = 0
        for node_type in present_node_types:
            x = h_dict[node_type]
            offsets[node_type] = start
            counts[node_type] = x.size(0)
            x_parts.append(x)
            start += x.size(0)

        x_all = torch.cat(x_parts, dim=0)
        device = x_all.device

        edge_index_parts = []
        edge_type_parts = []

        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type

            if src_type not in offsets or dst_type not in offsets:
                continue

            rel_id = self.relation_to_id[edge_type]

            src_offset = offsets[src_type]
            dst_offset = offsets[dst_type]

            edge_index_global = edge_index.clone().to(device)
            edge_index_global[0] += src_offset
            edge_index_global[1] += dst_offset

            edge_index_parts.append(edge_index_global)
            edge_type_parts.append(
                torch.full(
                    (edge_index_global.size(1),),
                    rel_id,
                    dtype=torch.long,
                    device=device,
                )
            )

        if len(edge_index_parts) == 0:
            edge_index_all = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_type_all = torch.empty((0,), dtype=torch.long, device=device)
        else:
            edge_index_all = torch.cat(edge_index_parts, dim=1)
            edge_type_all = torch.cat(edge_type_parts, dim=0)

        return x_all, edge_index_all, edge_type_all, present_node_types, offsets, counts

    def _to_heterogeneous(
        self,
        x_all: torch.Tensor,
        present_node_types: List[str],
        offsets: Dict[str, int],
        counts: Dict[str, int],
    ) -> Dict[str, torch.Tensor]:
        h_dict = {}
        for node_type in present_node_types:
            start = offsets[node_type]
            end = start + counts[node_type]
            h_dict[node_type] = x_all[start:end]
        return h_dict

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        # Proyección inicial
        h_dict = self._project_inputs(x_dict)

        # Mantener orden y offsets fijos entre capas
        x_all, edge_index_all, edge_type_all, present_node_types, offsets, counts = self._to_homogeneous(
            h_dict, edge_index_dict
        )

        # Capas R-GCN
        for i, conv in enumerate(self.convs):
            x_all = conv(x_all, edge_index_all, edge_type_all)

            # Reconstruir por tipo para normalizar/activar
            h_dict = self._to_heterogeneous(x_all, present_node_types, offsets, counts)

            for node_type in present_node_types:
                h = h_dict[node_type]

                if node_type in self.norms[i]:
                    h = self.norms[i][node_type](h)

                if i < self.num_layers - 1:
                    h = F.relu(h)
                    h = F.dropout(h, p=self.dropout, training=self.training)

                h_dict[node_type] = h

            # Volver a concatenar para la siguiente capa
            x_all = torch.cat([h_dict[node_type] for node_type in present_node_types], dim=0)

        return h_dict


class HANEncoder(nn.Module):
    """
    HAN: Heterogeneous Graph Attention Network
    (Wang et al., 2019)

    IMPLEMENTACIÓN SIMPLIFICADA:
    ----------------------------
    Usamos GATConv dentro de HeteroConv para capturar atención a nivel de nodo.
    La agregación entre relaciones se hace por suma, manteniendo el espíritu
    del encoder heterogéneo sin introducir una implementación completa de HAN.
    """

    def __init__(
        self,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        node_counts: Dict[str, int] = None,
        hidden_dim: int = 128,
        out_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        # OLD: Fixed input projection (expects 2D float tensors)
        # self.input_proj = nn.ModuleDict(
        #     {node_type: Linear(-1, hidden_dim) for node_type in node_types}
        # )

        # NEW: Learnable node embeddings (expects 1D index tensors from data_loader)
        self.node_embeddings = nn.ModuleDict(
            {node_type: nn.Embedding(node_counts[node_type], hidden_dim)
             for node_type in node_types if node_type in node_counts}
        ) if node_counts is not None else None

        self.convs = nn.ModuleList()

        for i in range(num_layers):
            in_channels = hidden_dim

            if i == num_layers - 1:
                out_channels = out_dim
                heads = 1
                concat = False
            else:
                out_channels = hidden_dim // num_heads
                heads = num_heads
                concat = True

            # OLD: single HeteroConv sums all edge-type messages with equal weight
            # conv_dict = {}
            # for edge_type in edge_types:
            #     conv_dict[edge_type] = GATConv(
            #         (in_channels, in_channels),
            #         out_channels,
            #         heads=heads,
            #         concat=concat,
            #         dropout=dropout,
            #         add_self_loops=False,
            #     )
            # self.convs.append(HeteroConv(conv_dict, aggr="sum"))

            # NEW: one GATConv per edge type stored in a ModuleDict so that
            # semantic_attention can scale each relation's output before summation
            per_edge_convs = nn.ModuleDict()
            for edge_type in edge_types:
                key = f"{edge_type[0]}_{edge_type[1]}_{edge_type[2]}"
                per_edge_convs[key] = GATConv(
                    (in_channels, in_channels),
                    out_channels,
                    heads=heads,
                    concat=concat,
                    dropout=dropout,
                    add_self_loops=False,
                )
            self.convs.append(per_edge_convs)

        self.norms = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        node_type: nn.LayerNorm(hidden_dim if i < num_layers - 1 else out_dim)
                        for node_type in node_types
                    }
                )
                for i in range(num_layers)
            ]
        )

        self.semantic_attention = nn.ParameterDict(
            {f"{et[0]}_{et[1]}_{et[2]}": nn.Parameter(torch.ones(1)) for et in edge_types}
        )

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        # OLD: fixed projection + HeteroConv sum (semantic_attention created but never used)
        # h_dict = {}
        # for node_type, x in x_dict.items():
        #     if node_type in self.input_proj:
        #         h_dict[node_type] = self.input_proj[node_type](x)
        #
        # for i, conv in enumerate(self.convs):
        #     h_dict_new = conv(h_dict, edge_index_dict)
        #     for node_type in h_dict:
        #         if node_type not in h_dict_new:
        #             h_dict_new[node_type] = h_dict[node_type]
        #     h_dict = h_dict_new
        #     for node_type in h_dict:
        #         h = h_dict[node_type]
        #         if node_type in self.norms[i]:
        #             h = self.norms[i][node_type](h)
        #         if i < self.num_layers - 1:
        #             h = F.elu(h)
        #             h = F.dropout(h, p=self.dropout, training=self.training)
        #         h_dict[node_type] = h
        # return h_dict

        # NEW: learnable embeddings + per-edge GATConv weighted by semantic attention
        h_dict = {}
        for node_type, x in x_dict.items():
            if self.node_embeddings is not None and node_type in self.node_embeddings:
                h_dict[node_type] = self.node_embeddings[node_type](x)

        for i, per_edge_convs in enumerate(self.convs):
            # Collect weighted contributions per destination node type
            contributions: Dict[str, list] = {}

            for edge_type in self.edge_types:
                src_type, rel, dst_type = edge_type
                key = f"{src_type}_{rel}_{dst_type}"

                if key not in per_edge_convs:
                    continue
                if edge_type not in edge_index_dict:
                    continue
                if src_type not in h_dict:
                    continue

                src_h = h_dict[src_type]
                dst_h = h_dict.get(dst_type, src_h)

                out = per_edge_convs[key]((src_h, dst_h), edge_index_dict[edge_type])
                # Scale by learned scalar attention weight for this relation
                w = torch.sigmoid(self.semantic_attention[key])
                weighted_out = w * out

                if dst_type not in contributions:
                    contributions[dst_type] = []
                contributions[dst_type].append(weighted_out)

            # Sum weighted contributions; fall back to previous embedding if no messages
            h_dict_new = {}
            for node_type in h_dict:
                if node_type in contributions and contributions[node_type]:
                    h_dict_new[node_type] = torch.stack(contributions[node_type]).sum(dim=0)
                else:
                    h_dict_new[node_type] = h_dict[node_type]

            h_dict = h_dict_new

            for node_type in h_dict:
                h = h_dict[node_type]
                if node_type in self.norms[i]:
                    h = self.norms[i][node_type](h)
                if i < self.num_layers - 1:
                    h = F.elu(h)
                    h = F.dropout(h, p=self.dropout, training=self.training)
                h_dict[node_type] = h

        return h_dict


class HeteroGraphSAGEEncoder(nn.Module):
    """
    GraphSAGE Heterogéneo
    (Hamilton et al., 2017, adaptado para grafos heterogéneos)
    """

    def __init__(
        self,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        node_counts: Dict[str, int] = None,
        hidden_dim: int = 128,
        out_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        aggregator: str = "mean",
    ):
        super().__init__()

        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.aggregator = aggregator

        # OLD: Fixed input projection (expects 2D float tensors)
        # self.input_proj = nn.ModuleDict(
        #     {node_type: Linear(-1, hidden_dim) for node_type in node_types}
        # )

        # NEW: Learnable node embeddings (expects 1D index tensors from data_loader)
        self.node_embeddings = nn.ModuleDict(
            {node_type: nn.Embedding(node_counts[node_type], hidden_dim)
             for node_type in node_types if node_type in node_counts}
        ) if node_counts is not None else None

        self.convs = nn.ModuleList()

        for i in range(num_layers):
            in_channels = hidden_dim
            out_channels = out_dim if i == num_layers - 1 else hidden_dim

            conv_dict = {}
            for edge_type in edge_types:
                conv_dict[edge_type] = SAGEConv(
                    (in_channels, in_channels),
                    out_channels,
                    aggr=aggregator,
                    normalize=True,
                    root_weight=True,
                )

            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        self.norms = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        node_type: nn.LayerNorm(hidden_dim if i < num_layers - 1 else out_dim)
                        for node_type in node_types
                    }
                )
                for i in range(num_layers)
            ]
        )

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        # OLD: fixed input projection (expects 2D float tensors)
        # h_dict = {}
        # for node_type, x in x_dict.items():
        #     if node_type in self.input_proj:
        #         h_dict[node_type] = self.input_proj[node_type](x)

        # NEW: learnable node embeddings (x is a 1D index tensor)
        h_dict = {}
        for node_type, x in x_dict.items():
            if self.node_embeddings is not None and node_type in self.node_embeddings:
                h_dict[node_type] = self.node_embeddings[node_type](x)

        for i, conv in enumerate(self.convs):
            h_dict_new = conv(h_dict, edge_index_dict)

            # Mantener tipos de nodo sin mensajes entrantes
            for node_type in h_dict:
                if node_type not in h_dict_new:
                    h_dict_new[node_type] = h_dict[node_type]

            h_dict = h_dict_new

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
    config,
    node_counts: Dict[str, int] = None,
) -> nn.Module:
    """
    Factory function para crear encoders.

    Args:
        encoder_type: "rgcn", "han", o "sage"
        node_types: Lista de tipos de nodo
        edge_types: Lista de tipos de arista
        config: Objeto de configuración
        node_counts: Número de nodos por tipo, requerido para nn.Embedding learnable

    Returns:
        Instancia del encoder apropiado
    """
    encoder_type = encoder_type.lower()

    if encoder_type == "rgcn":
        return RGCNEncoder(
            node_types=node_types,
            edge_types=edge_types,
            node_counts=node_counts,
            hidden_dim=config.model.hidden_dim,
            out_dim=config.model.out_dim,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
            num_bases=config.model.num_bases,
        )
    elif encoder_type == "han":
        return HANEncoder(
            node_types=node_types,
            edge_types=edge_types,
            node_counts=node_counts,
            hidden_dim=config.model.hidden_dim,
            out_dim=config.model.out_dim,
            num_layers=config.model.num_layers,
            num_heads=config.model.num_heads,
            dropout=config.model.dropout,
        )
    elif encoder_type == "sage":
        return HeteroGraphSAGEEncoder(
            node_types=node_types,
            edge_types=edge_types,
            node_counts=node_counts,
            hidden_dim=config.model.hidden_dim,
            out_dim=config.model.out_dim,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
            aggregator=config.model.sage_aggregator,
        )
    else:
        raise ValueError(
            f"Encoder desconocido: {encoder_type}. "
            f"Opciones: rgcn, han, sage"
        )