"""
Paquete de modelos para Drug Repurposing con GNNs.

Contiene:
- encoders: R-GCN, HAN, GraphSAGE heterogéneo
- decoders: DistMult, Dot Product, MLP
- full_model: Modelo completo encoder-decoder
"""

from models.encoders import (
    RGCNEncoder,
    HANEncoder,
    HeteroGraphSAGEEncoder,
    get_encoder
)

from models.decoders import (
    DotProductDecoder,
    DistMultDecoder,
    MLPDecoder,
    get_decoder
)

from models.full_model import (
    DrugRepurposingModel,
    LinkPredictionLoss,
    create_model
)

__all__ = [
    # Encoders
    "RGCNEncoder",
    "HANEncoder", 
    "HeteroGraphSAGEEncoder",
    "get_encoder",
    # Decoders
    "DotProductDecoder",
    "DistMultDecoder",
    "MLPDecoder",
    "get_decoder",
    # Full model
    "DrugRepurposingModel",
    "LinkPredictionLoss",
    "create_model",
]
