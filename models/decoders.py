"""
================================================================================
DECODERS.PY - Decoders para Link Prediction en Knowledge Graphs
================================================================================

En el framework encoder-decoder para link prediction:
- El ENCODER genera embeddings de nodos (h_drug, h_disease, etc.)
- El DECODER puntúa la probabilidad de que exista una arista entre dos nodos

Este archivo implementa dos decoders comunes:
1. Dot Product (producto punto)
2. DistMult (tensor factorization bilinear)

================================================================================
BASE TEÓRICA - LINK PREDICTION EN KNOWLEDGE GRAPHS:
================================================================================

Dado un Knowledge Graph G = (V, E) con tripletas (s, r, o) donde:
- s: sujeto (nodo fuente)
- r: relación (tipo de arista)
- o: objeto (nodo destino)

El objetivo es aprender una función de scoring:

    f(s, r, o) → ℝ

tal que:
- f(s, r, o) alto si la tripleta es verdadera
- f(s, r, o) bajo si es falsa

Diferentes decoders definen f() de distintas formas.

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DotProductDecoder(nn.Module):
    """
    Decoder basado en Producto Punto.
    
    ============================================================================
    BASE TEÓRICA:
    ============================================================================
    
    El producto punto es la forma más simple de medir similitud entre embeddings:
    
        f(s, o) = h_s · h_o = Σ_i h_s[i] * h_o[i]
    
    INTERPRETACIÓN GEOMÉTRICA:
    - El producto punto mide la proyección de un vector sobre otro
    - Si h_s y h_o apuntan en la misma dirección → f alto (similares)
    - Si son ortogonales → f = 0 (no relacionados)
    - Si apuntan en direcciones opuestas → f negativo
    
    VENTAJAS:
    - Muy eficiente computacionalmente: O(d) donde d es la dimensión
    - No añade parámetros adicionales
    - Funciona bien cuando las relaciones son simétricas
    
    DESVENTAJAS:
    - No modela la relación r explícitamente
    - Asume que la misma similitud aplica para todas las relaciones
    - Limitado para relaciones asimétricas (treats ≠ treated_by)
    
    VARIANTE CON PROYECCIÓN:
    Podemos añadir una capa de proyección para adaptar los embeddings
    antes del producto punto:
    
        f(s, o) = (W_s · h_s) · (W_o · h_o)
    
    Esto permite al modelo aprender transformaciones específicas.
    
    ============================================================================
    USO EN DRUG REPURPOSING:
    ============================================================================
    
    Para predecir (drug, treats, disease):
    - h_drug: embedding del fármaco
    - h_disease: embedding de la enfermedad
    - f(drug, disease) = h_drug · h_disease
    
    Intuitivamente, el modelo aprende a colocar:
    - Fármacos cerca de las enfermedades que tratan
    - Enfermedades similares cerca entre sí
    - Formando "clusters" de fármaco-enfermedad en el espacio latente
    """
    
    def __init__(
        self,
        embedding_dim: int,
        use_projection: bool = True,
        hidden_dim: Optional[int] = None
    ):
        """
        Inicializa el decoder de producto punto.
        
        Args:
            embedding_dim: Dimensión de los embeddings de entrada
            use_projection: Si True, añade capas de proyección
            hidden_dim: Dimensión de la capa oculta (si use_projection=True)
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.use_projection = use_projection
        
        if use_projection:
            hidden_dim = hidden_dim or embedding_dim
            
            # Proyecciones separadas para source y target
            # Permite modelar asimetría: drug → disease ≠ disease → drug
            self.proj_src = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embedding_dim)
            )
            self.proj_dst = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embedding_dim)
            )
    
    def forward(
        self,
        h_src: torch.Tensor,
        h_dst: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcula scores para pares de nodos.
        
        Args:
            h_src: Embeddings de nodos fuente [batch_size, embedding_dim]
            h_dst: Embeddings de nodos destino [batch_size, embedding_dim]
            
        Returns:
            Scores de probabilidad [batch_size]
        """
        if self.use_projection:
            h_src = self.proj_src(h_src)
            h_dst = self.proj_dst(h_dst)
        
        # Producto punto element-wise seguido de suma
        # Equivalente a: (h_src * h_dst).sum(dim=-1)
        scores = (h_src * h_dst).sum(dim=-1)
        
        return scores
    
    def forward_all(
        self,
        h_src: torch.Tensor,
        h_dst: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcula scores para TODOS los pares posibles.
        
        Útil para ranking durante evaluación.
        
        Args:
            h_src: Embeddings fuente [num_src, embedding_dim]
            h_dst: Embeddings destino [num_dst, embedding_dim]
            
        Returns:
            Matriz de scores [num_src, num_dst]
        """
        if self.use_projection:
            h_src = self.proj_src(h_src)
            h_dst = self.proj_dst(h_dst)
        
        # Multiplicación de matrices: [num_src, d] @ [d, num_dst] = [num_src, num_dst]
        scores = torch.matmul(h_src, h_dst.t())
        
        return scores


class DistMultDecoder(nn.Module):
    """
    DistMult: Diagonal Bilinear Model
    (Yang et al., 2015)
    
    ============================================================================
    BASE TEÓRICA:
    ============================================================================
    
    DistMult modela relaciones usando una matriz diagonal por relación:
    
        f(s, r, o) = h_s^T · diag(R_r) · h_o = Σ_i h_s[i] * R_r[i] * h_o[i]
    
    donde R_r es un vector de pesos específico para la relación r.
    
    INTERPRETACIÓN:
    - Cada dimensión del embedding captura un "factor latente"
    - R_r[i] pondera cuán importante es el factor i para la relación r
    - Si R_r[i] > 0: el factor i contribuye positivamente
    - Si R_r[i] < 0: el factor i contribuye negativamente
    - Si R_r[i] ≈ 0: el factor i es irrelevante para esta relación
    
    EJEMPLO INTUITIVO:
    Supongamos que los factores latentes capturan:
    - Dimensión 1: "actúa en sistema nervioso"
    - Dimensión 2: "es antiinflamatorio"
    - Dimensión 3: "tiene efectos cardiovasculares"
    
    Para la relación "treats_neurological_disease":
    - R_r = [alto, bajo, bajo]
    - Fármacos con alta dim 1 scorearán alto con enfermedades neurológicas
    
    COMPARACIÓN CON OTROS MODELOS:
    
    1. TransE: f(s,r,o) = ||h_s + r - h_o||
       - Modela relaciones como traslaciones
       - Bueno para relaciones 1-a-1
    
    2. ComplEx: f(s,r,o) = Re(h_s · diag(r) · conj(h_o))
       - Usa embeddings complejos
       - Captura relaciones asimétricas
    
    3. DistMult: f(s,r,o) = h_s · diag(r) · h_o
       - Bilinear con matriz diagonal
       - Simétrico: f(s,r,o) = f(o,r,s)
       - Eficiente pero limitado para relaciones asimétricas
    
    LIMITACIÓN DE SIMETRÍA:
    DistMult asume f(s,r,o) = f(o,r,s), lo cual NO es cierto para:
    - "treats" (drug→disease ≠ disease→drug)
    - "causes" (gene→disease ≠ disease→gene)
    
    SOLUCIÓN EN ESTE PROYECTO:
    Usamos embeddings diferentes para source y target del mismo tipo,
    rompiendo parcialmente la simetría.
    
    ============================================================================
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_relations: int = 1
    ):
        """
        Inicializa el decoder DistMult.
        
        Args:
            embedding_dim: Dimensión de los embeddings
            num_relations: Número de tipos de relación a modelar
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_relations = num_relations
        
        # Vector de relación (diagonal de la matriz R)
        # Inicializado con distribución uniforme en [-1, 1]
        self.relation_embedding = nn.Parameter(
            torch.empty(num_relations, embedding_dim)
        )
        nn.init.uniform_(self.relation_embedding, -1.0, 1.0)
        
        # Capas opcionales de transformación
        self.src_transform = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.dst_transform = nn.Linear(embedding_dim, embedding_dim, bias=False)
    
    def forward(
        self,
        h_src: torch.Tensor,
        h_dst: torch.Tensor,
        relation_idx: int = 0
    ) -> torch.Tensor:
        """
        Calcula scores DistMult para pares específicos.
        
        f(s, r, o) = Σ_i h_s[i] * R_r[i] * h_o[i]
        
        Args:
            h_src: Embeddings fuente [batch_size, embedding_dim]
            h_dst: Embeddings destino [batch_size, embedding_dim]
            relation_idx: Índice de la relación (default 0 para una sola relación)
            
        Returns:
            Scores [batch_size]
        """
        # Obtener embedding de relación
        rel = self.relation_embedding[relation_idx]  # [embedding_dim]
        
        # Transformar embeddings
        h_src = self.src_transform(h_src)
        h_dst = self.dst_transform(h_dst)
        
        # DistMult scoring: element-wise product con relación
        # h_src * rel * h_dst, sumado sobre dimensiones
        scores = (h_src * rel * h_dst).sum(dim=-1)
        
        return scores
    
    def forward_all(
        self,
        h_src: torch.Tensor,
        h_dst: torch.Tensor,
        relation_idx: int = 0
    ) -> torch.Tensor:
        """
        Calcula scores para todos los pares posibles.
        
        Implementación eficiente usando broadcasting.
        
        Args:
            h_src: Embeddings fuente [num_src, embedding_dim]
            h_dst: Embeddings destino [num_dst, embedding_dim]
            relation_idx: Índice de la relación
            
        Returns:
            Matriz de scores [num_src, num_dst]
        """
        rel = self.relation_embedding[relation_idx]  # [embedding_dim]
        
        # Transformar
        h_src = self.src_transform(h_src)  # [num_src, d]
        h_dst = self.dst_transform(h_dst)  # [num_dst, d]
        
        # Aplicar relación a source
        h_src_rel = h_src * rel  # [num_src, d]
        
        # Producto punto con todos los destinos
        # [num_src, d] @ [d, num_dst] = [num_src, num_dst]
        scores = torch.matmul(h_src_rel, h_dst.t())
        
        return scores


class MLPDecoder(nn.Module):
    """
    Decoder MLP (Multi-Layer Perceptron)
    
    ============================================================================
    BASE TEÓRICA:
    ============================================================================
    
    El decoder MLP es más expresivo que dot product o DistMult:
    
        f(s, o) = MLP([h_s || h_o])
    
    donde || denota concatenación.
    
    VENTAJAS:
    - Puede aprender cualquier función de scoring (universal approximator)
    - No asume simetría ni bilinearidad
    - Puede capturar interacciones complejas entre features
    
    DESVENTAJAS:
    - Más parámetros, más propenso a overfitting
    - Más lento de computar, especialmente para ranking de todos los pares
    - Menos interpretable que métodos bilineares
    
    USO TÍPICO:
    MLP se usa como baseline más poderoso para comparar con métodos
    más simples. Si DistMult iguala a MLP, preferimos DistMult por
    su eficiencia e interpretabilidad.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.3
    ):
        """
        Inicializa el decoder MLP.
        
        Args:
            embedding_dim: Dimensión de los embeddings de entrada
            hidden_dims: Dimensiones de las capas ocultas
            dropout: Tasa de dropout
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Input: concatenación de h_src y h_dst → 2 * embedding_dim
        layers = []
        in_dim = embedding_dim * 2
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        
        # Capa final: output escalar
        layers.append(nn.Linear(in_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(
        self,
        h_src: torch.Tensor,
        h_dst: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcula scores usando MLP.
        
        Args:
            h_src: Embeddings fuente [batch_size, embedding_dim]
            h_dst: Embeddings destino [batch_size, embedding_dim]
            
        Returns:
            Scores [batch_size]
        """
        # Concatenar embeddings
        h_concat = torch.cat([h_src, h_dst], dim=-1)  # [batch_size, 2*d]
        
        # Pasar por MLP
        scores = self.mlp(h_concat).squeeze(-1)  # [batch_size]
        
        return scores
    
    def forward_all(
        self,
        h_src: torch.Tensor,
        h_dst: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcula scores para todos los pares.
        
        NOTA: Esto es costoso para muchos nodos (O(n*m) forward passes).
        Para evaluación eficiente, considerar batching.
        
        Args:
            h_src: Embeddings fuente [num_src, embedding_dim]
            h_dst: Embeddings destino [num_dst, embedding_dim]
            
        Returns:
            Matriz de scores [num_src, num_dst]
        """
        num_src = h_src.size(0)
        num_dst = h_dst.size(0)
        
        # Expandir para todos los pares
        # h_src: [num_src, 1, d] → [num_src, num_dst, d]
        # h_dst: [1, num_dst, d] → [num_src, num_dst, d]
        h_src_exp = h_src.unsqueeze(1).expand(-1, num_dst, -1)
        h_dst_exp = h_dst.unsqueeze(0).expand(num_src, -1, -1)
        
        # Concatenar
        h_concat = torch.cat([h_src_exp, h_dst_exp], dim=-1)
        # [num_src, num_dst, 2*d]
        
        # Flatten, pasar por MLP, reshape
        h_flat = h_concat.view(-1, self.embedding_dim * 2)
        scores_flat = self.mlp(h_flat).squeeze(-1)
        scores = scores_flat.view(num_src, num_dst)
        
        return scores


def get_decoder(
    decoder_type: str,
    embedding_dim: int,
    **kwargs
) -> nn.Module:
    """
    Factory function para crear decoders.
    
    Args:
        decoder_type: "dot", "distmult", o "mlp"
        embedding_dim: Dimensión de los embeddings
        **kwargs: Argumentos adicionales específicos del decoder
        
    Returns:
        Instancia del decoder apropiado
    """
    decoder_type = decoder_type.lower()
    
    if decoder_type == "dot":
        return DotProductDecoder(
            embedding_dim=embedding_dim,
            use_projection=kwargs.get("use_projection", True),
            hidden_dim=kwargs.get("hidden_dim", None)
        )
    elif decoder_type == "distmult":
        return DistMultDecoder(
            embedding_dim=embedding_dim,
            num_relations=kwargs.get("num_relations", 1)
        )
    elif decoder_type == "mlp":
        return MLPDecoder(
            embedding_dim=embedding_dim,
            hidden_dims=kwargs.get("hidden_dims", (256, 128, 64)),
            dropout=kwargs.get("dropout", 0.3)
        )
    else:
        raise ValueError(f"Decoder desconocido: {decoder_type}. "
                        f"Opciones: dot, distmult, mlp")
