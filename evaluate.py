"""
================================================================================
EVALUATE.PY - Métricas de Evaluación para Link Prediction
================================================================================

Este módulo implementa las métricas estándar para evaluar modelos de
link prediction en Knowledge Graphs:

1. Hits@K: Proporción de positivos rankeados en top-K
2. MRR (Mean Reciprocal Rank): Media del inverso del ranking
3. AUC-ROC: Área bajo la curva ROC

================================================================================
BASE TEÓRICA - EVALUACIÓN DE LINK PREDICTION:
================================================================================

En link prediction, evaluamos la capacidad del modelo de rankear aristas
verdaderas por encima de aristas falsas.

SETUP DE EVALUACIÓN:
1. Para cada arista positiva (s, r, o) en test:
   - Generar candidatos negativos: (s, r, o') para todos los o' posibles
   - Calcular scores para positivo y todos los negativos
   - Rankear el positivo entre los negativos

MÉTRICAS:

1. HITS@K (también llamado Recall@K):
   - ¿Está el positivo entre los top-K candidatos?
   - Hits@1: ¿Es el positivo el mejor candidato?
   - Hits@10: ¿Está en los 10 mejores?
   
   Interpretación práctica:
   - Si un médico revisa los top-10 fármacos sugeridos para una enfermedad,
     Hits@10 indica con qué frecuencia encontrará uno efectivo.

2. MRR (Mean Reciprocal Rank):
   - Si el positivo está en posición k, su contribución es 1/k
   - MRR = media de 1/rank sobre todos los positivos
   
   Propiedades:
   - Rango: [0, 1], mayor es mejor
   - Premia más a las predicciones en top posiciones
   - MRR=1: todos los positivos en posición 1
   - MRR=0.5: positivos en posición 2 en promedio

3. AUC-ROC:
   - Probabilidad de que un positivo aleatorio tenga score mayor que
     un negativo aleatorio
   - AUC=0.5: modelo aleatorio
   - AUC=1.0: modelo perfecto

EVALUACIÓN FILTERED vs RAW:
- Raw: rankear contra todos los candidatos
- Filtered: excluir otros positivos del ranking

Ejemplo:
- Arista test: (drug_A, treats, disease_X)
- Arista train: (drug_A, treats, disease_Y)
- En raw: disease_Y compite con disease_X en el ranking
- En filtered: disease_Y se excluye

Filtered es más justo porque no penaliza al modelo por predecir
aristas que realmente existen (aunque sean de train).

================================================================================
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.data import HeteroData
from tqdm import tqdm


class LinkPredictionEvaluator:
    """
    Evaluador de métricas para link prediction.
    
    Calcula métricas de ranking para evaluar la calidad de las predicciones
    del modelo en el task de drug repurposing.
    """
    
    def __init__(
        self,
        hits_k_values: List[int] = [1, 3, 10, 50, 100],
        filtered: bool = True
    ):
        """
        Inicializa el evaluador.
        
        Args:
            hits_k_values: Valores de K para calcular Hits@K
            filtered: Si True, usa evaluación filtered
        """
        self.hits_k_values = hits_k_values
        self.filtered = filtered
    
    def compute_ranking_metrics(
        self,
        positive_scores: torch.Tensor,
        negative_scores: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Calcula métricas de ranking para un conjunto de predicciones.
        
        Para cada arista positiva, comparamos su score con los de negativos
        y calculamos su posición en el ranking.
        
        ALGORITMO:
        1. Para cada positivo i:
           a. Contar cuántos negativos tienen score >= score_positivo
           b. rank_i = 1 + count (1-indexed)
        2. Calcular métricas agregadas sobre todos los ranks
        
        Args:
            positive_scores: Scores de aristas positivas [num_pos]
            negative_scores: Scores de aristas negativas [num_neg] o [num_pos, num_neg_per_pos]
            
        Returns:
            Diccionario con métricas
        """
        positive_scores = positive_scores.detach().cpu().numpy()
        negative_scores = negative_scores.detach().cpu().numpy()
        
        # Si negative_scores es 2D, cada fila corresponde a un positivo
        if len(negative_scores.shape) == 2:
            # Caso: num_neg diferentes por cada positivo
            ranks = []
            for i, pos_score in enumerate(positive_scores):
                neg_scores = negative_scores[i]
                # Rank = 1 + número de negativos con score >= positivo
                rank = 1 + np.sum(neg_scores >= pos_score)
                ranks.append(rank)
            ranks = np.array(ranks)
        else:
            # Caso: todos los positivos comparten los mismos negativos
            # Expandir para comparar cada positivo con todos los negativos
            # positive_scores: [num_pos], negative_scores: [num_neg]
            # Comparar: [num_pos, 1] vs [1, num_neg]
            comparison = negative_scores[None, :] >= positive_scores[:, None]
            ranks = 1 + comparison.sum(axis=1)
        
        # Calcular métricas
        metrics = {}
        
        # MRR (Mean Reciprocal Rank)
        mrr = np.mean(1.0 / ranks)
        metrics['MRR'] = float(mrr)
        
        # Mean Rank (para referencia)
        metrics['Mean_Rank'] = float(np.mean(ranks))
        
        # Hits@K
        for k in self.hits_k_values:
            hits_at_k = np.mean(ranks <= k)
            metrics[f'Hits@{k}'] = float(hits_at_k)
        
        return metrics
    
    def compute_classification_metrics(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calcula métricas de clasificación binaria.
        
        Estas métricas tratan link prediction como clasificación binaria
        (existe/no existe) en lugar de ranking.
        
        Args:
            scores: Scores del modelo [num_samples]
            labels: Labels verdaderos [num_samples] (1=positivo, 0=negativo)
            
        Returns:
            Diccionario con métricas
        """
        scores = scores.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        
        metrics = {}
        
        # AUC-ROC
        try:
            auc_roc = roc_auc_score(labels, scores)
            metrics['AUC-ROC'] = float(auc_roc)
        except ValueError:
            # Puede fallar si solo hay una clase
            metrics['AUC-ROC'] = 0.5
        
        # Average Precision (AUC-PR)
        try:
            ap = average_precision_score(labels, scores)
            metrics['AP'] = float(ap)
        except ValueError:
            metrics['AP'] = 0.0
        
        return metrics
    
    def evaluate(
        self,
        model,
        data: HeteroData,
        edge_label_index: torch.Tensor,
        edge_label: torch.Tensor,
        src_type: str = "Compound",
        dst_type: str = "Disease",
        batch_size: int = 1024,
        existing_edges: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Evaluación completa del modelo.
        
        PROCESO:
        1. Obtener embeddings de todos los nodos
        2. Para cada batch de aristas de test:
           a. Calcular scores de positivos
           b. Generar y puntuar negativos
           c. Calcular rankings
        3. Agregar métricas
        
        Args:
            model: Modelo a evaluar
            data: Datos del grafo
            edge_label_index: Índices de aristas [2, num_edges]
            edge_label: Labels de aristas [num_edges]
            src_type: Tipo de nodo fuente
            dst_type: Tipo de nodo destino
            batch_size: Tamaño del batch para evaluación
            existing_edges: Aristas existentes para filtrar (opcional)
            
        Returns:
            Diccionario con todas las métricas
        """
        model.eval()
        device = next(model.parameters()).device
        
        # Mover datos a device
        data = data.to(device)
        edge_label_index = edge_label_index.to(device)
        edge_label = edge_label.to(device)
        
        # Separar positivos y negativos
        pos_mask = edge_label == 1
        neg_mask = edge_label == 0
        
        pos_edge_index = edge_label_index[:, pos_mask]
        neg_edge_index = edge_label_index[:, neg_mask]
        
        with torch.no_grad():
            # Obtener embeddings
            h_dict = model.get_embeddings(data)
            h_src = h_dict[src_type]
            h_dst = h_dict[dst_type]
            
            # Calcular scores para positivos y negativos
            all_scores = []
            all_labels = []
            
            num_edges = edge_label_index.size(1)
            for start in range(0, num_edges, batch_size):
                end = min(start + batch_size, num_edges)
                batch_edge_index = edge_label_index[:, start:end]
                batch_labels = edge_label[start:end]
                
                # Embeddings del batch
                batch_h_src = h_src[batch_edge_index[0]]
                batch_h_dst = h_dst[batch_edge_index[1]]
                
                # Scores
                batch_scores = model.decode(batch_h_src, batch_h_dst)
                
                all_scores.append(batch_scores)
                all_labels.append(batch_labels)
            
            all_scores = torch.cat(all_scores)
            all_labels = torch.cat(all_labels)
            
            # Métricas de clasificación
            metrics = self.compute_classification_metrics(all_scores, all_labels)
            
            # Métricas de ranking
            pos_scores = all_scores[pos_mask]
            neg_scores = all_scores[neg_mask]
            
            if len(pos_scores) > 0 and len(neg_scores) > 0:
                ranking_metrics = self.compute_ranking_metrics(pos_scores, neg_scores)
                metrics.update(ranking_metrics)
        
        return metrics


def evaluate_full_ranking(
    model,
    data: HeteroData,
    test_edges: torch.Tensor,
    src_type: str = "Compound",
    dst_type: str = "Disease",
    train_edges: Optional[torch.Tensor] = None,
    filtered: bool = True
) -> Dict[str, float]:
    """
    Evaluación con ranking completo contra todos los candidatos.
    
    Para cada arista positiva (src, dst) en test:
    1. Mantener src fijo
    2. Rankear dst contra TODOS los nodos de tipo dst
    3. Calcular posición del verdadero dst
    
    Esta es la evaluación más realista pero también más costosa.
    
    Args:
        model: Modelo a evaluar
        data: Datos del grafo
        test_edges: Aristas de test [2, num_test]
        src_type: Tipo de nodo fuente
        dst_type: Tipo de nodo destino
        train_edges: Aristas de train para filtrar
        filtered: Si True, excluir aristas de train del ranking
        
    Returns:
        Diccionario con métricas
    """
    model.eval()
    device = next(model.parameters()).device
    
    data = data.to(device)
    test_edges = test_edges.to(device)
    
    # Crear set de aristas de train para filtrado
    if filtered and train_edges is not None:
        train_set = set()
        train_edges_cpu = train_edges.cpu().numpy()
        for i in range(train_edges.size(1)):
            train_set.add((train_edges_cpu[0, i], train_edges_cpu[1, i]))
    else:
        train_set = set()
    
    with torch.no_grad():
        # Obtener todos los embeddings
        h_dict = model.get_embeddings(data)
        h_src_all = h_dict[src_type]  # [num_src, d]
        h_dst_all = h_dict[dst_type]  # [num_dst, d]
        
        num_dst = h_dst_all.size(0)
        ranks = []
        
        # Evaluar cada arista de test
        for i in tqdm(range(test_edges.size(1)), desc="Ranking evaluation"):
            src_idx = test_edges[0, i].item()
            true_dst_idx = test_edges[1, i].item()
            
            # Embedding del source
            h_src = h_src_all[src_idx:src_idx+1]  # [1, d]
            
            # Scores contra todos los destinos
            # [1, d] @ [d, num_dst] = [1, num_dst]
            all_scores = model.decoder.forward_all(h_src, h_dst_all).squeeze(0)
            
            # Score del verdadero destino
            true_score = all_scores[true_dst_idx].item()
            
            # Filtrar aristas de train si es necesario
            if filtered:
                # Mask para excluir aristas conocidas
                mask = torch.ones(num_dst, dtype=torch.bool, device=device)
                for dst_idx in range(num_dst):
                    if (src_idx, dst_idx) in train_set and dst_idx != true_dst_idx:
                        mask[dst_idx] = False
                
                filtered_scores = all_scores[mask]
                # Rank entre los scores filtrados
                rank = 1 + (filtered_scores > true_score).sum().item()
            else:
                # Rank = 1 + número de scores mayores
                rank = 1 + (all_scores > true_score).sum().item()
            
            ranks.append(rank)
        
        ranks = np.array(ranks)
        
        # Calcular métricas
        metrics = {
            'MRR': float(np.mean(1.0 / ranks)),
            'Mean_Rank': float(np.mean(ranks)),
        }
        
        for k in [1, 3, 10, 50, 100]:
            metrics[f'Hits@{k}'] = float(np.mean(ranks <= k))
    
    return metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    """
    Formatea métricas para impresión legible.
    
    Args:
        metrics: Diccionario de métricas
        
    Returns:
        String formateado
    """
    lines = []
    
    # Ordenar por tipo de métrica
    ranking_metrics = ['MRR', 'Mean_Rank'] + [k for k in metrics if k.startswith('Hits@')]
    classification_metrics = ['AUC-ROC', 'AP']
    
    lines.append("Ranking Metrics:")
    for key in ranking_metrics:
        if key in metrics:
            if key == 'Mean_Rank':
                lines.append(f"  {key}: {metrics[key]:.1f}")
            else:
                lines.append(f"  {key}: {metrics[key]:.4f}")
    
    lines.append("\nClassification Metrics:")
    for key in classification_metrics:
        if key in metrics:
            lines.append(f"  {key}: {metrics[key]:.4f}")
    
    return '\n'.join(lines)


if __name__ == "__main__":
    # Test del evaluador
    print("Testing LinkPredictionEvaluator...")
    
    # Scores simulados
    pos_scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])
    neg_scores = torch.tensor([0.4, 0.3, 0.2, 0.1, 0.05])
    
    evaluator = LinkPredictionEvaluator(hits_k_values=[1, 3, 5])
    
    # Test ranking metrics
    metrics = evaluator.compute_ranking_metrics(pos_scores, neg_scores)
    print("\nRanking metrics (todos positivos deberían estar en top-5):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Test classification metrics
    all_scores = torch.cat([pos_scores, neg_scores])
    all_labels = torch.cat([torch.ones(5), torch.zeros(5)])
    
    class_metrics = evaluator.compute_classification_metrics(all_scores, all_labels)
    print("\nClassification metrics:")
    for k, v in class_metrics.items():
        print(f"  {k}: {v:.4f}")
