"""
================================================================================
TRAIN.PY - Loop de Entrenamiento para Drug Repurposing
================================================================================

Este módulo implementa el pipeline de entrenamiento completo:
1. Preparación de datos (batching, negative sampling)
2. Loop de entrenamiento con validación
3. Early stopping y learning rate scheduling
4. Checkpointing del mejor modelo

================================================================================
BASE TEÓRICA - ENTRENAMIENTO DE GNNS PARA LINK PREDICTION:
================================================================================

NEGATIVE SAMPLING:
En link prediction, solo tenemos ejemplos positivos (aristas que existen).
Necesitamos generar ejemplos negativos para entrenamiento supervisado.

Estrategias de negative sampling:
1. Random: muestrear pares aleatorios que no existen
2. Corrupted: corromper aristas positivas cambiando src o dst
3. Hard negatives: muestrear negativos que son "difíciles" (cercanos a positivos)

MINI-BATCH TRAINING:
Para grafos grandes, no podemos procesar todo el grafo en memoria.
Estrategias:
1. Edge sampling: muestrear aristas para cada batch
2. Node sampling: muestrear subgrafos (como en GraphSAGE)
3. Cluster sampling: particionar el grafo en clusters

REGULARIZACIÓN:
1. Dropout: ya incluido en las capas GNN
2. Weight decay: L2 regularization de parámetros
3. Early stopping: detener si la validación no mejora
4. Label smoothing: suavizar labels para evitar overconfidence

================================================================================
"""

import os
import time
from typing import Dict, Tuple, Optional
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import HeteroData
import numpy as np

from config import Config, get_config
from models.full_model import DrugRepurposingModel, LinkPredictionLoss, create_model
from evaluate import LinkPredictionEvaluator, format_metrics


class Trainer:
    """
    Entrenador para modelos de Drug Repurposing.
    
    Implementa:
    - Loop de entrenamiento con negative sampling
    - Validación periódica
    - Early stopping
    - Learning rate scheduling
    - Checkpointing
    """
    
    def __init__(
        self,
        model: DrugRepurposingModel,
        config: Config,
        train_data: HeteroData,
        val_data: HeteroData,
        target_edge_type: Tuple[str, str, str]
    ):
        """
        Inicializa el entrenador.
        
        Args:
            model: Modelo a entrenar
            config: Configuración del entrenamiento
            train_data: Datos de entrenamiento
            val_data: Datos de validación
            target_edge_type: Tipo de arista objetivo (src_type, relation, dst_type)
        """
        self.model = model
        self.config = config
        self.train_data = train_data
        self.val_data = val_data
        self.target_edge_type = target_edge_type
        
        self.src_type = target_edge_type[0]
        self.dst_type = target_edge_type[2]
        
        self.device = config.training.device
        self.model = self.model.to(self.device)
        
        # Mover datos a device
        self.train_data = self.train_data.to(self.device)
        self.val_data = self.val_data.to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # Maximizar la métrica de validación
            factor=config.training.lr_scheduler_factor,
            patience=config.training.lr_scheduler_patience,
            verbose=True
        )
        
        # Loss function
        self.loss_fn = LinkPredictionLoss()
        
        # Evaluador
        self.evaluator = LinkPredictionEvaluator(
            hits_k_values=config.evaluation.hits_k_values
        )
        
        # Tracking
        self.best_val_metric = 0.0
        self.patience_counter = 0
        self.history = defaultdict(list)
        
        # Crear directorio de checkpoints
        os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    
    def get_edge_data(self, data: HeteroData, split: str = "train"):
        """
        Extrae los datos de aristas para entrenamiento/validación.
        
        RandomLinkSplit de PyG guarda los datos en atributos especiales:
        - edge_label_index: índices de aristas a predecir
        - edge_label: labels (1=positivo, 0=negativo)
        
        Args:
            data: HeteroData con los datos
            split: "train", "val", o "test"
            
        Returns:
            Tupla de (edge_label_index, edge_label)
        """
        edge_type = self.target_edge_type
        
        if hasattr(data[edge_type], 'edge_label_index'):
            edge_label_index = data[edge_type].edge_label_index
            edge_label = data[edge_type].edge_label
        else:
            # Fallback: usar edge_index directamente (todos positivos)
            edge_label_index = data[edge_type].edge_index
            edge_label = torch.ones(edge_label_index.size(1), device=self.device)
            
            # Generar negativos
            num_pos = edge_label_index.size(1)
            num_neg = num_pos * self.config.training.negative_sampling_ratio
            
            neg_edge_index = self._sample_negatives(data, num_neg)
            
            edge_label_index = torch.cat([edge_label_index, neg_edge_index], dim=1)
            edge_label = torch.cat([
                torch.ones(num_pos, device=self.device),
                torch.zeros(num_neg, device=self.device)
            ])
        
        return edge_label_index, edge_label
    
    def _sample_negatives(self, data: HeteroData, num_samples: int) -> torch.Tensor:
        """
        Muestrea aristas negativas (que no existen en el grafo).
        
        ESTRATEGIA:
        1. Muestrear pares aleatorios (src, dst)
        2. Verificar que no existan en el grafo
        3. Si existen, re-muestrear
        
        Esta es la estrategia más simple. Alternativas más sofisticadas:
        - Hard negatives: muestrear de nodos cercanos en el grafo
        - Type-constrained: solo muestrear tipos válidos
        
        Args:
            data: Datos del grafo
            num_samples: Número de negativos a muestrear
            
        Returns:
            edge_index de negativos [2, num_samples]
        """
        num_src = data[self.src_type].num_nodes
        num_dst = data[self.dst_type].num_nodes
        
        # Set de aristas existentes para verificación rápida
        existing_edges = set()
        if hasattr(data[self.target_edge_type], 'edge_index'):
            edge_index = data[self.target_edge_type].edge_index.cpu().numpy()
            for i in range(edge_index.shape[1]):
                existing_edges.add((edge_index[0, i], edge_index[1, i]))
        
        # Muestrear
        neg_src = []
        neg_dst = []
        
        while len(neg_src) < num_samples:
            # Muestrear más de lo necesario para compensar colisiones
            batch_size = min(num_samples - len(neg_src), num_samples)
            
            src_samples = torch.randint(0, num_src, (batch_size * 2,))
            dst_samples = torch.randint(0, num_dst, (batch_size * 2,))
            
            for src, dst in zip(src_samples.tolist(), dst_samples.tolist()):
                if (src, dst) not in existing_edges:
                    neg_src.append(src)
                    neg_dst.append(dst)
                    if len(neg_src) >= num_samples:
                        break
        
        neg_edge_index = torch.tensor([neg_src[:num_samples], neg_dst[:num_samples]], 
                                       device=self.device, dtype=torch.long)
        
        return neg_edge_index
    
    def train_epoch(self) -> float:
        """
        Ejecuta una época de entrenamiento.
        
        PROCESO:
        1. Poner modelo en modo train
        2. Obtener datos de aristas (positivos + negativos)
        3. Para cada batch:
           a. Forward pass
           b. Calcular loss
           c. Backward pass
           d. Actualizar parámetros
        4. Retornar loss promedio
        
        Returns:
            Loss promedio de la época
        """
        self.model.train()
        
        # Obtener datos de entrenamiento
        edge_label_index, edge_label = self.get_edge_data(self.train_data, "train")
        
        # Shuffle
        perm = torch.randperm(edge_label_index.size(1))
        edge_label_index = edge_label_index[:, perm]
        edge_label = edge_label[perm]
        
        total_loss = 0.0
        num_batches = 0
        batch_size = self.config.training.batch_size
        
        for start in range(0, edge_label_index.size(1), batch_size):
            end = min(start + batch_size, edge_label_index.size(1))
            
            # Batch de aristas
            batch_edge_index = edge_label_index[:, start:end]
            batch_labels = edge_label[start:end]
            
            # Forward
            scores = self.model(
                self.train_data,
                batch_edge_index,
                src_type=self.src_type,
                dst_type=self.dst_type
            )
            
            # Loss
            loss = self.loss_fn(scores, batch_labels)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping para estabilidad
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Evalúa el modelo en el conjunto de validación.
        
        Returns:
            Diccionario con métricas de validación
        """
        self.model.eval()
        
        # Obtener datos de validación
        edge_label_index, edge_label = self.get_edge_data(self.val_data, "val")
        
        # Evaluar
        metrics = self.evaluator.evaluate(
            model=self.model,
            data=self.val_data,
            edge_label_index=edge_label_index,
            edge_label=edge_label,
            src_type=self.src_type,
            dst_type=self.dst_type,
            batch_size=self.config.training.batch_size
        )
        
        return metrics
    
    def train(self) -> Dict[str, list]:
        """
        Loop de entrenamiento completo.
        
        PROCESO:
        1. Para cada época:
           a. Entrenar una época
           b. Validar
           c. Actualizar learning rate
           d. Checkpointing si mejora
           e. Early stopping si no mejora
        2. Cargar mejor modelo
        3. Retornar historial
        
        Returns:
            Diccionario con historial de entrenamiento
        """
        print(f"\n{'='*60}")
        print(f"Iniciando entrenamiento")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config.training.num_epochs}")
        print(f"Batch size: {self.config.training.batch_size}")
        print(f"Learning rate: {self.config.training.learning_rate}")
        print(f"{'='*60}\n")
        
        best_checkpoint_path = os.path.join(
            self.config.training.checkpoint_dir,
            f"best_model_{self.model.encoder_type}_{self.model.decoder_type}.pt"
        )
        
        start_time = time.time()
        
        for epoch in range(1, self.config.training.num_epochs + 1):
            epoch_start = time.time()
            
            # Entrenar
            train_loss = self.train_epoch()
            
            # Validar
            val_metrics = self.validate()
            
            # Métrica principal para early stopping (usamos MRR)
            val_metric = val_metrics.get('MRR', val_metrics.get('AUC-ROC', 0.0))
            
            # Actualizar scheduler
            self.scheduler.step(val_metric)
            
            # Logging
            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(train_loss)
            self.history['val_metric'].append(val_metric)
            self.history['learning_rate'].append(current_lr)
            
            # Imprimir progreso
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d}/{self.config.training.num_epochs} | "
                      f"Loss: {train_loss:.4f} | "
                      f"Val MRR: {val_metrics.get('MRR', 0):.4f} | "
                      f"Hits@10: {val_metrics.get('Hits@10', 0):.4f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Time: {epoch_time:.1f}s")
            
            # Checkpointing
            if val_metric > self.best_val_metric:
                self.best_val_metric = val_metric
                self.patience_counter = 0
                
                if self.config.training.save_best_model:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_metric': val_metric,
                        'config': self.config,
                    }, best_checkpoint_path)
                    print(f"  → Mejor modelo guardado (MRR: {val_metric:.4f})")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.training.patience:
                print(f"\nEarly stopping en epoch {epoch} "
                      f"(sin mejora en {self.config.training.patience} épocas)")
                break
        
        total_time = time.time() - start_time
        print(f"\nEntrenamiento completado en {total_time/60:.1f} minutos")
        print(f"Mejor MRR de validación: {self.best_val_metric:.4f}")
        
        # Cargar mejor modelo
        if self.config.training.save_best_model and os.path.exists(best_checkpoint_path):
            checkpoint = torch.load(best_checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Mejor modelo cargado desde epoch {checkpoint['epoch']}")
        
        return dict(self.history)


def train_model(
    config: Config,
    train_data: HeteroData,
    val_data: HeteroData,
    encoder_type: str = "rgcn",
    decoder_type: str = "distmult",
    target_edge_type: Optional[Tuple[str, str, str]] = None
) -> Tuple[DrugRepurposingModel, Dict]:
    """
    Función de alto nivel para entrenar un modelo.
    
    Args:
        config: Configuración
        train_data: Datos de entrenamiento
        val_data: Datos de validación
        encoder_type: Tipo de encoder
        decoder_type: Tipo de decoder
        target_edge_type: Tipo de arista objetivo
        
    Returns:
        Tupla de (modelo entrenado, historial)
    """
    # Determinar target_edge_type si no se especifica
    if target_edge_type is None:
        target_edge_type = config.data.target_edge_type
        # Buscar el tipo real en los datos
        for et in train_data.edge_types:
            if et[0] == target_edge_type[0] and et[2] == target_edge_type[2]:
                target_edge_type = et
                break
    
    print(f"\nCreando modelo: {encoder_type.upper()} + {decoder_type.upper()}")
    print(f"Target edge type: {target_edge_type}")
    
    # Crear modelo
    model = create_model(
        data=train_data,
        config=config,
        encoder_type=encoder_type,
        decoder_type=decoder_type
    )
    
    # Contar parámetros
    # num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Parámetros entrenables: {num_params:,}")
    
    # Crear trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_data=train_data,
        val_data=val_data,
        target_edge_type=target_edge_type
    )
    
    # Entrenar
    history = trainer.train()
    
    return model, history


if __name__ == "__main__":
    # Test del trainer
    from data_loader import HetionetDataLoader
    
    print("Testing Trainer...")
    
    config = get_config()
    config.training.num_epochs = 10  # Solo para test
    config.training.patience = 5
    
    # Cargar datos
    loader = HetionetDataLoader(config)
    data, train_data, val_data, test_data = loader.load_data()
    
    # Entrenar modelo
    model, history = train_model(
        config=config,
        train_data=train_data,
        val_data=val_data,
        encoder_type="rgcn",
        decoder_type="distmult"
    )
    
    print("\nHistorial de entrenamiento:")
    print(f"  Pérdida final: {history['train_loss'][-1]:.4f}")
    print(f"  Mejor métrica de validación: {max(history['val_metric']):.4f}")
