"""
================================================================================
TRAIN.PY - Loop de Entrenamiento para Drug Repurposing
================================================================================
"""

import os
import time
from collections import defaultdict
from typing import Dict, Optional, Tuple

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import HeteroData

from config import Config, get_config
from evaluate import LinkPredictionEvaluator
from models.full_model import DrugRepurposingModel, LinkPredictionLoss, create_model


class Trainer:
    """
    Entrenador para modelos de Drug Repurposing.
    """

    def __init__(
        self,
        model: DrugRepurposingModel,
        config: Config,
        train_data: HeteroData,
        val_data: HeteroData,
        target_edge_type: Tuple[str, str, str],
    ):
        self.model = model
        self.config = config
        self.train_data = train_data
        self.val_data = val_data
        self.target_edge_type = target_edge_type

        self.src_type = target_edge_type[0]
        self.dst_type = target_edge_type[2]

        self.device = torch.device(config.training.device)
        self.model = self.model.to(self.device)

        self.train_data = self.train_data.to(self.device)
        self.val_data = self.val_data.to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=config.training.lr_scheduler_factor,
            patience=config.training.lr_scheduler_patience,
        )

        self.loss_fn = LinkPredictionLoss()

        self.evaluator = LinkPredictionEvaluator(
            hits_k_values=config.evaluation.hits_k_values,
            filtered=config.evaluation.filtered,
        )

        self.best_val_metric = 0.0
        self.patience_counter = 0
        self.history = defaultdict(list)

        os.makedirs(config.training.checkpoint_dir, exist_ok=True)

    def get_edge_data(self, data: HeteroData, split: str = "train"):
        """
        Extrae edge_label_index y edge_label para entrenamiento/validación.
        """
        del split
        edge_type = self.target_edge_type

        if hasattr(data[edge_type], "edge_label_index") and hasattr(data[edge_type], "edge_label"):
            edge_label_index = data[edge_type].edge_label_index
            edge_label = data[edge_type].edge_label
        else:
            edge_label_index = data[edge_type].edge_index
            edge_label = torch.ones(edge_label_index.size(1), device=self.device)

            num_pos = edge_label_index.size(1)
            num_neg = num_pos * self.config.training.negative_sampling_ratio
            neg_edge_index = self._sample_negatives(data, num_neg)

            edge_label_index = torch.cat([edge_label_index, neg_edge_index], dim=1)
            edge_label = torch.cat(
                [
                    torch.ones(num_pos, device=self.device),
                    torch.zeros(num_neg, device=self.device),
                ]
            )

        return edge_label_index.to(self.device), edge_label.to(self.device)

    def _sample_negatives(self, data: HeteroData, num_samples: int) -> torch.Tensor:
        """
        Muestrea aristas negativas (que no existen).
        """
        num_src = data[self.src_type].num_nodes
        num_dst = data[self.dst_type].num_nodes

        existing_edges = set()
        if hasattr(data[self.target_edge_type], "edge_index"):
            edge_index = data[self.target_edge_type].edge_index.cpu().numpy()
            for i in range(edge_index.shape[1]):
                existing_edges.add((edge_index[0, i], edge_index[1, i]))

        neg_src = []
        neg_dst = []

        while len(neg_src) < num_samples:
            remaining = num_samples - len(neg_src)
            batch_size = max(remaining * 2, 32)

            src_samples = torch.randint(0, num_src, (batch_size,))
            dst_samples = torch.randint(0, num_dst, (batch_size,))

            for src, dst in zip(src_samples.tolist(), dst_samples.tolist()):
                if (src, dst) not in existing_edges:
                    neg_src.append(src)
                    neg_dst.append(dst)
                    existing_edges.add((src, dst))
                    if len(neg_src) >= num_samples:
                        break

        neg_edge_index = torch.tensor(
            [neg_src[:num_samples], neg_dst[:num_samples]],
            device=self.device,
            dtype=torch.long,
        )
        return neg_edge_index

    def train_epoch(self) -> float:
        """
        Ejecuta una época de entrenamiento.
        """
        self.model.train()

        edge_label_index, edge_label = self.get_edge_data(self.train_data, "train")

        perm = torch.randperm(edge_label_index.size(1), device=self.device)
        edge_label_index = edge_label_index[:, perm]
        edge_label = edge_label[perm]

        total_loss = 0.0
        num_batches = 0
        batch_size = self.config.training.batch_size

        for start in range(0, edge_label_index.size(1), batch_size):
            end = min(start + batch_size, edge_label_index.size(1))

            batch_edge_index = edge_label_index[:, start:end]
            batch_labels = edge_label[start:end]

            scores = self.model(
                self.train_data,
                batch_edge_index,
                src_type=self.src_type,
                dst_type=self.dst_type,
            )

            loss = self.loss_fn(scores, batch_labels)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Evalúa el modelo en validación.
        """
        self.model.eval()

        edge_label_index, edge_label = self.get_edge_data(self.val_data, "val")

        # Pasamos como existing edges aristas verdaderas de train y positivos de val
        existing_edges = self.train_data[self.target_edge_type].edge_index
        if hasattr(self.val_data[self.target_edge_type], "edge_label_index") and hasattr(self.val_data[self.target_edge_type], "edge_label"):
            val_pos_mask = self.val_data[self.target_edge_type].edge_label == 1
            val_pos_edges = self.val_data[self.target_edge_type].edge_label_index[:, val_pos_mask]
            existing_edges = torch.cat([existing_edges, val_pos_edges], dim=1)
            existing_edges = torch.unique(existing_edges, dim=1)

        metrics = self.evaluator.evaluate(
            model=self.model,
            data=self.val_data,
            edge_label_index=edge_label_index,
            edge_label=edge_label,
            src_type=self.src_type,
            dst_type=self.dst_type,
            batch_size=self.config.training.batch_size,
            existing_edges=existing_edges,
        )
        return metrics

    def train(self) -> Dict[str, list]:
        """
        Loop de entrenamiento completo.
        """
        print(f"\n{'=' * 60}")
        print("Iniciando entrenamiento")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config.training.num_epochs}")
        print(f"Batch size: {self.config.training.batch_size}")
        print(f"Learning rate: {self.config.training.learning_rate}")
        print(f"{'=' * 60}\n")

        best_checkpoint_path = os.path.join(
            self.config.training.checkpoint_dir,
            f"best_model_{self.model.encoder_type}_{self.model.decoder_type}.pt",
        )

        start_time = time.time()

        for epoch in range(1, self.config.training.num_epochs + 1):
            epoch_start = time.time()

            train_loss = self.train_epoch()
            val_metrics = self.validate()

            val_metric = val_metrics.get("MRR", val_metrics.get("AUC-ROC", 0.0))
            self.scheduler.step(val_metric)

            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_loss)
            self.history["val_metric"].append(val_metric)
            self.history["learning_rate"].append(current_lr)
            self.history["val_MRR"].append(val_metrics.get("MRR", 0.0))
            self.history["val_Hits@10"].append(val_metrics.get("Hits@10", 0.0))
            self.history["val_AUC-ROC"].append(val_metrics.get("AUC-ROC", 0.0))

            if epoch % 5 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:3d}/{self.config.training.num_epochs} | "
                    f"Loss: {train_loss:.4f} | "
                    f"Val MRR: {val_metrics.get('MRR', 0):.4f} | "
                    f"Hits@10: {val_metrics.get('Hits@10', 0):.4f} | "
                    f"AUC-ROC: {val_metrics.get('AUC-ROC', 0):.4f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Time: {epoch_time:.1f}s"
                )

            if val_metric > self.best_val_metric:
                self.best_val_metric = val_metric
                self.patience_counter = 0

                if self.config.training.save_best_model:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "val_metric": val_metric,
                        },
                        best_checkpoint_path,
                    )
                    print(f"  → Mejor modelo guardado (MRR: {val_metric:.4f})")
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.config.training.patience:
                print(
                    f"\nEarly stopping en epoch {epoch} "
                    f"(sin mejora en {self.config.training.patience} épocas)"
                )
                break

        total_time = time.time() - start_time
        print(f"\nEntrenamiento completado en {total_time / 60:.1f} minutos")
        print(f"Mejor MRR de validación: {self.best_val_metric:.4f}")

        if self.config.training.save_best_model and os.path.exists(best_checkpoint_path):
            checkpoint = torch.load(best_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Mejor modelo cargado desde epoch {checkpoint['epoch']}")

        return dict(self.history)


def train_model(
    config: Config,
    train_data: HeteroData,
    val_data: HeteroData,
    encoder_type: str = "rgcn",
    decoder_type: str = "distmult",
    target_edge_type: Optional[Tuple[str, str, str]] = None,
) -> Tuple[DrugRepurposingModel, Dict]:
    """
    Función de alto nivel para entrenar un modelo.
    """
    if target_edge_type is None:
        target_edge_type = config.data.target_edge_type
        for et in train_data.edge_types:
            if et[0] == target_edge_type[0] and et[2] == target_edge_type[2]:
                target_edge_type = et
                break

    print(f"\nCreando modelo: {encoder_type.upper()} + {decoder_type.upper()}")
    print(f"Target edge type: {target_edge_type}")

    model = create_model(
        data=train_data,
        config=config,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
    )

    trainer = Trainer(
        model=model,
        config=config,
        train_data=train_data,
        val_data=val_data,
        target_edge_type=target_edge_type,
    )

    history = trainer.train()
    return trainer.model, history


if __name__ == "__main__":
    from data_loader import HetionetDataLoader

    print("Testing Trainer...")

    config = get_config()
    config.training.num_epochs = 10
    config.training.patience = 5

    loader = HetionetDataLoader(config)
    data, train_data, val_data, test_data = loader.load_data()

    model, history = train_model(
        config=config,
        train_data=train_data,
        val_data=val_data,
        encoder_type="rgcn",
        decoder_type="distmult",
    )

    print("\nHistorial de entrenamiento:")
    print(f"  Pérdida final: {history['train_loss'][-1]:.4f}")
    print(f"  Mejor métrica de validación: {max(history['val_metric']):.4f}")