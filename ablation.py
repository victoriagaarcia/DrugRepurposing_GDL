"""
================================================================================
ABLATION.PY - Estudio de Ablación para Entidades Intermedias
================================================================================

Este módulo implementa el estudio de ablación para analizar la contribución
de cada tipo de entidad intermedia (genes, anatomías) a la predicción de
asociaciones fármaco-enfermedad.

================================================================================
BASE TEÓRICA - ¿POR QUÉ HACER ABLACIÓN?
================================================================================

HIPÓTESIS DE NETWORK MEDICINE (Barabási et al.):
Los fármacos no actúan directamente sobre enfermedades, sino a través de
redes de interacción molecular. Específicamente:

1. Los fármacos tienen TARGETS (proteínas/genes)
2. Las enfermedades tienen "módulos de enfermedad" (conjuntos de genes)
3. Si un fármaco actúa cerca del módulo de una enfermedad en la red PPI,
   es más probable que sea efectivo

Esta hipótesis implica que las entidades intermedias (genes, proteínas)
son CRUCIALES para predecir qué fármacos tratan qué enfermedades.

VALIDACIÓN VÍA ABLACIÓN:
Si la hipótesis es correcta:
- Quitar genes → degradación SEVERA del rendimiento
- Quitar anatomías → degradación MENOR (menos central en la hipótesis)
- Quitar ambos → predicciones casi aleatorias

Si la ablación NO degrada el rendimiento:
- El modelo podría estar usando shortcuts
- Las asociaciones directas drug-disease son suficientes (memorización)
- El grafo no está capturando la señal biológica

================================================================================
DISEÑO DEL EXPERIMENTO:
================================================================================

CONFIGURACIONES A PROBAR:
1. FULL: Todos los tipos de nodo (Drug, Disease, Gene, Anatomy)
2. NO_ANATOMY: Sin anatomías (Drug, Disease, Gene)
3. NO_GENE: Sin genes (Drug, Disease, Anatomy)
4. NO_INTERMEDIATE: Solo Drug y Disease (baseline directo)

MÉTRICAS A REPORTAR:
- MRR, Hits@10, AUC-ROC para cada configuración
- Delta relativo respecto a FULL
- Intervalos de confianza (múltiples runs con diferentes seeds)

INTERPRETACIÓN:
- Si NO_GENE << FULL: genes son críticos (valida network medicine)
- Si NO_ANATOMY ≈ FULL: anatomías aportan poco
- Si NO_INTERMEDIATE ≈ FULL: modelo memoriza asociaciones directas

================================================================================
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np

import torch
from torch_geometric.data import HeteroData

from config import Config, get_config
from data_loader import HetionetDataLoader, create_ablation_data
from train import train_model
from evaluate import LinkPredictionEvaluator, format_metrics


class AblationStudy:
    """
    Ejecuta y analiza el estudio de ablación.
    """
    
    def __init__(self, config: Config):
        """
        Inicializa el estudio de ablación.
        
        Args:
            config: Configuración del experimento
        """
        self.config = config
        self.results = {}
        
        # Crear directorio para resultados
        self.results_dir = os.path.join(config.training.checkpoint_dir, "ablation_results")
        os.makedirs(self.results_dir, exist_ok=True)
    
    def run_single_experiment(
        self,
        ablation_config_name: str,
        node_types: List[str],
        encoder_type: str = "rgcn",
        decoder_type: str = "distmult",
        seed: int = 42
    ) -> Dict[str, float]:
        """
        Ejecuta un único experimento de ablación.
        
        Args:
            ablation_config_name: Nombre de esta configuración (para logging)
            node_types: Tipos de nodo a incluir
            encoder_type: Tipo de encoder
            decoder_type: Tipo de decoder
            seed: Semilla para reproducibilidad
            
        Returns:
            Métricas de evaluación
        """
        print(f"\n{'='*60}")
        print(f"Experimento: {ablation_config_name}")
        print(f"Node types: {node_types}")
        print(f"Encoder: {encoder_type}, Decoder: {decoder_type}")
        print(f"Seed: {seed}")
        print(f"{'='*60}")
        
        # Fijar semilla
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Crear copia de config con node_types modificados
        from copy import deepcopy
        exp_config = deepcopy(self.config)
        exp_config.data.node_types = node_types
        
        # Cargar datos con los tipos especificados
        try:
            data, train_data, val_data, test_data = create_ablation_data(
                exp_config, node_types
            )
        except Exception as e:
            print(f"Error cargando datos: {e}")
            return {"error": str(e)}
        
        # Encontrar el target edge type disponible
        target_edge_type = None
        for et in train_data.edge_types:
            if et[0] == "Compound" and et[2] == "Disease":
                target_edge_type = et
                break
        
        if target_edge_type is None:
            print(f"WARNING: No se encontró arista Compound->Disease")
            # Usar el primer tipo disponible para demo
            if len(train_data.edge_types) > 0:
                target_edge_type = list(train_data.edge_types)[0]
            else:
                return {"error": "No edge types found"}
        
        # Entrenar modelo
        try:
            model, history = train_model(
                config=exp_config,
                train_data=train_data,
                val_data=val_data,
                encoder_type=encoder_type,
                decoder_type=decoder_type,
                target_edge_type=target_edge_type
            )
        except Exception as e:
            print(f"Error en entrenamiento: {e}")
            return {"error": str(e)}
        
        # Evaluar en test
        print("\nEvaluando en conjunto de test...")
        evaluator = LinkPredictionEvaluator(
            hits_k_values=self.config.evaluation.hits_k_values,
            filtered=self.config.evaluation.filtered
        )
        
        # Obtener datos de test
        test_data = test_data.to(exp_config.training.device)
        
        if hasattr(test_data[target_edge_type], 'edge_label_index'):
            edge_label_index = test_data[target_edge_type].edge_label_index
            edge_label = test_data[target_edge_type].edge_label
        else:
            # Fallback
            edge_label_index = test_data[target_edge_type].edge_index
            edge_label = torch.ones(edge_label_index.size(1), 
                                   device=exp_config.training.device)
        
        try:
            known_edges = data[target_edge_type].edge_index.to(exp_config.training.device)

            metrics = evaluator.evaluate(
                model=model,
                data=test_data,
                edge_label_index=edge_label_index,
                edge_label=edge_label,
                src_type=target_edge_type[0],
                dst_type=target_edge_type[2],
                batch_size=exp_config.training.batch_size,
                existing_edges=known_edges
            )
        except Exception as e:
            print(f"Error en evaluación: {e}")
            return {"error": str(e)}
        
        print("\nResultados:")
        print(format_metrics(metrics))
        
        # Añadir metadatos
        metrics['ablation_config'] = ablation_config_name
        metrics['node_types'] = node_types
        metrics['encoder_type'] = encoder_type
        metrics['decoder_type'] = decoder_type
        metrics['seed'] = seed
        
        return metrics
    
    def run_full_study(
        self,
        encoder_types: List[str] = ["rgcn", "han", "sage"],
        decoder_type: str = "distmult",
        seeds: Optional[List[int]] = None,
    ) -> Dict[str, Dict]:
        """
        Ejecuta el estudio de ablación completo.
        
        Para cada combinación de:
        - Configuración de ablación (full, no_gene, no_anatomy, no_intermediate)
        - Tipo de encoder (R-GCN, HAN, SAGE)
        - Múltiples seeds (para intervalos de confianza)
        
        Args:
            encoder_types: Lista de encoders a probar
            decoder_type: Decoder a usar
            
        Returns:
            Resultados completos del estudio
        """
        print("\n" + "="*60)
        print("ESTUDIO DE ABLACIÓN COMPLETO")
        print("="*60)
        
        ablation_configs = self.config.ablation.ablation_configs
        num_runs = self.config.ablation.num_runs
        seeds = seeds if seeds is not None else [self.config.seed + i for i in range(num_runs)]
        
        all_results = {}
        
        for ablation_cfg in ablation_configs:
            config_name = ablation_cfg["name"]
            node_types = ablation_cfg["node_types"]
            
            config_results = {}
            
            for encoder_type in encoder_types:
                encoder_results = []
                
                for run in range(num_runs):
                    seed = seeds[run] if run < len(seeds) else self.config.seed + run
                    
                    metrics = self.run_single_experiment(
                        ablation_config_name=config_name,
                        node_types=node_types,
                        encoder_type=encoder_type,
                        decoder_type=decoder_type,
                        seed=seed
                    )
                    
                    if "error" not in metrics:
                        encoder_results.append(metrics)
                
                config_results[encoder_type] = encoder_results
            
            all_results[config_name] = config_results
        
        self.results = all_results
        
        # Guardar resultados
        self._save_results()
        
        return all_results
    
    def _save_results(self):
        """Guarda los resultados en formato JSON."""
        # Convertir a formato serializable
        serializable_results = {}
        for config_name, config_results in self.results.items():
            serializable_results[config_name] = {}
            for encoder_type, runs in config_results.items():
                serializable_results[config_name][encoder_type] = [
                    {k: v if not isinstance(v, list) else v 
                     for k, v in run.items()}
                    for run in runs
                ]
        
        filepath = os.path.join(self.results_dir, "ablation_results.json")
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\nResultados guardados en: {filepath}")
    
    def analyze_results(self) -> str:
        """
        Analiza los resultados del estudio de ablación.
        
        Calcula:
        - Media y desviación estándar de métricas
        - Delta relativo respecto a configuración full
        - Ranking de encoders por configuración
        
        Returns:
            String con el análisis formateado
        """
        if not self.results:
            return "No hay resultados para analizar."
        
        lines = []
        lines.append("\n" + "="*80)
        lines.append("ANÁLISIS DE RESULTADOS DE ABLACIÓN")
        lines.append("="*80)
        
        # Obtener métricas de referencia (configuración full)
        full_results = self.results.get("full", {})
        
        for config_name, config_results in self.results.items():
            lines.append(f"\n{'='*60}")
            lines.append(f"Configuración: {config_name.upper()}")
            lines.append(f"{'='*60}")
            
            for encoder_type, runs in config_results.items():
                if not runs:
                    continue
                
                lines.append(f"\n  Encoder: {encoder_type.upper()}")
                
                # Calcular estadísticas
                metrics_to_show = ['MRR', 'Hits@10', 'AUC-ROC']
                
                for metric in metrics_to_show:
                    values = [r.get(metric, 0) for r in runs if metric in r]
                    if values:
                        mean_val = np.mean(values)
                        std_val = np.std(values) if len(values) > 1 else 0
                        
                        # Calcular delta respecto a full
                        if config_name != "full" and encoder_type in full_results:
                            full_values = [r.get(metric, 0) 
                                          for r in full_results[encoder_type]]
                            if full_values:
                                full_mean = np.mean(full_values)
                                if full_mean > 0:
                                    delta = ((mean_val - full_mean) / full_mean) * 100
                                    delta_str = f" (Δ: {delta:+.1f}%)"
                                else:
                                    delta_str = ""
                            else:
                                delta_str = ""
                        else:
                            delta_str = ""
                        
                        lines.append(f"    {metric}: {mean_val:.4f} ± {std_val:.4f}{delta_str}")
        
        # Conclusiones
        lines.append("\n" + "="*80)
        lines.append("CONCLUSIONES")
        lines.append("="*80)
        
        lines.append("""
INTERPRETACIÓN DE RESULTADOS:

1. IMPACTO DE GENES (comparar 'full' vs 'no_gene'):
   - Si hay degradación significativa (>10% en MRR):
     → Los genes son cruciales para la predicción
     → Valida la hipótesis de network medicine
   - Si la degradación es menor:
     → El modelo podría usar shortcuts o memorización

2. IMPACTO DE ANATOMÍAS (comparar 'full' vs 'no_anatomy'):
   - Generalmente esperamos menor impacto que genes
   - Si hay impacto significativo:
     → La expresión tisular aporta información relevante

3. BASELINE DIRECTO (configuración 'no_intermediate'):
   - Este es el caso extremo: solo asociaciones directas drug-disease
   - Si el rendimiento es comparable a 'full':
     → El modelo memoriza asociaciones existentes
     → Las entidades intermedias no aportan generalización
   - Si es mucho peor:
     → Las entidades intermedias son esenciales para inferencia

4. COMPARACIÓN DE ARQUITECTURAS:
   - GAT/HAN: esperamos que capture mejor relaciones heterogéneas
   - GraphSAGE: más robusto para inducción
   - R-GCN: baseline sólido para grafos relacionales
""")
        
        return '\n'.join(lines)


def run_quick_ablation(config: Config, encoder_type: str = "rgcn") -> Dict:
    """
    Versión rápida del estudio de ablación (un solo encoder, un solo run).
    
    Útil para debugging y verificación del pipeline.
    
    Args:
        config: Configuración
        encoder_type: Encoder a usar
        
    Returns:
        Resultados del estudio
    """
    study = AblationStudy(config)
    
    results = {}
    
    for ablation_cfg in config.ablation.ablation_configs:
        config_name = ablation_cfg["name"]
        node_types = ablation_cfg["node_types"]
        
        metrics = study.run_single_experiment(
            ablation_config_name=config_name,
            node_types=node_types,
            encoder_type=encoder_type,
            seed=config.seed
        )
        
        results[config_name] = metrics
    
    study.results = {k: {encoder_type: [v]} for k, v in results.items()}
    
    print(study.analyze_results())
    
    return results


if __name__ == "__main__":
    print("Ejecutando estudio de ablación rápido...")
    
    config = get_config()
    
    # Reducir epochs para testing
    config.training.num_epochs = 20
    config.training.patience = 10
    config.ablation.num_runs = 1
    
    # Ejecutar ablación rápida
    results = run_quick_ablation(config, encoder_type="rgcn")
    
    print("\n" + "="*60)
    print("Resumen de resultados:")
    print("="*60)
    
    for config_name, metrics in results.items():
        if "error" not in metrics:
            print(f"\n{config_name}:")
            print(f"  MRR: {metrics.get('MRR', 'N/A'):.4f}")
            print(f"  Hits@10: {metrics.get('Hits@10', 'N/A'):.4f}")
