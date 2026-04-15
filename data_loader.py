"""
================================================================================
DATA_LOADER.PY - Carga y Preprocesamiento del Knowledge Graph Biomédico
================================================================================

BASE TEÓRICA:
-------------

KNOWLEDGE GRAPHS BIOMÉDICOS:
Un Knowledge Graph (KG) es un grafo dirigido donde:
- Nodos representan entidades (fármacos, enfermedades, genes, etc.)
- Aristas representan relaciones entre entidades
- Cada arista tiene un tipo específico (treats, targets, associates, etc.)

HETIONET:
Hetionet (Himmelstein et al., 2017) es un KG integrador que combina:
- DrugBank: interacciones fármaco-proteína
- DisGeNET: asociaciones gen-enfermedad  
- Human Interactome: interacciones proteína-proteína
- SIDER: efectos secundarios de fármacos
- Y muchas otras fuentes

Estructura de Hetionet v1.0:
- 47,031 nodos de 11 tipos
- 2,250,197 aristas de 24 tipos

TIPOS DE NODOS RELEVANTES:
1. Compound (1,552 nodos): Fármacos y compuestos químicos
2. Disease (137 nodos): Enfermedades
3. Gene (20,945 nodos): Genes humanos
4. Anatomy (402 nodos): Estructuras anatómicas

TIPOS DE RELACIONES CLAVE:
- Compound-treats-Disease: Indicaciones terapéuticas aprobadas (755 aristas)
- Compound-palliates-Disease: Tratamientos paliativos
- Compound-targets-Gene: Targets farmacológicos
- Disease-associates-Gene: Asociaciones gen-enfermedad
- Gene-interacts-Gene: Interacciones proteína-proteína
- Anatomy-expresses-Gene: Expresión tisular de genes

PREPROCESAMIENTO PARA PYTORCH GEOMETRIC:
PyG usa el formato HeteroData para grafos heterogéneos:
- data[node_type].x: features de nodos
- data[src, relation, dst].edge_index: aristas
- Permite definir convoluciones diferentes por tipo de relación

================================================================================
"""

import os
import json
import bz2
import urllib.request
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
import numpy as np
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit
from sklearn.preprocessing import LabelEncoder

from first_draft.config import Config, get_config


class HetionetDataLoader:
    """
    Cargador de datos para Hetionet.
    
    Este cargador:
    1. Descarga Hetionet si no existe localmente
    2. Parsea el JSON y extrae nodos/aristas relevantes
    3. Crea el objeto HeteroData de PyTorch Geometric
    4. Genera el split train/val/test para link prediction
    """
    
    def __init__(self, config: Config):
        """
        Inicializa el cargador.
        
        Args:
            config: Objeto de configuración con parámetros del dataset
        """
        self.config = config
        self.data_dir = config.data.data_dir
        self.node_types = config.data.node_types
        self.target_edge_type = config.data.target_edge_type
        
        # Mapeos de IDs a índices numéricos
        self.node_to_idx: Dict[str, Dict[str, int]] = {}
        self.idx_to_node: Dict[str, Dict[int, str]] = {}
        
        # Crear directorio de datos
        os.makedirs(self.data_dir, exist_ok=True)
        
    def download_hetionet(self) -> str:
        """
        Descarga Hetionet si no existe localmente.
        
        Hetionet se distribuye como JSON comprimido con bz2.
        El archivo contiene toda la estructura del grafo.
        
        Returns:
            Ruta al archivo descargado
        """
        filepath = os.path.join(self.data_dir, "hetionet-v1.0.json.bz2")
        
        if not os.path.exists(filepath):
            print("Descargando Hetionet v1.0...")
            print(f"URL: {self.config.data.hetionet_url}")
            
            try:
                urllib.request.urlretrieve(
                    self.config.data.hetionet_url, 
                    filepath
                )
                print(f"Descargado en: {filepath}")
            except Exception as e:
                print(f"Error descargando Hetionet: {e}")
                print("Creando dataset sintético de ejemplo...")
                return self._create_synthetic_data()
        else:
            print(f"Hetionet ya existe en: {filepath}")
            
        return filepath
    
    def _create_synthetic_data(self) -> str:
        """
        Crea un dataset sintético para testing si la descarga falla.
        
        NOTA: Este dataset es solo para verificar que el código funciona.
        Para experimentos reales, usar Hetionet u otro KG real.
        """
        print("Creando dataset sintético de ejemplo...")
        
        # Crear estructura similar a Hetionet pero pequeña
        np.random.seed(self.config.seed)
        
        synthetic_data = {
            "nodes": [],
            "edges": []
        }
        
        # Crear nodos sintéticos
        node_counts = {
            "Compound": 200,
            "Disease": 50,
            "Gene": 500,
            "Anatomy": 30
        }
        
        for node_type, count in node_counts.items():
            for i in range(count):
                node_id = f"{node_type}::{i}"
                synthetic_data["nodes"].append({
                    "identifier": node_id,
                    "kind": node_type,
                    "name": f"{node_type}_{i}"
                })
        
        # Crear aristas sintéticas
        edge_configs = [
            ("Compound", "treats", "Disease", 300),
            ("Compound", "targets", "Gene", 1000),
            ("Disease", "associates", "Gene", 800),
            ("Gene", "interacts", "Gene", 2000),
            ("Anatomy", "expresses", "Gene", 1500),
            ("Compound", "palliates", "Disease", 150),
        ]
        
        for src_type, rel, dst_type, count in edge_configs:
            src_nodes = [n["identifier"] for n in synthetic_data["nodes"] 
                        if n["kind"] == src_type]
            dst_nodes = [n["identifier"] for n in synthetic_data["nodes"] 
                        if n["kind"] == dst_type]
            
            for _ in range(count):
                src = np.random.choice(src_nodes)
                dst = np.random.choice(dst_nodes)
                
                # Evitar self-loops para interacciones Gene-Gene
                if src_type == dst_type and src == dst:
                    continue
                    
                synthetic_data["edges"].append({
                    "source": src,
                    "target": dst,
                    "kind": f"{src_type[0]}{''.join(w[0] for w in rel.split('_')[:1])}{dst_type[0]}",
                    "direction": "both" if src_type == dst_type else "forward",
                    "source_kind": src_type,
                    "target_kind": dst_type,
                    "relation": rel
                })
        
        # Guardar como JSON
        filepath = os.path.join(self.data_dir, "synthetic_hetionet.json")
        with open(filepath, 'w') as f:
            json.dump(synthetic_data, f)
            
        print(f"Dataset sintético creado: {filepath}")
        print(f"  - Nodos: {len(synthetic_data['nodes'])}")
        print(f"  - Aristas: {len(synthetic_data['edges'])}")
        
        return filepath
    
    def parse_hetionet(self, filepath: str) -> Tuple[Dict, Dict]:
        """
        Parsea el archivo JSON de Hetionet.
        
        Hetionet JSON tiene la estructura:
        {
            "nodes": [{"identifier": ..., "kind": ..., "name": ...}, ...],
            "edges": [{"source": ..., "target": ..., "kind": ..., ...}, ...]
        }
        
        Args:
            filepath: Ruta al archivo JSON (comprimido o no)
            
        Returns:
            Tupla de (nodes_dict, edges_dict) organizados por tipo
        """
        print(f"Parseando Hetionet desde: {filepath}")
        
        # Cargar JSON (comprimido o no)
        if filepath.endswith('.bz2'):
            with bz2.open(filepath, 'rt') as f:
                data = json.load(f)
        else:
            with open(filepath, 'r') as f:
                data = json.load(f)
        
        # Organizar nodos por tipo
        nodes_by_type: Dict[str, List[Dict]] = defaultdict(list)
        for node in data.get("nodes", []):
            kind = node.get("kind", "Unknown")
            if kind in self.node_types:
                nodes_by_type[kind].append(node)
        
        # Crear mapeos de ID a índice
        for node_type in self.node_types:
            nodes = nodes_by_type[node_type]
            self.node_to_idx[node_type] = {}
            self.idx_to_node[node_type] = {}
            
            for idx, node in enumerate(nodes):
                node_id = node["identifier"]
                self.node_to_idx[node_type][node_id] = idx
                self.idx_to_node[node_type][idx] = node_id
        
        # Organizar aristas por tipo de relación
        # El tipo de arista en PyG es una tupla: (src_type, relation, dst_type)
        edges_by_type: Dict[Tuple, List[Tuple[int, int]]] = defaultdict(list)
        
        for edge in data.get("edges", []):
            source_id = edge.get("source", "")
            target_id = edge.get("target", "")
            
            # Extraer tipo de nodo del identificador o usar campo explícito
            if "source_kind" in edge:
                src_type = edge["source_kind"]
                dst_type = edge["target_kind"]
                relation = edge.get("relation", edge.get("kind", "related"))
            else:
                # En Hetionet, el identifier tiene formato "tipo::nombre"
                # o el kind del edge indica la relación
                src_type = source_id.split("::")[0] if "::" in source_id else None
                dst_type = target_id.split("::")[0] if "::" in target_id else None
                
                # El kind del edge en Hetionet es como "CtD" (Compound-treats-Disease)
                # Necesitamos parsearlo
                edge_kind = edge.get("kind", "")
                relation = self._parse_edge_kind(edge_kind)
            
            # Verificar que ambos tipos de nodo están en nuestra lista
            if src_type not in self.node_types or dst_type not in self.node_types:
                continue
                
            # Obtener índices
            if source_id in self.node_to_idx.get(src_type, {}):
                src_idx = self.node_to_idx[src_type][source_id]
            else:
                continue
                
            if target_id in self.node_to_idx.get(dst_type, {}):
                dst_idx = self.node_to_idx[dst_type][target_id]
            else:
                continue
            
            # Agregar arista
            edge_type = (src_type, relation, dst_type)
            edges_by_type[edge_type].append((src_idx, dst_idx))
            
            # Si es bidireccional, agregar la inversa
            if edge.get("direction", "forward") == "both":
                reverse_type = (dst_type, f"{relation}_rev", src_type)
                edges_by_type[reverse_type].append((dst_idx, src_idx))
        
        print(f"Nodos parseados por tipo:")
        for node_type in self.node_types:
            print(f"  - {node_type}: {len(nodes_by_type[node_type])}")
            
        print(f"Aristas parseadas por tipo:")
        for edge_type, edges in edges_by_type.items():
            print(f"  - {edge_type}: {len(edges)}")
        
        return dict(nodes_by_type), dict(edges_by_type)
    
    def _parse_edge_kind(self, kind: str) -> str:
        """
        Convierte el formato de edge kind de Hetionet a nombre legible.
        
        Hetionet usa abreviaciones como:
        - CtD: Compound-treats-Disease
        - CpD: Compound-palliates-Disease
        - CtG: Compound-targets-Gene
        - DaG: Disease-associates-Gene
        - GiG: Gene-interacts-Gene
        
        Args:
            kind: Abreviación del tipo de arista
            
        Returns:
            Nombre legible de la relación
        """
        kind_mapping = {
            "CtD": "treats",
            "CpD": "palliates", 
            "CtG": "targets",
            "CbG": "binds",
            "CuG": "upregulates",
            "CdG": "downregulates",
            "DaG": "associates",
            "DuG": "upregulates",
            "DdG": "downregulates",
            "DlA": "localizes",
            "AeG": "expresses",
            "AuG": "upregulates",
            "AdG": "downregulates",
            "GiG": "interacts",
            "GrG": "regulates",
            "GcG": "covaries",
        }
        return kind_mapping.get(kind, kind.lower())
    
    def create_hetero_data(
        self, 
        nodes_by_type: Dict[str, List[Dict]],
        edges_by_type: Dict[Tuple, List[Tuple[int, int]]]
    ) -> HeteroData:
        """
        Crea el objeto HeteroData de PyTorch Geometric.
        
        HeteroData es la estructura de PyG para grafos heterogéneos:
        - Almacena features y aristas por tipo separadamente
        - Permite aplicar diferentes operaciones a cada tipo
        
        FEATURES DE NODOS:
        En este caso, usamos embeddings aleatorios iniciales (learnable).
        Esto es común cuando no hay features inherentes de los nodos.
        El modelo aprenderá las representaciones óptimas durante el entrenamiento.
        
        Alternativas:
        - Para fármacos: fingerprints moleculares, descriptores químicos
        - Para genes: embeddings de secuencia, ontología funcional
        - Para enfermedades: embeddings de texto de descripciones
        
        Args:
            nodes_by_type: Diccionario de nodos por tipo
            edges_by_type: Diccionario de aristas por tipo
            
        Returns:
            Objeto HeteroData listo para PyG
        """
        data = HeteroData()
        
        # Agregar features de nodos
        # Usamos embeddings aleatorios que se aprenderán (common en KG embedding)
        for node_type in self.node_types:
            num_nodes = len(nodes_by_type.get(node_type, []))
            if num_nodes > 0:
                # Feature inicial: embedding de identidad (one-hot sería muy grande)
                # Usamos un índice que luego se mapea a embedding learnable
                data[node_type].num_nodes = num_nodes
                
                # Podríamos inicializar con features random o aprender embeddings
                # Aquí usamos random inicial que se refinará con el entrenamiento
                data[node_type].x = torch.randn(num_nodes, self.config.model.hidden_dim)
                
        print(f"\nHeteroData creado:")
        print(f"  Tipos de nodo: {data.node_types}")
        
        # Agregar aristas
        for edge_type, edges in edges_by_type.items():
            if len(edges) > 0:
                # Convertir lista de tuplas a tensor
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                data[edge_type].edge_index = edge_index
                
        print(f"  Tipos de arista: {data.edge_types}")
        
        return data
    
    def create_link_split(self, data: HeteroData) -> Tuple[HeteroData, HeteroData, HeteroData]:
        """
        Crea el split train/val/test para link prediction.
        
        CONSIDERACIONES IMPORTANTES PARA LINK PREDICTION:
        
        1. SPLIT SOBRE ARISTAS TARGET:
           Solo dividimos las aristas del tipo objetivo (Compound-treats-Disease).
           Las demás aristas se mantienen en todos los splits como información
           estructural del grafo.
        
        2. EVITAR DATA LEAKAGE:
           - Las aristas de test NO deben estar en train
           - El modelo no debe poder "memorizar" conexiones de test
           
        3. INDUCTIVE vs TRANSDUCTIVE:
           - Transductive: todos los nodos vistos en train (solo predecir aristas)
           - Inductive: algunos nodos no vistos (más difícil)
           Aquí usamos transductive (más común para drug repurposing)
        
        4. NEGATIVE EDGES:
           RandomLinkSplit genera automáticamente aristas negativas
           (pares que no existen en el grafo)
        
        Args:
            data: Objeto HeteroData con el grafo completo
            
        Returns:
            Tupla de (train_data, val_data, test_data)
        """
        target = self.target_edge_type
        
        # Verificar que el target edge type existe
        if target not in data.edge_types:
            # Buscar un tipo similar
            available = [et for et in data.edge_types 
                        if et[0] == target[0] and et[2] == target[2]]
            if available:
                target = available[0]
                print(f"Usando edge type: {target}")
            else:
                print(f"ADVERTENCIA: Edge type {target} no encontrado")
                print(f"Disponibles: {data.edge_types}")
                # Usar el primer edge type disponible para demo
                target = list(data.edge_types)[0]
                print(f"Usando: {target}")
        
        # RandomLinkSplit de PyG
        # Divide las aristas del tipo especificado en train/val/test
        transform = RandomLinkSplit(
            num_val=self.config.data.val_ratio,
            num_test=self.config.data.test_ratio,
            is_undirected=False,  # Grafos dirigidos en KG
            edge_types=[target],  # Solo dividir aristas objetivo
            rev_edge_types=None,  # No hay aristas reversas explícitas
            add_negative_train_samples=True,
            neg_sampling_ratio=self.config.training.negative_sampling_ratio,
        )
        
        train_data, val_data, test_data = transform(data)
        
        # Estadísticas del split
        print(f"\nSplit de aristas '{target}':")
        if hasattr(train_data[target], 'edge_label_index'):
            print(f"  Train: {train_data[target].edge_label_index.size(1)} aristas con labels")
        if hasattr(val_data[target], 'edge_label_index'):
            print(f"  Val: {val_data[target].edge_label_index.size(1)} aristas con labels")
        if hasattr(test_data[target], 'edge_label_index'):
            print(f"  Test: {test_data[target].edge_label_index.size(1)} aristas con labels")
        
        return train_data, val_data, test_data
    
    def load_data(self) -> Tuple[HeteroData, HeteroData, HeteroData, HeteroData]:
        """
        Pipeline completo de carga de datos.
        
        Returns:
            Tupla de (data_full, train_data, val_data, test_data)
        """
        # 1. Descargar si es necesario
        filepath = self.download_hetionet()
        
        # 2. Parsear
        nodes_by_type, edges_by_type = self.parse_hetionet(filepath)
        
        # 3. Crear HeteroData
        data = self.create_hetero_data(nodes_by_type, edges_by_type)
        
        # 4. Crear split
        train_data, val_data, test_data = self.create_link_split(data)
        
        return data, train_data, val_data, test_data


def create_ablation_data(
    config: Config, 
    node_types_to_keep: List[str]
) -> Tuple[HeteroData, HeteroData, HeteroData, HeteroData]:
    """
    Crea versiones del dataset para estudio de ablación.
    
    Modifica la configuración para incluir solo ciertos tipos de nodos,
    permitiendo medir el impacto de cada tipo de entidad intermedia.
    
    Args:
        config: Configuración base
        node_types_to_keep: Lista de tipos de nodo a mantener
        
    Returns:
        Datos con solo los tipos especificados
    """
    # Crear copia de config con node_types modificados
    from copy import deepcopy
    ablation_config = deepcopy(config)
    ablation_config.data.node_types = node_types_to_keep
    
    print(f"\n{'='*60}")
    print(f"Creando dataset de ablación")
    print(f"Tipos de nodo: {node_types_to_keep}")
    print(f"{'='*60}")
    
    loader = HetionetDataLoader(ablation_config)
    return loader.load_data()


if __name__ == "__main__":
    # Test del data loader
    config = get_config()
    loader = HetionetDataLoader(config)
    
    data, train_data, val_data, test_data = loader.load_data()
    
    print("\n" + "="*60)
    print("RESUMEN DEL DATASET")
    print("="*60)
    print(f"\nGrafo completo:")
    print(f"  Node types: {data.node_types}")
    print(f"  Edge types: {data.edge_types}")
    
    for node_type in data.node_types:
        if hasattr(data[node_type], 'num_nodes'):
            print(f"  {node_type}: {data[node_type].num_nodes} nodos")
