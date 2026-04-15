# Drug Repurposing mediante Graph Neural Networks

## 📋 Descripción

Este proyecto implementa un sistema de **Drug Repurposing** (reposicionamiento de fármacos) como un problema de **Link Prediction** sobre un **Knowledge Graph biomédico heterogéneo**. Utiliza Graph Neural Networks (GNNs) para aprender representaciones de fármacos y enfermedades, y predecir nuevas asociaciones terapéuticas.

### Pregunta de Investigación

> ¿Qué arquitectura GNN (R-GCN, HAN, GraphSAGE) es más efectiva para predecir asociaciones fármaco-enfermedad en un knowledge graph biomédico heterogéneo, y qué papel juega la inclusión de entidades intermedias (genes, anatomías) en la calidad de las predicciones?

## 🏗️ Arquitectura del Proyecto

```
drug_repurposing_gnn/
├── config.py           # Configuraciones centralizadas
├── data_loader.py      # Carga y preprocesamiento de Hetionet
├── models/
│   ├── __init__.py     # Exports del módulo
│   ├── encoders.py     # R-GCN, HAN, GraphSAGE heterogéneos
│   ├── decoders.py     # DistMult, DotProduct, MLP
│   └── full_model.py   # Modelo completo encoder+decoder
├── train.py            # Pipeline de entrenamiento
├── evaluate.py         # Métricas de evaluación
├── ablation.py         # Estudio de ablación
├── utils.py            # Utilidades generales
├── main.py             # Punto de entrada principal
├── requirements.txt    # Dependencias
└── README.md           # Esta documentación
```

## 🔬 Base Teórica

### Graph Neural Networks para Knowledge Graphs

Los **Knowledge Graphs biomédicos** representan el conocimiento médico como un grafo heterogéneo donde:
- **Nodos**: Fármacos, enfermedades, genes, anatomías
- **Aristas**: Relaciones como "treats", "targets", "associates"

Las **GNNs** aprenden representaciones (embeddings) de los nodos mediante **message passing**:

```
h_v^{(l+1)} = UPDATE(h_v^{(l)}, AGGREGATE({h_u^{(l)} : u ∈ N(v)}))
```

### Arquitecturas Implementadas

#### 1. R-GCN (Relational Graph Convolutional Network)
- **Idea**: Una transformación por tipo de relación
- **Fórmula**: `h_v^{(l+1)} = σ(Σ_r Σ_{u∈N_r(v)} (1/|N_r(v)|) W_r^{(l)} h_u^{(l)})`
- **Ventaja**: Captura semántica de diferentes tipos de relaciones
- **Referencia**: Schlichtkrull et al. (2018)

#### 2. HAN (Heterogeneous Attention Network)
- **Idea**: Attention a nivel de nodo y metapath
- **Fórmula**: `α_vu = softmax(LeakyReLU(a^T[Wh_v || Wh_u]))`
- **Ventaja**: Pondera la importancia de diferentes vecinos
- **Referencia**: Wang et al. (2019)

#### 3. GraphSAGE Heterogéneo
- **Idea**: Sample and aggregate para inductivo learning
- **Fórmula**: `h_v^{(l+1)} = σ(W · CONCAT(h_v^{(l)}, AGG({h_u : u ∈ Sample(N(v))})))`
- **Ventaja**: Escalable y generaliza a nodos nuevos
- **Referencia**: Hamilton et al. (2017)

### Decoders para Link Prediction

#### DistMult
```python
score(h, t, r) = h^T · diag(R_r) · t
```

#### Dot Product
```python
score(h, t) = h^T · t
```

#### MLP
```python
score(h, t) = MLP(concat(h, t))
```

### Dataset: Hetionet

[Hetionet](https://github.com/hetio/hetionet) es un knowledge graph biomédico integrador:
- **~47,000 nodos** (11 tipos)
- **~2.25M aristas** (24 tipos)
- Integra datos de DrugBank, DisGeNET, y más

## 🚀 Instalación

### Requisitos
- Python 3.8+
- CUDA 11.8+ (opcional, para GPU)

### Pasos

```bash
# Clonar o descargar el proyecto
cd drug_repurposing_gnn

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o: venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt

# Verificar instalación
python -c "import torch; import torch_geometric; print('OK!')"
```

### Instalación con CUDA (recomendado)

```bash
# Primero PyTorch con CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Luego PyG
pip install torch-geometric

# Dependencias adicionales
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

## 📖 Uso

### Test Rápido (verificar instalación)

```bash
python main.py --mode quick
```

Ejecuta un experimento con datos sintéticos para verificar que todo funciona.

### Experimento Único

```bash
# R-GCN + DistMult
python main.py --mode single --encoder rgcn --decoder distmult

# HAN + Dot Product
python main.py --mode single --encoder han --decoder dotproduct

# GraphSAGE + MLP
python main.py --mode single --encoder graphsage --decoder mlp

# Con datos sintéticos (más rápido)
python main.py --mode single --encoder rgcn --synthetic
```

### Estudio de Ablación Completo

```bash
# Con 3 semillas diferentes
python main.py --mode ablation --seeds 42 123 456

# Con datos sintéticos (para testing)
python main.py --mode ablation --synthetic
```

### Análisis de Predicciones

```bash
python main.py --mode analyze --checkpoint results/exp/final_model.pt
```

## 📊 Métricas de Evaluación

### Métricas de Ranking
- **MRR** (Mean Reciprocal Rank): Promedio de 1/rank del item correcto
- **Hits@K**: Proporción de predicciones correctas en top-K
- **Mean Rank**: Rank promedio del item correcto

### Métricas de Clasificación
- **AUC-ROC**: Área bajo la curva ROC
- **Average Precision**: Precisión promedio

## 🔍 Estudio de Ablación

El estudio evalúa la contribución de entidades intermedias:

| Configuración | Descripción |
|--------------|-------------|
| `full` | Grafo completo (Compound, Disease, Gene, Anatomy) |
| `no_anatomy` | Sin nodos de anatomía |
| `no_gene` | Sin nodos de genes |
| `no_intermediate` | Solo Compound-Disease (baseline) |

### Hipótesis

Según la teoría de **Network Medicine** (Barabási et al.), los fármacos actúan a través de vecindarios de proteínas, no directamente sobre enfermedades. Por tanto:

- Si **genes** son cruciales → valida network medicine
- Si **anatomías** ayudan → la localización tisular es informativa
- Si **ambos** son necesarios → el contexto biológico completo es importante

## 📁 Estructura de Resultados

```
results/
├── quick_test/           # Resultados del test rápido
├── single_experiments/   # Experimentos individuales
│   └── rgcn_distmult_20240414_120000/
│       ├── experiment.log
│       ├── graph_statistics.json
│       ├── training_curves.png
│       ├── results.json
│       └── final_model.pt
└── ablation/            # Estudio de ablación
    ├── ablation_MRR.png
    ├── ablation_Hits@10.png
    └── full_results.json
```

## 🧩 Extensiones Posibles

1. **Más arquitecturas**: Implementar GAT puro, CompGCN
2. **Decoders avanzados**: TransE, RotatE, ComplEx
3. **Datos adicionales**: DRKG, BioKG
4. **Cold-start**: Evaluar generalización a fármacos nuevos
5. **Validación clínica**: Comparar con Clinical Trials

## 📚 Referencias

- **Decagon**: Zitnik et al. (2018). Modeling polypharmacy side effects with graph convolutional networks
- **R-GCN**: Schlichtkrull et al. (2018). Modeling relational data with graph convolutional networks
- **Hetionet**: Himmelstein et al. (2017). Systematic integration of biomedical knowledge prioritizes drugs for repurposing
- **Network Medicine**: Gysi et al. (2020). Network medicine framework for identifying drug-repurposing opportunities
- **GraphSAGE**: Hamilton et al. (2017). Inductive representation learning on large graphs
- **HAN**: Wang et al. (2019). Heterogeneous graph attention network

## 📄 Licencia

Este proyecto es para fines educativos como parte del curso de Geometric Deep Learning.

## ✍️ Autor

Proyecto desarrollado para el curso de Geometric Deep Learning.
