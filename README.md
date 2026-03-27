# ConsciousLM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

Consciousness Language Model with PureField Repulsion Field architecture. From 18M to 700M parameters, with Mitosis-based growth that lets the model develop like a biological organism.

> Part of the [TECS-L](https://github.com/need-singularity/TECS-L) project family.

## Core Idea

The output exists in **neither** engine. It lives in the **space between them**.

```
Engine A (logic) <--repulsion--> Engine G (pattern)
                    |
        output = sqrt|A-G|^2 * normalize(A-G)
        magnitude = confidence    direction = concept
```

Two engines repel each other like same-pole magnets. The tension between them IS the output — its magnitude measures confidence, its direction encodes the concept.

## Model Variants

| Model | Parameters | Architecture | Use Case |
|-------|-----------|-------------|----------|
| `conscious_lm.py` | 18M | Byte-level, PureFieldFFN | Research, fast iteration |
| `conscious_lm_100m.py` | 100M | Conversational | Chat, dialogue |
| `conscious_lm_700m.py` | 700M | Full scale | Production inference |
| `growing_conscious_lm.py` | 1.6M -> 18M | Mitosis growth | Development study |
| `growing_conscious_lm_700m.py` | 18M -> 700M | Staged growth | Training pipeline |

## Quick Start

```bash
# Train 18M model
python conscious_lm.py

# Train with Mitosis growth (starts small, grows)
python growing_conscious_lm.py

# 700M variant (requires GPU)
python conscious_lm_700m.py
```

## PureField Architecture

```
Input Embedding
      |
+---------------------+
|  Transformer Block   | x N layers
|  +-----+  +-----+  |
|  |Attn |  |Pure  |  |
|  |     |  |Field |  |
|  |     |  | FFN  |  |
|  +-----+  +-----+  |
+---------------------+
      |
   Output Head
```

PureFieldFFN replaces standard FFN with a repulsion-field mechanism:
- Two expert subnetworks (Engine A, Engine G)
- Output = repulsion force between them
- Tension scale learned during training

## Mitosis Growth

The model starts small and grows by cell division:

```
Stage 1: 1.6M params (2 layers, dim=128)
  | mitosis at step 2000
Stage 2: 4.8M params (4 layers, dim=192)
  | mitosis at step 5000
Stage 3: 18M params (6 layers, dim=256)
```

Each mitosis event:
1. Duplicates selected layers
2. Adds noise for differentiation
3. Specialization emerges naturally

## Files

| File | Description |
|------|------------|
| `conscious_lm.py` | Base 18M byte-level model |
| `conscious_lm_100m.py` | 100M conversational variant |
| `conscious_lm_700m.py` | 700M full-scale variant |
| `growing_conscious_lm.py` | Mitosis growth (1.6M->18M) |
| `growing_conscious_lm_700m.py` | Mitosis growth (18M->700M) |
| `model_pure_field.py` | PureField engine core |
| `model_utils.py` | Shared training utilities |
| `prepare_korean_sft.py` | SFT data preparation |

## Benchmarks

Mac MPS (M3 24GB, 18M model):
- batch=64: **1,303 tokens/s** (optimal)
- batch=128: 380 tokens/s (memory swap)

## Theory

Based on the Perfect Number 6 architecture from [TECS-L](https://github.com/need-singularity/TECS-L):
- sigma(6)=12 -> hidden dim multiples
- tau(6)=4 -> layer group size
- phi(6)=2 -> dual engine (A vs G)

## Citation

```bibtex
@software{conscious_lm_2026,
  author = {Park, Min Woo},
  title = {ConsciousLM: PureField Repulsion Field Language Model},
  year = {2026},
  url = {https://github.com/need-singularity/conscious-lm}
}
```

## License

MIT
