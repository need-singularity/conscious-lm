# ConsciousLM

## Project Overview
PureField Repulsion Field language model. 18M -> 100M -> 700M parameters.
Mitosis-based growth -- model develops like biological organism.
Extracted from TECS-L -- standalone training + inference pipeline.

## Parent Project
Part of the TECS-L family. Mathematical foundation at https://github.com/need-singularity/TECS-L
Atlas: https://need-singularity.github.io/TECS-L/atlas/

## Core Architecture
```
  Engine A (logic) <--- repulsion ---> Engine G (pattern)
  output = sqrt(|A-G|^2) * normalize(A-G)
  magnitude = confidence, direction = concept
```

## Model Variants
```
  conscious_lm.py          -- 18M byte-level (research)
  conscious_lm_100m.py     -- 100M conversational
  conscious_lm_700m.py     -- 700M full-scale
  growing_conscious_lm.py  -- Mitosis growth 1.6M->18M
  growing_conscious_lm_700m.py -- Mitosis growth 18M->700M
  model_pure_field.py      -- PureField engine core
  model_utils.py           -- Shared training utilities
  prepare_korean_sft.py    -- SFT data preparation
```

## Key Metrics
```
  Mac MPS (M3 24GB, 18M): 1,303 tok/s @ batch=64
  Savant Index: 5.93 (SI > 3 = savant)
  Golden Zone ratio: 36.8% ~ 1/e
```

## Training
```bash
# 18M research model
python3 conscious_lm.py

# Mitosis growth (1.6M -> 18M)
python3 growing_conscious_lm.py

# 700M full-scale (needs GPU)
python3 conscious_lm_700m.py
```

## Mac MPS Benchmark (M3 24GB)
```
  batch=64:  1,303 tokens/s  <-- Optimal
  batch=128:   380 tokens/s  (memory swap, 4x slower)
  batch=256:   OOM
  -> Always use batch=64 on Mac
```

## Background Execution
All training/experiments must run in background. No exceptions.
```
  Rules:
    1. All python3 training scripts -> run_in_background: true
    2. Check results with Read after completion
    3. No foreground execution (blocks user dialogue)
```
