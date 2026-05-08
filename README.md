# What Survives Distillation? A Mechanistic Interpretability Study of GPT-2 and DistilGPT-2

## Overview

This repository contains the experiments comparing GPT-2 (12-layer teacher) and DistilGPT-2 (6-layer student) across representational similarity and circuit-level analyses.

The central finding is a **compression-collapse dichotomy**: the Greater-Than (GT) circuit survives distillation through active remapping (**87.7% task recovery**), while the Indirect Object Identification (IOI) circuit collapses completely (**35.2% accuracy vs 94.8%**).

## Structure

```
Distillation.ipynb   # Main notebook — all experiments
```

The notebook is organised into the following sections:

| Section | Description |
|---|---|
| CKA & representational similarity | Debiased linear CKA and cosine similarity under same-index and proportional layer mappings |
| Logit lens | Tuned logit lens next-token accuracy by layer |
| Attention pattern analysis | Entropy, positional bias, and cross-model head cosine similarity |
| SAE feature analysis | GPT-2 SAE applied to both models; feature survival, L0 sparsity, magnitude drift |
| IOI circuit analysis | Activation patching, path patching, DLA, name-mover head matching |
| GT circuit analysis | Activation patching, DLA, L4H1 ↔ L9H1 attention pattern verification |
| Representational capacity | Probe validity, input-type separability, factual recall, representational capacity tax |

## Setup

```bash
pip install transformers transformer-lens torch scikit-learn matplotlib seaborn pandas tqdm datasets
```

The notebook was developed on Google Colab (NVIDIA L4 GPU). Mount your Drive and set `OUTPUT_DIR` at the top of the main cell to your preferred results path.

## Models

Both models load automatically from HuggingFace:

- `gpt2` — 12 layers, 117M parameters
- `distilgpt2` — 6 layers, 82M parameters

Pre-trained GPT-2 SAEs (`gpt2-small-res-jb`) are loaded via [SAELens](https://github.com/jbloomAus/SAELens).
