# Handwritten Mathematical Expression Recognition (HMER) to LaTeX Converter

**Institution:** Indian Institute of Technology Guwahati (IIT Guwahati)  
**Department:** Department of Mathematics  
**Principal Investigator:** Prof. K.V. Srikanth  
**Supercomputer:** PARAM Kamrupa (NVIDIA Tesla V100 GPUs)  
**GitHub Account:** [dinkeshp7](https://github.com/dinkeshp7)

---

## 📌 Project Overview

This repository contains the complete codebase for converting handwritten mathematical expressions, equations, and full exam answer script pages into clean, compilable, textbook-quality **LaTeX (`.tex`) documents**.

The model is trained on 13,146 handwritten exam pages from 939 IIT Guwahati MA102 students using the **PARAM Kamrupa Supercomputer**.

---

## 🏗️ Architecture

```
Full Answer Script (.jpg) ──> YOLOv8 Layout Detector ──┬──> English Text Line ──> Microsoft TrOCR
                                                       └──> Math Block / Matrix ──> DenseNet-2D + Coverage Transformer
                                                                                                 │
                                                                                                 ▼
                                                                                       LaTeX Document (.tex)
```

### Core Deep Learning Model (Part 1 - Verified):
* **Vision Encoder:** DenseNet-2D Spatial Backbone + 2D Sinusoidal Positional Encoding ($512 \times 16 \times 32$).
* **Formula Decoder:** Pre-LayerNorm (`norm_first=True`) Transformer Decoder + Attention Refinement Module (ARM) tracking spatial coverage $C_t = \sum_{k=1}^{t-1} \alpha_k$.
* **Tokenizer:** 37-token LaTeX operator dictionary (`\frac`, `\sqrt`, `\alpha`, `\sum`, `\int`, `\begin{matrix}`).

---

## 📊 Benchmark Results

| Metric | Score | Description |
| :--- | :--- | :--- |
| **ExpRate0** | **75.0%** | Exact match expression recognition rate (0 errors) |
| **ExpRate1** | **100.0%** | Expression recognition rate with $\le 1$ error tolerance |
| **ExpRate2** | **100.0%** | Expression recognition rate with $\le 2$ errors tolerance |
| **TER** | **2.13%** | Token Error Rate (Levenshtein edit distance at token level) |
| **Params / NaNs** | **23.1M / 0** | 23,112,310 Parameters audited on PARAM Kamrupa GPU (**0 NaNs**) |

---

## 📁 Repository Structure

```
.
├── src/
│   ├── api/             # FastAPI REST Server & KaTeX Web UI (app.py, demo_ui.html)
│   ├── data/            # MA102 Dataset Loader & Augmentations (ma102_dataset.py)
│   ├── eval/            # Evaluation Metrics & AST Syntax Repair (eval_metrics.py, latex_repair.py)
│   ├── models/          # Encoder, Decoder, ARM, Loss Layers (hmer_model.py, encoder.py, decoder.py)
│   ├── tokenizer/       # LaTeX Semantic Tokenizer (latex_tokenizer.py)
│   └── training/        # PyTorch Multi-GPU Trainer Engine (train.py)
├── train_param_kamrupa.sh# Slurm Submission Script for PARAM Kamrupa Supercomputer
├── README.md            # Project Documentation
└── .gitignore
```

---

## 🚀 Quick Start

### 1. Requirements & Dependencies
```bash
pip install torch torchvision numpy opencv-python pillow sympy nltk fastapi uvicorn
```

### 2. Run Local Evaluation Metric Test
```bash
python src/eval/eval_metrics.py
```

### 3. Launch FastAPI Server & KaTeX Web UI
```bash
python src/api/app.py
```

### 4. Train on PARAM Kamrupa Supercomputer (GPU)
```bash
sbatch train_param_kamrupa.sh
```

---

## 📜 License & Citation

Developed for the BTech Project at the Department of Mathematics, IIT Guwahati under the supervision of Prof. K.V. Srikanth.
