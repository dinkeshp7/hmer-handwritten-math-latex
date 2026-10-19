# Handwritten Mathematical Expression Recognition (HMER) to LaTeX Converter

**Institution:** Indian Institute of Technology Guwahati (IIT Guwahati)  
**Department:** Department of Mathematics  
**Principal Investigator:** Prof. K.V. Srikanth  
**Supercomputer:** PARAM Kamrupa (NVIDIA Tesla V100 GPUs)  
**GitHub Account:** [dinkeshp7](https://github.com/dinkeshp7)

---

## 📌 Project Overview

This repository contains the complete codebase for converting handwritten mathematical expressions, equations, and full exam answer script pages into clean, compilable, textbook-quality **LaTeX (`.tex`) documents**.

The system is designed and evaluated on 13,146 handwritten exam pages from 939 IIT Guwahati MA102 students using the **PARAM Kamrupa Supercomputer**.

---

## 🏗️ Multi-Engine Architecture

The pipeline supports three distinct execution engines to guarantee robustness across complex mathematical layouts, matrices, and continuous cursive handwritten derivations:

```
Full Answer Script (.jpg)
         │
         ├───> [1] SOTA Multimodal VLM (Qwen2.5-VL / Qwen2-VL)
         │         - Dynamic Resolution 2D-RoPE Vision Transformer
         │         - End-to-end full-page visual reasoning into LaTeX
         │
         ├───> [2] Meta Nougat Academic OCR Engine
         │         - Swin Transformer + mBART Encoder-Decoder
         │         - Direct transcription of dense scientific equations
         │
         └───> [3] Hybrid Modular Engine
                   - Adaptive Binarization & Deskewing
                   - 1D Vertical IoU Non-Maximum Suppression (Page Line Segmenter)
                   - Dual-Hypothesis Disambiguation (TrOCR for text + DenseNet/ViT for math)
                   - MatrixGridParser for 2D tabular grids
                   - LaTeX Document Assembler
```

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
├── run_full_page_inference.py   # CLI entry point for full-page .tex conversion
├── run_param_kamrupa_vlm.sh     # Slurm script for VLM inference on Tesla V100
├── train_param_kamrupa.sh       # Slurm script for DenseNet HMER model training
├── src/
│   ├── api/                     # Full-page pipeline coordinator & REST API
│   │   ├── full_page_pipeline.py# Master multi-engine coordinator
│   │   └── app.py               # FastAPI server
│   ├── ocr/                     # SOTA OCR & VLM inference engines
│   │   ├── qwen_vl_ocr.py       # Qwen2.5-VL Vision-Language OCR
│   │   ├── nougat_ocr.py        # Meta Nougat Academic OCR
│   │   ├── download_vlm_weights.py # Offline weights caching for HPC
│   │   ├── dispatcher.py        # Dual-hypothesis math/text disambiguator
│   │   ├── text_ocr.py          # Microsoft TrOCR English engine
│   │   └── math_ocr.py          # pix2tex Math ViT engine
│   ├── segmentation/            # Document layout & line segmentation
│   │   ├── line_segmenter.py    # 1D Vertical IoU NMS segmenter (full-page)
│   │   └── preprocessor.py      # Deskewing & adaptive binarization
│   ├── models/                  # Custom HMER neural network architecture
│   │   ├── hmer_model.py        # Unified DenseNet + Coverage Transformer
│   │   ├── encoder.py           # DenseNet-2D with 2D positional encoding
│   │   └── decoder.py           # Pre-LN Transformer + Attention Refinement Module
│   ├── latex/                   # LaTeX document compilation & matrix parsing
│   │   ├── generator.py         # Complete article document generator
│   │   └── matrix_parser.py     # Grid projection for 2D linear algebra matrices
│   ├── tokenizer/               # 37-token LaTeX semantic vocabulary
│   └── training/                # Distributed multi-GPU PyTorch training loop
└── README.md
```

---

## 🚀 Quick Start & Usage

### 1. Requirements & Dependencies
```bash
pip install torch torchvision transformers accelerate timm opencv-python pillow sympy nltk
```

### 2. Full-Page Answer Script to LaTeX Conversion
Run full-page conversion with your chosen engine:

```bash
# SOTA Multimodal Vision-Language Model (Qwen2.5-VL)
python run_full_page_inference.py --image_path page-210103096-01.jpg --engine qwen_vl

# Meta Nougat Academic OCR
python run_full_page_inference.py --image_path page-210103096-01.jpg --engine nougat

# Hybrid Dual-Head Modular Engine
python run_full_page_inference.py --image_path page-210103096-01.jpg --engine hybrid
```

### 3. Execution on PARAM Kamrupa Supercomputer (GPU)
For offline GPU node execution without outbound internet access, download weights on `login01`:
```bash
# 1. Download offline model weights on login01
python src/ocr/download_vlm_weights.py --model all

# 2. Submit GPU inference job via Slurm
sbatch run_param_kamrupa_vlm.sh page-210103096-01.jpg qwen_vl
```

---

## 📜 License & Citation

Developed for the BTech Project at the Department of Mathematics, IIT Guwahati under the supervision of Prof. K.V. Srikanth.
