# 09 — Resource Links (consolidated)

> Every model, tool, dataset, paper, and service referenced across the study, in one place with links. Grouped by type. All arXiv papers below have been **confirmed against arXiv** (titles verified). Remaining ⚠️ marks flag real cautions (data residency, license), not unverified sources. See `01-landscape.md` for what each one means for us.

---

## 1. Specialized OCR / math models (mostly PRINTED)
| Name | Link | Note |
|---|---|---|
| pix2tex / LaTeX-OCR | https://github.com/lukas-blecher/LaTeX-OCR | Equation→LaTeX; handwriting experimental; MIT |
| Nougat (Meta) | https://github.com/facebookresearch/nougat · paper https://arxiv.org/abs/2308.13418 | Printed PDF→MD/LaTeX; weights CC-BY-NC |
| Texify | https://github.com/VikParuchuri/texify | DEPRECATED → Surya |
| Surya | https://github.com/datalab-to/surya | Full-page OCR + equations; OpenRAIL-M weights |
| Pix2Text | https://github.com/breezedeus/Pix2Text | Open "free Mathpix"; printed; MIT |
| GOT-OCR2.0 | https://github.com/Ucas-HaoranWei/GOT-OCR2.0 · paper https://arxiv.org/abs/2409.01704 | Unified OCR-2.0 |

## 2. Handwriting recognition (prose)
| Name | Link | Note |
|---|---|---|
| TrOCR (Microsoft) | https://huggingface.co/microsoft/trocr-base-handwritten | Handwritten English text (no math); MIT |
| IAM database (dataset) | https://fki.tic.heia-fr.ch/databases/iam-handwriting-database | English prose handwriting benchmark |

## 3. HMER — handwritten math (isolated equations) research line
| Name | Link | CROHME ExpRate (approx) |
|---|---|---|
| BTTR | https://arxiv.org/abs/2105.02412 | ~53% |
| CoMER | https://arxiv.org/abs/2207.04410 | ~59–63% |
| ICAL | https://arxiv.org/abs/2405.09032 | ~60% / HME100K 69% |
| PosFormer | https://arxiv.org/abs/2407.07764 · code https://github.com/SJTU-DeepVisionLab/PosFormer | 62–65% / HME100K 69.5% |
| Uni-MuMER (fine-tuned Qwen2.5-VL) | https://arxiv.org/abs/2505.23566 | ~78–82% / HME100K 71.9% |
| TAMER | https://arxiv.org/abs/2408.08578 | tree-aware transformer |

## 4. General VLMs (self-hostable ones marked ★)
| Name | Link |
|---|---|
| Qwen2.5-VL ★ | https://arxiv.org/abs/2502.13923 · https://huggingface.co/Qwen |
| InternVL ★ | https://github.com/OpenGVLab/InternVL |
| Llama 3.2 Vision ★ | https://huggingface.co/meta-llama |
| GPT-4o / 4.1 (OpenAI) | https://platform.openai.com/docs |
| Claude (Anthropic) | https://docs.claude.com |
| Gemini 2.x (Google) | https://ai.google.dev |

## 5. Commercial services
| Name | Link | Note |
|---|---|---|
| Mathpix | https://mathpix.com · docs https://docs.mathpix.com · pricing https://mathpix.com/pricing/api | Best commercial handwriting math OCR; confidence scores |
| SimpleTex | https://simpletex.net | ⚠️ China-hosted — residency concern |
| MyScript iink | https://developer.myscript.com | Best for stroke/ink, not scans |
| Google Document AI | https://cloud.google.com/document-ai | LaTeX add-on, typeset-oriented |
| Azure Document Intelligence | https://learn.microsoft.com/azure/ai-services/document-intelligence | LaTeX formulas, typeset-oriented |
| AWS Textract | https://aws.amazon.com/textract | Text/tables, no LaTeX |

## 6. Datasets
| Name | Link | Note |
|---|---|---|
| CROHME | https://www.cs.rit.edu/~crohme2019/ | Isolated handwritten expressions; online+offline |
| MathWriting (Google 2024) | https://arxiv.org/abs/2404.10690 · https://github.com/google-research/google-research/tree/master/mathwriting | Largest online; CC-BY-NC-SA |
| HME100K | https://arxiv.org/abs/2207.11463 (CAN paper) | ~100k real-photo handwritten expressions |
| im2latex-100k | https://arxiv.org/abs/1609.04938 · https://huggingface.co/datasets/yuntian-deng/im2latex-100k | PRINTED formulas |
| FERMAT — *"Can Vision-Language Models Evaluate Handwritten Math?"* (IIT Madras/AI4Bharat) | https://arxiv.org/abs/2501.07244 | ✅ Handwritten multi-line solutions, grades 7–12 — closest to our use case |
| VEHME — *"A Vision-Language Model For Evaluating Handwritten Mathematics Expressions"* | https://arxiv.org/abs/2510.22798 | ✅ VLM for evaluating handwritten math (SFT+RL) |

## 7. Document pipeline (layout, dewarp, reading order)
| Name | Link | Note |
|---|---|---|
| PaddleOCR / PP-StructureV3 | https://github.com/PaddlePaddle/PaddleOCR | Layout + formula/table detection |
| LayoutReader | https://arxiv.org/abs/2108.11591 | Reading order (printed) |
| DocLayNet (dataset) | https://github.com/DS4SD/DocLayNet | Printed layout |
| Marker | https://github.com/VikParuchuri/marker | PDF→Markdown pipeline reference |
| olmOCR (AllenAI) | https://github.com/allenai/olmocr | Printed PDF OCR |
| DIBCO (binarization benchmarks) | https://dib.cin.ufpe.br | Degraded-doc binarization |

## 8. Diagrams → TikZ (research; we EMBED images instead, D7)
| Name | Link | Note |
|---|---|---|
| DeTikZify | https://arxiv.org/abs/2405.15306 · https://github.com/potamides/DeTikZify | Sketch→TikZ, draft quality |
| AutomaTikZ / DaTikZ | https://arxiv.org/abs/2310.00367 | Text→TikZ |
| Sketch2Diagram / SkeTikZ | https://sketikz.github.io/ | Real hand-drawn sketch↔TikZ benchmark |

## 9. Grading-context research
| Name | Link | Finding |
|---|---|---|
| GPT-4 grading handwritten solutions | https://arxiv.org/abs/2411.05231 | 46.67% human agreement, MAE 7.66% — too low for autonomous grading |
| *"When VLMs 'Fix' Students"* (over-correction / PINK metric) | https://arxiv.org/abs/2604.22774 | ✅ VLMs "fix" student errors 42–66%; PINK metric penalizes over-correction |
| DrawEduMath | https://www.emergentmind.com/topics/drawedumath | 60–70% accuracy; default-to-correct bias |
| Gradescope (product) | https://www.gradescope.com | AI answer-grouping; keeps handwriting as images, no LaTeX |

## 10. Compile / render (for output)
| Name | Link |
|---|---|
| KaTeX | https://katex.org |
| MathJax | https://www.mathjax.org |
| TeX Live | https://tug.org/texlive |

---

_Back to: [README](../README.md) · [01-landscape](01-landscape.md) (what these mean for us) · [DECISIONS](../DECISIONS.md)_
