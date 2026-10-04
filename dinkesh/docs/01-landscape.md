# 01 — Landscape: Models, Tools, Datasets & Services

> Research-backed survey (state of the art as of **2025–2026**) of everything that could help turn handwritten math answer sheets into LaTeX.
> **Reading note:** metrics and claims are cited inline. The three 2025–2026 papers previously flagged for verification (FERMAT, VEHME, the over-correction/PINK paper) have now been **confirmed against arXiv** — exact titles added below. Remaining ⚠️ marks denote real cautions (non-commercial licenses, data-residency, no FERPA clause), not unverified sources. Treat vendor "accuracy" numbers as marketing unless a methodology is cited.

---

## TL;DR — the five things that matter

1. **Printed math → LaTeX is essentially solved. Handwritten math → LaTeX is not.** The polished open-source tools (pix2tex, Nougat, Texify/Surya, Pix2Text) are **printed-first**; genuine handwriting support lives in the academic **HMER** research line, which only recognizes **single isolated equations** at **~62–80% exact-match** on *clean* benchmarks.
2. **No off-the-shelf system does our actual task** — a *full handwritten page* of mixed prose + multi-line equations + proofs → coherent LaTeX. That is an **open research problem**; any build today is a **staged pipeline**, not one model.
3. **General VLMs (GPT-4o, Claude, Gemini) silently "correct" students' wrong math in 42–66% of transcriptions** — and this gets *worse* with more capable models. For grading (where the mistake is the point), that is **disqualifying** for a naive "VLM reads and grades" approach.
4. **Mathpix is the most deployable commercial recognizer** — real handwriting support, LaTeX output, **per-symbol confidence scores** (ideal for human-in-the-loop routing), ~**$0.005/page**, training-on-your-data **off by default**, on-prem option. But no rigorous third-party benchmark on handwritten IIT-level math exists.
5. **The dataset we need does not exist.** All major datasets are *isolated expressions*. Full handwritten *solutions/proofs* with LaTeX ground truth only started appearing in 2025 (FERMAT, VEHME) and are **school-grade (7–12), small, and not IIT-level.** Expect to **build our own corpus.**

---

## 2. Specialized open-source / academic models

### 2.1 Printed-first tools (do NOT genuinely handle handwriting)

| Tool | What it does | Handwriting? | Reported accuracy (printed) | License | Maturity |
|---|---|---|---|---|---|
| **pix2tex / LaTeX-OCR** ([repo](https://github.com/lukas-blecher/LaTeX-OCR)) | Single equation image → LaTeX (ResNet+ViT encoder, Transformer decoder) | ⚠️ Experimental only — README: *"support handwritten formulae (kinda done)"* | BLEU 0.88, norm. edit dist 0.10, token acc 0.60 | **MIT** | Mature (~16.5k★), equation-only |
| **Nougat** (Meta) ([repo](https://github.com/facebookresearch/nougat), [paper](https://arxiv.org/abs/2308.13418)) | Academic **PDF page** → Markdown + LaTeX math + tables | ❌ None (trained on arXiv/PMC printed) | Edit-dist/BLEU/F1 on arXiv; repetition-loop hallucination on OOD pages | Code MIT; **weights CC-BY-NC** (non-commercial ⚠️) | Well-known (~10k★) |
| **Texify** ([repo](https://github.com/VikParuchuri/texify)) | Equation + context → MathJax/LaTeX (Donut-based) | ❌ | BLEU 0.842 (beats Nougat 0.698, pix2tex 0.383) | CC BY-SA 4.0 | **DEPRECATED — archived Jan 2025**, migrated to Surya |
| **Surya** (Texify successor) ([repo](https://github.com/datalab-to/surya)) | Full-page OCR + inline equation recognition (KaTeX LaTeX) | ~Better on notes generally, **no HW-math claim** | ~83.3% on olmOCR-bench (overall) | Code Apache 2.0; **weights OpenRAIL-M** (commercial gated ⚠️) | Active |
| **Pix2Text** ([repo](https://github.com/breezedeus/Pix2Text)) | Open "free Mathpix": layout+tables+formula detect/recognize → Markdown, 80+ langs | ❌ No HW-math documented | Claims SOTA on own MFD/MFR, no comparable metric | **MIT** | Active (~3.2k★) |
| **GOT-OCR2.0** ([paper](https://arxiv.org/abs/2409.01704)) | Unified OCR-2.0: text/formulas/tables/charts/geometry → text/markdown/tikz | Not a stated target | — | — | Notable 2024 model |

**Takeaway:** great as **architectural references** and for the *printed* parts of a pipeline, but none solve handwritten input.

### 2.2 Handwriting text recognition (prose, NOT math)

- **TrOCR (Microsoft)** — [`trocr-base-handwritten`](https://huggingface.co/microsoft/trocr-base-handwritten). BEiT encoder + RoBERTa decoder. **Genuinely handles English handwriting**, but **plain text only — no math / no 2-D structure.** MIT. **Good candidate for the prose portions** of answers; would need heavy fine-tuning (and isn't architecturally suited) for equations.

### 2.3 HMER — the real handwriting track (isolated single equations)

This research line is where genuine **handwritten math** recognition lives. **All models are single-equation** (input = one cropped expression). Metric = **ExpRate** (% of expressions matched *exactly*, all-or-nothing). Numbers = offline CROHME test sets; HME100K = larger real-photo set.

| Model | Venue | CROHME'14 | '16 | '19 | HME100K |
|---|---|---|---|---|---|
| BTTR | ICDAR'21 | 53.96 | 52.31 | 52.96 | — |
| SAN | — | 56.2 | 53.6 | 53.5 | 67.1 |
| CoMER ([paper](https://arxiv.org/abs/2207.04410)) | ECCV'22 | 59.33 | 59.81 | 62.97 | — |
| ICAL ([paper](https://arxiv.org/abs/2405.09032)) | ICDAR'24 | ~60.6 | ~58.7 | ~60.4 | 69.06 |
| **PosFormer** ([paper](https://arxiv.org/abs/2407.07764)) | ECCV'24 | **62.68** | **61.03** | **64.97** | **69.51** |
| **Uni-MuMER** (fine-tuned Qwen2.5-VL-3B) ([paper](https://arxiv.org/abs/2505.23566)) | NeurIPS'25 | **~82** | **~78** | **~79** | **71.93** |

**Reality check:** even 2024–2025 SOTA is **~62–80% exact-match on clean, *isolated* equations** — roughly **1 in 4–5 still wrong**, and CROHME expressions are far simpler and cleaner than IITG proof steps. Multi-line / nested / structured expressions are markedly worse. **None** of these segment a page, handle prose, or understand proof narrative.

### 2.4 Fine-tuned VLM recognizers (2024–2026 momentum)

Hugging Face lists ~95 fine-tuned "LaTeX-OCR" VLMs (e.g. `Qwen2-VL-7B-Latex-OCR`, `paligemma2_latex_ocr`, `qwen2.5-vl-latex-ocr`). **Caveat:** most are fine-tuned on the *printed* `linxy/LaTeX_OCR` dataset → inherit printed-only bias. **Uni-MuMER** (above) is the standout that fine-tunes for handwriting and beats frontier VLMs by ~20–50 ExpRate points.

---

## 3. General VLMs (GPT-4o/4.1, Claude, Gemini, Qwen-VL, InternVL, Llama-V)

### 3.1 The single most important finding for grading — over-correction / hallucination
- **42.1%–66.2% of VLM transcriptions silently "fix" a student's errors**, and **larger/more capable models over-correct *more*** ("reasoning suppresses visual grounding"). ✅ Confirmed: [*"When VLMs 'Fix' Students: Identifying and Penalizing Over-Correction in the Evaluation of Multi-line Handwritten Math OCR"*, arXiv:2604.22774](https://arxiv.org/abs/2604.22774) — introduces the **PINK** metric that penalizes over-correction; tested 15 VLMs on FERMAT (GPT-4o penalized for aggressive correction, Gemini 2.5 Flash more faithful).
- **DrawEduMath** (2,030 student handwritten-math images): top models 60–70% accuracy; **29–35% of model mistakes on erroneous work replicate the majority-correct solution** ("default-to-correct bias"). ([overview](https://www.emergentmind.com/topics/drawedumath))
- **FERMAT** finding: accuracy rises monotonically as input goes handwritten image → printed → plain text — i.e. **reading the handwriting, not the math reasoning, is the bottleneck.** ([paper](https://arxiv.org/abs/2501.07244))

> **Design consequence:** never let a single VLM's transcription stand as ground truth, and never fold transcription+grading into one uninspected step. This is baked into `02` and `03`.

### 3.2 Zero-shot recognition accuracy (ExpRate) — from Uni-MuMER
| System | CROHME'14 | '16 | '19 | HME100K (realistic) |
|---|---|---|---|---|
| GPT-4o zero-shot | 50.61 | 46.03 | 49.79 | **22.96** |
| Gemini 2.5-flash zero-shot | 58.01 | 52.74 | 55.21 | **28.14** |
| Uni-MuMER (fine-tuned 3B) | 82.05 | 77.94 | 79.23 | **71.93** |

On realistic photographed handwriting (HME100K), frontier VLMs get only **~23–28% of expressions fully correct** — a fine-tuned 3B specialist triples that.

### 3.3 Per-model notes
| Model | Notes for our use case | Self-host? |
|---|---|---|
| **GPT-4o / 4.1** (OpenAI) | Capable but "insufficient for practical use" as grader (46.67% exact agreement w/ humans, MAE 7.66% — [paper](https://arxiv.org/html/2411.05231v2)); heavy over-corrector | No |
| **Claude Opus/Sonnet** (Anthropic) | ~60–70% tier (DrawEduMath); strongest **default privacy** posture (no training on data by default) | No |
| **Gemini 2.x** (Google) | **Best faithful transcriber** in fidelity-aware tests; best FERMAT error-corrector (77%). ⚠️ **Free tier trains on your data** — must use paid tier | No |
| **Qwen2.5-VL** (Alibaba) | 72B matches GPT-4o/Claude on doc understanding ([paper](https://arxiv.org/abs/2502.13923)); **open-weight → self-hostable**; best fine-tune base (Uni-MuMER) | **Yes** |
| **InternVL** | Competitive general OCR; open-weight | **Yes** |
| **Llama 3.2 Vision** (Meta) | Weakest here (#24 / 50.8% on IDP leaderboard); open-weight | **Yes** |

**Structural failure mode (all VLMs):** even with correct symbols, **2-D structure** (nested fractions, matrices, sub/superscripts) breaks as complexity rises, and models emit confident wrong LaTeX rather than abstaining.

---

## 4. Commercial services

### 4.1 Mathpix — the leading math OCR
- **Handwriting: first-class, marketed.** Recognizes printed + handwritten math, text, tables, chemistry; response has `is_handwritten` flag. Handwritten *text* documented for Latin + Hindi; English prose likely the weaker link. ([docs](https://docs.mathpix.com/reference/introduction))
- **API (mature):** `v3/text`, `v3/pdf`, `v3/batch`, `v3/strokes`, Files API, on-prem Secure Conversion Service; official Python SDK. Tunable `confidence_threshold` / `confidence_rate_threshold` (default 0.75) → **auto-flag low-confidence handwriting for human review.** ([docs](https://docs.mathpix.com/reference/post-v3-text))
- **Output:** LaTeX (`latex_styled`), Mathpix Markdown, MathML, AsciiMath, HTML, SMILES; DOCX/XLSX/PPTX/PDF export; per-line/word data with `confidence` (0–1).
- **Pricing** ([source](https://mathpix.com/pricing/api)): images **$0.002** each (≤1M/mo); **PDF $0.005/page**; strokes $0.005–0.01/session; $19.99 one-time setup; **$29 free credit**. ⚠️ Images with >12 rows of text bill at PDF page rate → full exam pages ≈ **$0.005/page (~$5 per 1,000 pages)**.
- **Accuracy claims:** only marketing ("industry leader"); a third-party "99%" figure is **unverified**. One independent *printed* benchmark ([Igor Rivin](https://igorrivin.github.io/blog/ocr-benchmark/)) found Gemini more accurate & cheaper and documented Mathpix errors. **No rigorous third-party handwritten-math benchmark exists.**
- **Privacy (favorable):** training on your data **off by default** (opt-in flag); SOC 2; auto-collected data ≤30 days; **on-prem** option. ⚠️ **No explicit FERPA/education clause — needs legal review** before student PII. ([privacy](https://mathpix.com/privacy))

### 4.2 General cloud OCR — mostly text-only; LaTeX add-ons exist but are typeset-oriented
| Service | Handwritten text | Math → LaTeX? |
|---|---|---|
| **Google Document AI** | Strong | 🟡 LaTeX via add-on/processor, but **built for typeset math, unvalidated on handwriting** |
| **Azure AI Document Intelligence** | Strong | 🟡 LaTeX for formulas, but typeset-oriented; confidence hard-coded |
| **Google Cloud Vision** (basic) | Strong | ❌ text only (used as a *baseline* recognizer in Google's MathWriting paper) |
| **AWS Textract** | Yes (text/forms/tables) | ❌ No LaTeX |

**Correction from initial research:** Google Document AI and Azure Document Intelligence *do* expose LaTeX formula output, but these are engineered for **printed/typeset** documents and are **not validated on handwriting** — treat as unproven for our case. Basic Vision/Textract flatten equations into garbled linear text. Enterprise DPAs + regional residency generally make these acceptable for student data *with a signed agreement*, but none is a proven handwritten-math solution.

### 4.3 Other services (fitness for **offline scanned** handwritten pages)
| Service | HW math? | Output | License / price | Residency |
|---|---|---|---|---|
| **SimpleTex** ([site](https://simpletex.net)) | ✅ flagship | LaTeX/MathML + confidence | Free tiers (Turbo 2k/day); paid RMB credits | 🔴 **HK + mainland China servers, PRC-law ToS, Baidu analytics — likely disqualifying for student PII; no on-prem** |
| **MyScript iink** ([dev](https://developer.myscript.com)) | ✅ best for **stroke/ink** (live tablet), weak on scans | LaTeX, MathML | Commercial SDK, quote-based | ✅ on-device / EU — strong |
| **Surya 2** ([repo](https://github.com/datalab-to/surya)) | ✅ full-page incl. cursive | KaTeX LaTeX | Apache-2 code; OpenRAIL-M weights | ✅ self-host |
| **Photomath / MS Math Solver** | ✅ (app) | ❌ no LaTeX API | consumer app | n/a |

**Bottom line on services:** MyScript is best for *live ink*, not scanned pages. For **offline scanned pages while keeping student data in-jurisdiction**, self-hosted VLM OCR (Surya 2, GOT-OCR2.0, or fine-tuned Qwen2.5-VL) is the only route that both handles handwriting and avoids third-party / offshore data transfer. Mathpix (§4.1) remains the best-supported commercial option if its privacy posture clears legal review.

---

## 5. Datasets & the critical gap

### 5.1 Isolated-expression datasets (the field's mainstay)
| Dataset | Size | Online/Offline | Scope | Ground truth | License |
|---|---|---|---|---|---|
| **CROHME** (2011–2023) | ~8,836 train; test'14=986/'16=1,147/'19=1,199; CROHME23 ~164k inks (~10% human) | Online (InkML) + offline images (2019+) | **Isolated expressions** | LaTeX + MathML + stroke labels | Per-edition ⚠️ |
| **MathWriting** (Google 2024) ([paper](https://arxiv.org/abs/2404.10690)) | ~650k inks (253k human + 396k synthetic); largest to date | Online (renderable offline) | **Isolated expressions**, *not* documents | LaTeX + normalized LaTeX | **CC-BY-NC-SA 4.0** (non-commercial ⚠️) |
| **HME100K** ([CAN paper](https://arxiv.org/abs/2207.11463)) | ~99,109 real-photo images; 249 classes | **Offline real photos** (blur, color, backgrounds) | **Isolated expressions** | LaTeX | license ⚠️ reconfirm |
| **im2latex-100k** ([paper](https://arxiv.org/abs/1609.04938)) | 103,556 formulas | Offline, **PRINTED** (arXiv-rendered) | Isolated printed | Rendered LaTeX | Open |
| **IAM** | ~1,539 pages / 13,353 lines / 657 writers | Offline | English **prose** (no math) | Plain text | Research |

### 5.2 The gap you actually care about — **confirmed**
Every major dataset is **isolated single expressions**. Full multi-line **solutions/proofs** with LaTeX ground truth only emerged in 2025, and only at **school level (grades 7–12), small scale**:

- **FERMAT** (Jan 2025, IIT Madras / AI4Bharat — directly relevant to Indian context) — ✅ [*"Can Vision-Language Models Evaluate Handwritten Math?"*, arXiv:2501.07244](https://arxiv.org/abs/2501.07244) (Nath, Bathina, Khan, Khapra): **2,200+ handwritten solutions**, 609 problems, **grades 7–12**, multi-line derivations, **LaTeX ground truth**, error taxonomy (computational/conceptual/notational/presentation). Key result: VLMs degrade sharply on handwritten vs printed; best error-correction 77% (Gemini-1.5-Pro).
- **VEHME** (Oct 2025) — ✅ [*"VEHME: A Vision-Language Model For Evaluating Handwritten Mathematics Expressions"*, arXiv:2510.22798](https://arxiv.org/abs/2510.22798): open VLM for **evaluating** open-form handwritten math; two-phase training (SFT + RL) with an Expression-Aware Visual Prompting Module; evaluated on AIHub + FERMAT; competitive with proprietary systems; code on GitHub.
- **AutoOpt** (2025): ~11k handwritten+printed *optimization-problem* images w/ LaTeX — niche.

> **No public dataset covers university/IIT-level handwritten proofs with LaTeX ground truth.** FERMAT (school-grade, 2,244 solutions) is the nearest analog. **Plan to build/annotate a bespoke corpus**, bootstrapping recognizers from MathWriting/HME100K (expressions) + IAM (prose).

---

## 6. License & privacy cheat-sheet
| Asset | License / posture | Commercial-safe? |
|---|---|---|
| pix2tex, Pix2Text, TrOCR | MIT | ✅ |
| Texify | CC BY-SA 4.0 | ✅ (share-alike) |
| Nougat **weights** | CC-BY-NC | ❌ non-commercial |
| Surya **weights** | OpenRAIL-M (gated) | ⚠️ gated |
| MathWriting dataset | CC-BY-NC-SA 4.0 | ❌ non-commercial |
| Mathpix | Off-by-default training; SOC2; on-prem | ✅ with legal review (no FERPA clause) |
| Qwen2.5-VL / InternVL / Llama-V | Open-weight (self-host) | ✅ **privacy-optimal** |
| Gemini free tier | **Trains on your data** | ❌ use paid tier |
| SimpleTex | China-hosted | ⚠️ residency risk |

---

## 7. What this means for the approach (hand-off to `02`)
1. **Composite pipeline, not one model** — no single system does full handwritten page → LaTeX.
2. **Separate transcription from grading** — over-correction makes fused VLM grading unsafe.
3. **Confidence-gated human-in-the-loop** — mandatory; Mathpix confidence or model logprobs route hard cases to humans.
4. **Two viable recognizer bets:** Mathpix (fastest, confidence scores, ~$0.005/page) **or** a fine-tuned open Qwen2.5-VL specialist (privacy-optimal, ~72–80% ExpRate).
5. **Budget for building data** — the IIT-level solution corpus doesn't exist.
6. **Diagrams → embed as images**, don't attempt TikZ (see `02`/`03`).

*Method note: research was gathered via direct arXiv/GitHub/vendor-doc fetches. The three 2025–2026 papers once flagged for verification (FERMAT 2501.07244, VEHME 2510.22798, over-correction/PINK 2604.22774) have since been confirmed against arXiv with exact titles. SimpleTex specifics and vendor pricing should still be reconfirmed at citation time as they can change.*
