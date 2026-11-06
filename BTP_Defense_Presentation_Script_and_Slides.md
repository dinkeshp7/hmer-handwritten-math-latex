# BTP Defense Presentation: Complete 15-Slide Deck, Speaker Script & Defense Strategy
## *Automated Full-Page Handwritten Mathematical Expression Recognition (HMER) and Exam Script-to-LaTeX Transcription*
### *High-Resolution Multi-Band Vision-Language Inference on PARAM Kamrupa Supercomputer*

**Date of Defense:** November 7, 2026  
**Candidates:** Dinkesh Pal (Roll No. 230123021) & Rithvik Ponnapalli (Roll No. 230123052)  
**Supervisor:** Dr. K. V. Srikanth (Assistant Professor), Department of Mathematics, IIT Guwahati  
**Evaluation Committee:**  
* **Dr. K. V. Srikanth** (Assistant Professor, Supervisor)  
* **Dr. H. Ramesh** (Associate Professor, DUPC Member Secretary — *Research Area: Word Representable Graphs, Formal Languages and Automata Theory*)  
* **Dr. Vinay Wagh** (Associate Professor)  
**HPC Infrastructure:** PARAM Kamrupa Supercomputer (CCC, IIT Guwahati)  
**Target Duration:** 15–18 minutes (15 slides, ~1.0–1.2 minutes per slide) + 5–10 minutes Committee Q&A  

---

# PRESENTATION STRUCTURE & TIMING MAP (15 SLIDES)

| Slide | Title & Core Topic | Speaker Focus | Target Time |
| :---: | :--- | :--- | :---: |
| **1** | **Title Slide & Visual Contract** | Greetings, team intro, institutional scope, Ink-to-LaTeX contract | 1.0 min |
| **2** | **Problem Definition, Scale & Scope** | MA102 900+ students, 13,146 scans, Assist Don't Grade philosophy | 1.0 min |
| **3** | **Hardware Bottlenecks on PARAM Kamrupa** | 16 GB Tesla V100 ceiling, $O(N^2)$ ViT crash, Lustre scratch, systems hacks | 1.2 min |
| **4** | **Evolution of Approaches (Roadmap)** | 4-phase iterative journey: GANs $\to$ Hybrid $\to$ Monolithic $\to$ Multi-Band | 1.2 min |
| **5** | **Algorithmic Deep-Dive: Modular Hybrid Pipeline** | Sauvola binarization, Vertical IoU, MatrixGridParser, 2D DAG sort | 1.2 min |
| **6** | **Attention Coverage Math & Custom DenseNet** | 23.1M params, coverage vector $c_t$, $\mathcal{L}_{\text{cov}}$ solving over/under-parsing | 1.2 min |
| **7** | **Master Architecture: Multi-Band Slicing** | Header crop, 3 overlapping bands (8% buffer), 1200 px native resolution | 1.2 min |
| **8** | **Mathematical Proof Case Study (Student 1)** | Roll 240101079, MA102 Q1(a) subspace proof breakdown across 3 bands | 1.2 min |
| **9** | **Visual Verification: Actual Ink vs. Compiled PDF** | Side-by-side visual proof: 300 DPI raw ink crop vs. Tectonic PDF typeset | 1.0 min |
| **10** | **Quantitative Accuracy Benchmarks** | eval_metrics.py edit distance, 88.24% accuracy, $+28.2\%$ over monolithic | 1.0 min |
| **11** | **Edge Cases, Model Failure Modes & Fixes** | Roll 240122044: Tutor prior fixed by negative prompt; blank loop fixed by Laplacian | 1.2 min |
| **12** | **Automated LaTeX Stitching & Compilation** | Prefill locking, CFG Pushdown Automaton LIFO stack, underscore masking, Tectonic | 1.2 min |
| **13** | **Architectural Comparison Matrix** | 7-metric deep comparison: Monolithic Full-Page vs. 3-Band Slicing | 1.0 min |
| **14** | **Summary of Contributions & Phase 2 Roadmap** | Phase 1 systems achievements vs. Phase 2 QLoRA/INT4 & faculty grading UI | 1.0 min |
| **15** | **Conclusion, Deliverables & Q&A** | Summary, verified repository artifacts, acknowledgements, opening for Q&A | 1.0 min |

---

# SLIDE 1: Title Slide & Visual Contract

### Slide Content (Visual Layout)
* **Title:** Automated Full-Page Handwritten Mathematical Expression Recognition (HMER) and Exam Script-to-\LaTeX\ Transcription
* **Subtitle:** High-Resolution Multi-Band Vision-Language Inference on PARAM Kamrupa Supercomputer
* **Candidates:** Dinkesh Pal (230123021) & Rithvik Ponnapalli (230123052)
* **Supervisor:** Dr. K. V. Srikanth (Assistant Professor)
* **Evaluation Committee:** Dr. K. V. Srikanth, Dr. H. Ramesh (DUPC Sec.), Dr. Vinay Wagh
* **Department:** Department of Mathematics & Computer and Communication Centre (CCC), IIT Guwahati
* **Visual Ink $\to$ LaTeX Contract Bar:**
  * Raw Handwritten Ink (300 DPI): `[ -1.0  8.5 ] [  4.5  2.5 ]` (4x4 unruled matrix ink with decimals)
  * Verbatim Compilable LaTeX: `\begin{bmatrix} -1.0 & 8.5 \\ 4.5 & 2.5 \end{bmatrix}` (100% PDF pass rate)

### Speaker Script (What to Say)
> *"Respected Supervisor Dr. Srikanth, respected committee members Dr. Ramesh and Dr. Wagh, good morning.
>
> Today, Rithvik and I are presenting our B.Tech Project – I titled **'Automated Full-Page Handwritten Mathematical Expression Recognition and Exam Script-to-LaTeX Transcription'**.
>
> This is an applied machine learning and systems engineering project. In this work, we address the computational and algorithmic bottlenecks of transcribing dense, unconstrained university mathematics examination scripts into compilable, production-ready LaTeX on the PARAM Kamrupa Supercomputer.
>
> Over the semester, we progressed from classical modular computer vision pipelines and custom DenseNet models to large multimodal vision-language models. To overcome severe hardware constraints on our cluster, we engineered a novel **Multi-Band High-Resolution Slicing and Automated LaTeX Stitching Architecture** that achieves **88.24% word-for-word token accuracy** on authentic IIT Guwahati exam scripts.
>
> Let us begin with the institutional problem and scale that motivated this research."*

### What NOT to Say
* ❌ *Don't say:* "We built an AI that replaces evaluators and grades exams." (Emphasize *assistive transcription*, never auto-grading).
* ❌ *Don't say:* "We called OpenAI or Gemini cloud APIs." (Exams are private institutional documents; emphasize *100% air-gapped on-premise inference on PARAM Kamrupa*).

---

# SLIDE 2: Problem Definition, Scale & Strict Project Scope

### Slide Content (Visual Layout)
* **Institutional Scale (MA102 Context at IIT Guwahati):**
  * Compulsory core course: **MA102 (Linear Algebra & Calculus)**.
  * Enrollment: **900+ undergraduate students** every spring semester.
  * Examination booklet: 14 unruled pages $\implies$ **$\approx 13,146$ scanned pages per exam cycle**.
  * Evaluator Cognitive Burden: Faculty and TAs spend over 60 cumulative hours deciphering handwriting, leading to grading fatigue.
* **The Fundamental Vision-Language Challenge:**
  * Real exam scripts mix cursive prose with multi-dimensional mathematics (matrices, piecewise functions, systems of ODEs).
  * Extreme visual noise: scratched-out errors, margin notes, ink bleed-through, baseline slant ($\pm 15^\circ$).
* **Strict Project Scope: Assist, Don't Grade:**
  * **What the System Does:** 100% verbatim, faithful transcription of student ink into compilable LaTeX, preserving student errors and notation.
  * **What the System NEVER Does:** It does NOT evaluate, judge correctness, or award marks. It is purely an assistive transcription engine.

### Speaker Script (What to Say)
> *"To understand why this is a challenging computer vision and document intelligence problem, consider the sheer scale of MA102 here at IIT Guwahati. With over 900 students submitting 14-page physical booklets, each exam generates over 13,000 scanned pages.
>
> Faculty and teaching assistants spend dozens of grueling hours deciphering handwriting across multi-line derivations. Standard OCR engines like Tesseract completely fail because they assume linear 1D text, whereas mathematics is inherently a 2D spatial tree.
>
> Furthermore, we want to establish a strict engineering scope: our system is an assistive transcription engine, not an auto-grader. It transcribes student ink verbatim—preserving mistakes, cross-outs, and notation—producing a clean, compilable LaTeX document so evaluators can read and grade with clarity without eye fatigue."*

---

# SLIDE 3: Hardware Bottlenecks & Systems Engineering on PARAM Kamrupa

### Slide Content (Visual Layout)
* **Cluster Environment & Memory Arithmetic:**
  * PARAM Kamrupa HPC partition: NVIDIA Tesla V100 PCIe (16 GB VRAM limit = 15.78 GiB addressable).
  * 7B Parameter Model (FP16 weights): **14.07 GiB**.
  * Available headroom for activations + KV cache: **Only 1.71 GiB (1,750 MiB)**!
* **The Monolithic Failure ($O(N^2)$ ViT Self-Attention):**
  * Full 300 DPI uncompressed exam scan ($1653 \times 2338$ px) generates **> 5,800 image patches**.
  * ViT self-attention matrix ($N \times N$) allocates contiguous 1.30 GiB block $\implies$ **Instant CUDA Out-Of-Memory (OOM) Crash**.
  * Downscaling to fit VRAM compresses width to 848 px $\implies$ 30 lines compressed $\implies$ subscripts, decimals, and minus signs blur together ($3.5 \to 8.5$).
* **Systems Engineering Hacks on PARAM Kamrupa:**
  * **Lustre Scratch Storage:** Bypassed 50 GB home quota by re-routing checkpoints to 10 TB Lustre parallel filesystem.
  * **CentOS 7 libstdc++ Preload:** Intercepted legacy OS glibc via `ctypes.CDLL` preload before PyTorch initialization.
  * **Volta-Optimized Attention:** Enforced Cutlass fallback kernels for FP16 inference on Volta microarchitecture.
  * **Automated SLURM Batching:** Bash worker scripts with auto-resubmission and signal traps on 4-hour queue limits.

### Speaker Script (What to Say)
> *"When we scaled to 7-billion parameter vision-language models, we encountered a severe hardware bottleneck on the PARAM Kamrupa Supercomputer.
>
> Our compute nodes feature NVIDIA Tesla V100 GPUs with a 16 GB VRAM ceiling. The FP16 model weights alone consume 14.07 GiB, leaving less than 1.7 GiB of memory headroom for all visual activations and KV cache.
>
> Because Vision Transformer attention scales quadratically with image patch count—$O(N^2)$—feeding an uncompressed exam page with over 5,800 patches caused instant CUDA Out-Of-Memory crashes. Conversely, downscaling the page to 848 pixels compressed 30 lines of student writing, turning critical minus signs and decimal points into indistinguishable blurs.
>
> Additionally, we engineered several systems-level solutions: routing model weights to Lustre parallel scratch storage to avoid home directory quota kills, dynamic library preloading to resolve CentOS 7 compatibility, and automated SLURM batch scheduling with signal traps for multi-hour jobs."*

---

# SLIDE 4: Evolution of Approaches (Roadmap)

### Slide Content (Visual Layout)
* **Phase 1A: Two-Stage GAN Pipeline (August 2026):**
  * Synthetic paired dataset generator: GAN font stylization + OpenCV degradation.
  * Finding: High formula-level accuracy on CROHME, but completely failed on multi-line page layouts and cursive English text.
* **Phase 1B: Modular Hybrid Architecture (September 2026):**
  * Sauvola binarization + Projection profile segmentation + TrOCR + Mathpix/SymPy parser.
  * Finding: Severe compounding pipeline error. If line segmentation shifted by 5 pixels, matrices broke; TrOCR drifted into 19th-century legal prose (*"House of Representatives... Chamberlain v."*).
* **Phase 1C: Monolithic Vision-Language Model (October 2026):**
  * Direct full-page inference with Qwen2.5-VL-7B.
  * Finding: Either CUDA OOM on high resolution, or severe hallucination and dropped equations on downscaled images.
* **Phase 1D: Master Solution — Multi-Band High-Resolution Slicing (November 2026):**
  * 3 overlapping horizontal bands at native 1200 px resolution + automated LaTeX stitching $\implies$ Zero OOM, 88.24% accuracy!

### Speaker Script (What to Say)
> *"Our project followed a rigorous, iterative systems engineering journey across four distinct architectures:
>
> In Phase 1A, we explored a Two-Stage GAN approach. While effective on isolated synthetic formulas, it completely collapsed on full exam pages mixing English prose with math.
>
> In Phase 1B, we developed a modular pipeline: classical line segmentation followed by TrOCR. However, errors compounded at every interface: a 5-pixel segmentation error fractured matrices, and TrOCR suffered catastrophic historical text drift, hallucinating legal citations from its 19th-century pre-training corpus.
>
> In Phase 1C, we tested end-to-end 7B Vision-Language Models. While representationally powerful, feeding the monolithic page caused either immediate 16 GB VRAM OOM crashes or dropped equations due to downscaling.
>
> This critical realization led to Phase 1D: our Multi-Band High-Resolution Slicing architecture, which preserves full optical resolution while cutting ViT memory consumption by 82%."*

---

# SLIDE 5 [NEW]: Algorithmic Deep-Dive: Modular Hybrid Pipeline & MatrixGridParser

### Slide Content (Visual Layout)
* **Modular Pipeline Components:**
  * **Sauvola Adaptive Binarization:** Dynamic local thresholding ($k=0.2, R=128$) to handle uneven scanner illumination and paper grain.
  * **1D Vertical IoU ($\ge 0.35$) Line Clustering:** Prevents tall mathematical symbols (integrals, brackets, fractions) from being falsely fractured into multiple text lines.
  * **2D Spatial Directed Acyclic Graph (DAG):** Linearizes multi-column math and marginal annotations into topological reading order.
* **Algorithmic Feature: MatrixGridParser:**
  * Computes horizontal and vertical pixel projection profiles to isolate inner grid cell boundaries.
  * Extracts isolated cell crops, transcribes each cell independently, and formats them into clean `\begin{bmatrix} ... \end{bmatrix}` syntax.
* **Real Visual Embedded Artifact:**
  * Embedded crop of authentic 4x4 matrix from student exam script (`crop_8_matrix.png`).

### Speaker Script (What to Say)
> *"Before transitioning to large multimodal models, we built a fully autonomous modular computer vision pipeline from first principles.
>
> To handle severe scanner noise and faint pencil strokes, we implemented Sauvola adaptive binarization. To segment lines without splitting tall symbols like integrals or square roots, we developed a 1D Vertical Intersection-over-Union merger: bounding boxes with vertical overlap exceeding 35% are merged into single cohesive derivation lines.
>
> For complex 2D structures like matrices, we engineered `MatrixGridParser`. As shown in the embedded matrix crop from our dataset, the algorithm projects horizontal and vertical density profiles to locate inner cell partitions, transcribes each cell, and formats it directly into LaTeX `bmatrix` syntax.
>
> To serialize non-linear structures and marginal notes, we formulated a 2D Spatial Directed Acyclic Graph, performing a topological sort to establish unambiguous reading order—a formal language and graph-theoretic approach that laid the groundwork for our data structuring."*

---

# SLIDE 6 [NEW]: The Custom DenseNet-Coverage Model & Attention Coverage Math

### Slide Content (Visual Layout)
* **Custom Architecture (23.1M Parameters, Trained from Scratch):**
  * DenseNet Encoder (DenseBlock $\times 4$, growth rate $k=24$) with transition layers and multi-scale feature reuse.
  * Coverage-Aware Spatial Attention Decoder (256-dim GRU with 2D spatial feature mapping).
* **The Mathematical Innovation: Attention Coverage ($\mathcal{L}_{\text{cov}}$):**
  * **The Problem:** Standard seq2seq attention suffers from *over-parsing* (repeating the same handwritten symbol) and *under-parsing* (skipping indices or minus signs).
  * **Cumulative Coverage Vector:**
    $$c_t = \sum_{\tau=1}^{t-1} \alpha_\tau$$
  * **Coverage Loss Penalty:**
    $$\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda_{\text{cov}} \sum_{t=1}^{T} \sum_{i=1}^{H \times W} \min(c_{t,i}, \alpha_{t,i})$$
  * When current attention $\alpha_t$ focuses on previously attended locations ($c_t > 0$), the penalty spikes, mathematically forcing the model to scan unparsed handwriting!
* **Performance:** Achieved **75.0% Exact Match** and **2.13% TER** on benchmark mathematical formulas.

### Speaker Script (What to Say)
> *"In Chapter 3 of our report, we developed and trained a custom deep neural network from scratch: a 23.1-million parameter DenseNet with Attention Coverage.
>
> A notorious failure mode in mathematical expression recognition is attention drift: the decoder either repeats symbols—known as over-parsing—or skips small characters like superscripts and minus signs—under-parsing.
>
> To eliminate this mathematically, we implemented an Attention Coverage mechanism. We maintain a cumulative coverage vector $c_t$, which accumulates the historical attention distributions across all prior decoder timesteps. We then penalize the model whenever the current attention map overlaps with previously visited spatial coordinates via a dedicated coverage loss term, weighted by $\lambda_{\text{cov}}$.
>
> This forces the attention mechanism to monotonically advance across the student's handwriting, eliminating repetitive loops and achieving a 75% exact match rate on mathematical formulas."*

---

# SLIDE 7: The Master Solution: Multi-Band High-Resolution Slicing Architecture

### Slide Content (Visual Layout)
* **Core Hypothesis:** Human examiners do not grade an entire exam sheet in one glance; they read it section-by-section. Slicing the page horizontally feeds native optical clarity into the VLM without exceeding VRAM.
* **Three-Stage Architecture:**
  1. **Booklet Instruction Stripping:** Crops the printed IITG header box ($y \le 0.18H$), eliminating non-answer booklet boilerplate.
  2. **Overlapping 3-Band Partition:** Splits the active canvas into 3 horizontal bands with an **8% vertical safety buffer (~180 px)** so boundary equations are never cut in half.
  3. **High-Resolution Scaling:** Rescales each band to `max_dim = 1200 px`. Width is preserved at 1200 px for only ~8 lines of text $\implies$ **$3.5\times$ higher token density per line**!
* **Hardware Result:** ViT peak activation memory drops from **1.30 GiB down to 240 MiB**, leaving **> 1.4 GiB of safe headroom** on our 16 GB V100 GPU (zero OOM)!

### Speaker Script (What to Say)
> *"Here is how our Multi-Band Slicing Architecture works in practice:
>
> When an evaluator grades an answer script, they don't read all 30 lines simultaneously; they read derivation by derivation. We mimicked this cognitive process.
>
> First, we automatically crop out the printed IIT Guwahati booklet instructions at the top 18% of the page. Next, we partition the student answer canvas into three horizontal bands with an 8% vertical safety overlap—roughly 180 pixels. This guarantees that mathematical fractions, square roots, or matrices straddling partition borders are never sliced in half.
>
> Each band is then processed at a native resolution of 1200 pixels. Because a single band contains only 7 to 8 lines of writing instead of 30, the effective token density per line increases by 3.5 times.
>
> Critically, from a systems perspective, the Vision Transformer attention matrix drops from 1.30 GiB to just 240 MiB—yielding zero CUDA OOM errors and running reliably within our 16 GB limit."*

---

# SLIDE 8: Mathematical Proof Case Study: Student 1 (Roll 240101079, Page 01)

### Slide Content (Visual Layout)
* **Exam Problem:** MA102 Question 1(a) — Prove whether matrix set $S \subset M_{4 \times 4}(\mathbb{R})$ (where Col 4 = Col 1 + Col 2 + Col 3) forms a subspace.
* **Slice-by-Slice Transcription Analysis:**
  * **Band 1 (Axiom 1: Zero Vector Inclusion):**
    * Ground Truth: $S$ includes zero matrix; fourth col = $0+0+0=0$.
    * Output: 100% verbatim arithmetic capture. Edit Distance: 1 / 47 tokens $\implies$ **TER: 2.13% (97.87% Token Accuracy)**.
  * **Band 2 (Axiom 2: Additive Closure Proof):**
    * Ground Truth: $W_1, W_2 \in S \implies W_1 + W_2 \in S$; distribution across 4th column.
    * Output: Seamless logical continuation across boundary. Edit Distance: 4 / 58 tokens $\implies$ **TER: 6.90% (93.10% Token Accuracy)**.
  * **Band 3 (Axiom 3: Scalar Multiplication Closure):**
    * Ground Truth: $tW \in S$ via $t(a_1 + a_2 + a_3) = ta_1 + ta_2 + ta_3$.
    * Output: Correct completion of proof. Edit Distance: 15 / 65 tokens $\implies$ **TER: 23.08% (76.92% Token Accuracy)**.
* **Overall Outcome:** **100% Proof Completeness** — all 3 subspace axioms captured and mathematically verifiable!

### Speaker Script (What to Say)
> *"Let us examine the empirical reality on an authentic exam script: Student Roll 240101079, Page 01.
>
> The student is answering MA102 Question 1(a), testing whether a subset of 4x4 matrices forms a vector subspace.
>
> In Band 1, the model transcribes Axiom 1—Zero Vector Inclusion—with 97.87% accuracy, capturing the literal column summation.
>
> In Band 2, the student establishes Additive Closure. Our 8% overlap buffer enables the model to pick up the derivation with 93.10% accuracy, correctly continuing the algebraic expansion across the cut boundary.
>
> In Band 3, the model completes Axiom 3—Scalar Multiplication Closure. Across all three bands, our pipeline captured 100% of the mathematical proof steps, producing compilable LaTeX that exactly mirrors the student's solution."*

---

# SLIDE 9 [NEW]: Visual Verification: Actual Ink Scan vs. Compiled LaTeX PDF

### Slide Content (Visual Layout)
* **Left Card: Scanned Student Ink (300 DPI Raw Optical Crop):**
  * Displays authentic raw handwriting snippet from Student 1 (`page-240101079-01_part1.jpg`).
  * Shows unruled paper, cursive English handwriting (*"S must include zero matrix..."*), and mathematical expressions.
* **Right Card: Compiled Production PDF (Tectonic XeTeX Typeset Output):**
  * Displays rendered PNG crop of the compiled PDF (`rendered_typeset_crop.png`).
  * Formatted with Computer Modern typography, balanced AMS math blocks, clean alignment, and zero syntax errors.
* **Key Visual Takeaway:** High fidelity, clean readability, 100% preserved mathematical truth, completely eliminating evaluator cognitive fatigue!

### Speaker Script (What to Say)
> *"Here is the visual proof of our system on authentic student ink from Student 1.
>
> On the left is the raw 300 DPI scanner crop from the physical unruled answer booklet. Notice the cursive slant, variable word spacing, and mathematical notation.
>
> On the right is the direct, automated output of our pipeline: a publication-grade PDF compiled autonomously by Tectonic XeTeX from the stitched LaTeX code.
>
> Observe the exact visual correspondence: the student's exact mathematical reasoning is preserved verbatim in pristine Computer Modern typography. This eliminates cognitive eye fatigue for evaluators while providing a digital, archived record of the exam."*

---

# SLIDE 10: Quantitative Accuracy Benchmarks

### Slide Content (Visual Layout)
* **Standardized Evaluation Engine:** `eval_metrics.py` utilizing Levenshtein Token Error Rate (TER):
  $$\text{TER} = \frac{S + D + I}{N} \times 100\%$$
* **Comprehensive Benchmark Results Table (Student 1, Roll 240101079):**

| Evaluation Unit | Mathematical Target | Ref Tokens | Hyp Tokens | Edit Dist | TER (%) | Token Accuracy (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Slice Part 1** | Axiom 1: Zero Vector | 47 | 47 | 1 | **2.13%** | **97.87%** |
| **Slice Part 2** | Axiom 2: Additive Closure | 58 | 58 | 4 | **6.90%** | **93.10%** |
| **Slice Part 3** | Axiom 3: Scalar Closure | 65 | 72 | 15 | **23.08%** | **76.92%** |
| **Full Stitched Page** | Complete MA102 Proof | **170** | **177** | **20** | **11.76%** | **88.24%** |

* **Comparative Takeaways:**
  * Multi-Band Slicing (**88.24% Accuracy**) out-performs Monolithic VLM runs (**~60% Accuracy**) by **+28.24%**.
  * Out-performs legacy OCR engines like Tesseract / TrOCR (**< 40% Accuracy**) by **> 48%**!

### Speaker Script (What to Say)
> *"This table presents our quantitative accuracy benchmarks on Student 1's full-page derivation using our standardized evaluation engine, `eval_metrics.py`.
>
> We measure Token Error Rate based on Levenshtein edit distance at the token level.
>
> For Part 1, the model achieved 97.87% token accuracy. For Part 2, 93.10%. In Part 3, dense cursive phrasing yielded 76.92%.
>
> Across the entire stitched full-page proof, our pipeline achieved an overall word-for-word accuracy of **88.24%**, corresponding to a Token Error Rate of only 11.76%.
>
> In contrast, monolithic full-page models achieved only ~60% accuracy due to visual downscaling, while legacy OCR engines achieved below 40%. This represents an absolute accuracy gain of 28 percentage points."*

---

# SLIDE 11: Edge Cases, Model Failure Modes & Scientific Observations

### Slide Content (Visual Layout)
* **Empirical Failure Analysis (Student Roll 240122044, Page 14):**
  * **Failure Mode 1: The 'Over-Helpful Tutor' Prior (Semantic Hallucination):**
    * *Observation:* The student left their proof incomplete at the bottom of the page. The model automatically appended: *"Therefore, S is not a subspace."*
    * *Root Cause:* Pre-trained conversational RLHF / instruction tuning causes the VLM to act like an interactive tutor and solve the problem rather than strictly transcribing.
    * *Engineering Fix:* Enforced strict negative prompting system constraints: *"Transcribe strictly what is written. Never complete derivations. Never evaluate correctness."*
  * **Failure Mode 2: Blank Paper Infinite Generation Loop:**
    * *Observation:* On unwritten paper or margins, the model enters repetitive hallucination loops.
    * *Engineering Fix:* Pre-inference Laplacian variance ink filter ($\sigma^2(\nabla^2 I) < 1\%$), instantly bypassing unwritten canvases.

### Speaker Script (What to Say)
> *"In an academic defense, analyzing failure modes demonstrates true engineering depth. On Student 2 (Roll 240122044), we diagnosed two critical edge cases:
>
> First is what we term the 'Over-Helpful Tutor' prior. When a student left a proof incomplete, the base VLM hallucinated a conclusion: 'Therefore, S is not a subspace.' This occurs because commercial foundation models are instruction-tuned as helpful chatbots that want to solve the math problem.
>
> We resolved this by injecting strict negative prompt constraints, explicitly instructing the model to act as an optical scribe that never completes incomplete proofs.
>
> Second, on blank booklet pages, the model occasionally entered repetitive hallucination loops. We eliminated this at zero compute cost by implementing a classical Laplacian variance ink filter that detects blank pages prior to inference."*

---

# SLIDE 12: Automated LaTeX Stitching & Direct Compilation Pipeline

### Slide Content (Visual Layout)
* **Four-Stage Post-Processing Pipeline (`stitch_latex_parts.py`):**
  1. **Body Extraction & Prefill Locking:**
     * Employs Assistant Prefill Locking (pre-filling `\documentclass`) to force raw LaTeX output.
     * Sanitizes redundant preambles and markdown fences (````latex ... ````).
  2. **Pushdown Automaton (PDA) Stack Balancer:**
     * Formulated as a **Deterministic Pushdown Automaton** recognizing balanced context-free grammar tags: `\begin{env}` and `\end{env}`.
     * Maintains a pushdown LIFO stack of open environments, automatically injecting matching closer tags at slice boundaries.
  3. **Underscore Escaping Engine:**
     * Insidious bug: student shorthand abbreviations (e.g., `fourth_col`) crash LaTeX compilers.
     * Masks valid math with `ZZMATHBLOCKiZZ`, escapes prose underscores (`\_`), and safely restores math blocks.
  4. **Tectonic XeTeX Compilation:**
     * Non-interactive, multi-pass XeTeX engine with automatic font caching $\implies$ **Compiles in < 5 seconds with 100% compilation pass rate!**

### Speaker Script (What to Say)
> *"Once the three image slices are transcribed, we face a software engineering challenge: how do we stitch separate text fragments into a single, syntactically valid LaTeX document?
>
> Our post-processing engine, `stitch_latex_parts.py`, performs four automated steps:
>
> First, we extract the document body and strip duplicate preambles using assistant prefill locking.
>
> Second—an algorithmic contribution of particular interest to formal language theory—we formulated a Deterministic Pushdown Automaton. Because LaTeX environments like `\begin{aligned}` and `\begin{matrix}` form a context-free grammar, our stack balancer maintains a LIFO pushdown stack to detect unclosed tags across slice boundaries and inject matching closures.
>
> Third, our underscore engine resolves a subtle bug: student shorthand like 'fourth_col' in prose causes compiler crashes. We mask all math environments with unique tokens, escape prose underscores, and restore the math.
>
> Finally, we pipe the stitched output to Tectonic, an embedded XeTeX compiler, achieving a 100% compilation pass rate in under 5 seconds with zero manual intervention."*

---

# SLIDE 13: Architectural Comparison Matrix: Monolithic vs. 3-Band Slicing

### Slide Content (Visual Layout)
* **Direct 7-Metric Architectural Comparison:**

| Architectural Metric | Monolithic Full-Page (1 Page $\to$ 1 Call) | Multi-Band High-Res Slicing (3 Bands $\to$ 3 Calls) | Engineering Advantage |
| :--- | :---: | :---: | :--- |
| **ViT Input Patches ($N$)** | > 5,800 patches | ~1,900 patches per slice | **$67\%$ reduction in spatial sequence length** |
| **ViT Self-Attention Memory** | **1.30 GiB** (contiguous allocation) | **~240 MiB** (easily fits V100) | **$81.5\%$ reduction in peak activation VRAM** |
| **GPU Safety Headroom** | < 450 MiB (High crash risk) | **> 1,400 MiB** (Guaranteed stable) | **Zero CUDA Out-Of-Memory crashes** |
| **Effective Pixel Width** | 848 px (downscaled page) | **1200 px** (native slice) | **$1.41\times$ wider optical raster** |
| **Lines per Visual Frame** | 25–30 handwritten lines | **7–8 handwritten lines** | **$3.5\times$ higher token density per line** |
| **Subscripts / Minus Clarity** | Blurry; $3.5 \to 8.5$, $-$ dropped | **Sharp, crisp stroke definition** | **High fidelity on small mathematical notation** |
| **Token Accuracy (Full Page)** | ~60% (severe equation drop) | **88.24%** (Levenshtein TER 11.76%) | **$+28.2\%$ absolute accuracy improvement** |

### Speaker Script (What to Say)
> *"This comparison matrix directly summarizes the engineering advantages of our 3-band slicing architecture over the monolithic baseline.
>
> Look at the resolution arithmetic: by dividing the page into 3 slices, we reduce the patch count per forward pass from 5,800 to 1,900, which slashes the quadratic ViT attention memory from 1.30 GiB down to 240 MiB.
>
> At the same time, because each slice only contains 7 to 8 lines of writing instead of 30, the effective token density per line increases by 3.5 times. This single architectural decision is what allows the model to clearly distinguish minus signs, decimal points, and subscripts, directly driving our 28 percentage point accuracy leap."*

---

# SLIDE 14: Summary of Contributions & Future Work (Phase 2 Roadmap)

### Slide Content (Visual Layout)
* **Summary of Phase 1 Deliverables:**
  * **Curated Institutional Dataset:** 13,146 real exam pages from MA102, establishing an authentic handwritten benchmark.
  * **Hardware Ceiling Solved:** 16 GB Tesla V100 memory limit overcome via Multi-Band Slicing with zero OOM crashes.
  * **High Verbatim Accuracy:** Achieved **88.24% Token Accuracy** on real undergraduate student handwriting.
  * **Autonomous Pipeline:** End-to-end Python engine compiling raw scans to production PDFs via Tectonic in < 5 seconds.
* **Phase 2 Engineering Roadmap:**
  * **QLoRA Parameter-Efficient Fine-Tuning:** Fine-tune model weights on PARAM Kamrupa to teach the model IITG handwriting styles.
  * **INT4 Model Quantization (AWQ/GPTQ):** Compress 7B model from 14 GB to 4.5 GB, cutting inference latency from 15s to 3s per page.
  * **Faculty Grading UI:** Web-based split-screen interface displaying original scanned ink next to typeset PDF for fast evaluator review.

### Speaker Script (What to Say)
> *"To summarize our Phase 1 accomplishments:
>
> We curated an institutional dataset of over 13,000 exam pages. We solved the 16 GB memory ceiling on PARAM Kamrupa, achieved 88.24% token accuracy on unconstrained student handwriting, and built an automated stitching compiler that generates typeset PDFs in seconds.
>
> Looking ahead to Phase 2, our roadmap targets three engineering milestones:
>
> First, conducting QLoRA parameter-efficient fine-tuning on our curated exam dataset to directly eliminate conversational priors.
>
> Second, implementing INT4 quantization using AWQ, which will compress the model from 14 GB down to 4.5 GB and triple inference throughput.
>
> And third, building an assistive web interface that displays original student ink alongside compiled LaTeX for instantaneous faculty verification."*

---

# SLIDE 15: Conclusion, Deliverables & Q&A

### Slide Content (Visual Layout)
* **Core Takeaways:**
  * Demonstrated that 7B Vision-Language Models can be successfully adapted to full-page university examinations on constrained hardware.
  * Multi-Band Slicing + CFG Pushdown Automaton solves both the visual resolution dilemma and LaTeX syntax balancing.
* **Open-Source & Verifiable Deliverables on GitHub/Cluster:**
  * Complete source code: `batch_slice_transcribe.py`, `stitch_latex_parts.py`, `eval_metrics.py`.
  * Fully compiled presentation deck and 53-page BTP-I Report.
  * SLURM orchestration scripts and benchmark logs on PARAM Kamrupa.
* **Acknowledgements:**
  * Sincere gratitude to our supervisor **Dr. K. V. Srikanth**, and evaluation committee members **Dr. H. Ramesh** and **Dr. Vinay Wagh** for their guidance and feedback.

### Speaker Script (What to Say)
> *"In conclusion, this project demonstrates that high-capacity vision-language models can be successfully adapted to full-page university examinations on constrained supercomputing hardware.
>
> By combining multi-band visual slicing with formal language stack balancing, we have built a robust, compilable, and end-to-end on-premise pipeline.
>
> All source code, SLURM scripts, evaluation metrics, and documentation are fully functional and verifiable.
>
> We express our heartfelt gratitude to Dr. Srikanth, Dr. Ramesh, and Dr. Wagh for their valuable support.
>
> Thank you very much. We are now open for questions and discussion."*

---

# STRATEGIC COMMITTEE Q&A DEFENSE PREPARATION

### Questions Anticipated from Dr. H. Ramesh
*(Research Area: Word Representable Graphs, Formal Languages and Automata Theory)*

#### Q1: "How do you guarantee that your LaTeX stitching engine produces syntactically valid code? What grammar class does LaTeX belong to?"
* **Strong Answer:** *"LaTeX mathematical syntax is fundamentally a Context-Free Grammar (CFG), specifically because environments like `\begin{matrix}` and `\end{matrix}` require matching nested brackets and delimiters that cannot be recognized by a Finite State Automaton. In our stitching engine `stitch_latex_parts.py`, we implemented a Deterministic Pushdown Automaton (PDA). We maintain an explicit LIFO pushdown stack that pushes every `\begin` tag and pops upon encountering `\end`. If the input stream terminates with an unempty stack due to a band boundary cut, the automaton deterministically generates and appends the missing closure tokens in reverse topological order, guaranteeing syntax closure."*

#### Q2: "In your Modular Pipeline (Slide 5), you mentioned a 2D Spatial DAG. How do you construct the graph and handle edge conflicts in two-column or marginal handwriting?"
* **Strong Answer:** *"We model each detected bounding box as a vertex $V_i$. An edge $(V_i, V_j)$ is directed from $i$ to $j$ if $V_i$ strictly precedes $V_j$ in standard reading order: either $y_j - y_i \ge \theta_y$ (vertical precedence), or for boxes within the same horizontal line ($|y_i - y_j| < \theta_y$), $x_j - x_i \ge \theta_x$ (left-to-right precedence). To ensure the graph remains acyclic, we enforce a strict partial order. For marginal annotations, we project a vertical bounding separator; elements residing in the margin are topologically sorted after the primary derivation block to prevent interleaved reading order."*

---

### Questions Anticipated from Dr. K. V. Srikanth & Dr. Vinay Wagh
*(Supervisor & Committee: Mathematics & Computing)*

#### Q3: "Why did you choose a 7B parameter model instead of training a specialized CNN-Transformer from scratch like CROHME winners?"
* **Strong Answer:** *"As detailed in Chapter 3 and Slide 6, we actually did train a specialized 23.1M parameter DenseNet model from scratch. However, specialized HMER models like DenseNet or CoMER are trained on CROHME, which consists exclusively of single, isolated mathematical formulas. Real IIT Guwahati exam scripts are fundamentally document-level: they combine cursive English prose sentences, scratch-outs, margin notes, and multi-line derivations. A 7B parameter multimodal model possesses a rich linguistic prior that can transcribe English explanation alongside complex math in a unified token space, which smaller isolated formula models cannot do."*

#### Q4: "What happens if a student writes an expression exactly across the horizontal cut boundary? Won't you get duplicate tokens?"
* **Strong Answer:** *"Our 8% vertical safety overlap translates to approximately 180 to 200 pixels—larger than the maximum height of a handwritten derivation line (~80–120 px). This guarantees that neither slice has a truncated symbol. In `stitch_latex_parts.py`, we implement boundary deduplication: we compute the character-level overlap between the suffix of Slice $k$ and the prefix of Slice $k+1$. If the edit distance is below an alignment threshold, the duplicate prefix in Slice $k+1$ is discarded before concatenation."*

#### Q5: "Why not use cloud APIs like GPT-4o or Gemini 1.5 Pro? They have unlimited compute."
* **Strong Answer:** *"First and foremost, institutional privacy and academic integrity: university exam papers cannot be uploaded to external commercial cloud servers. Second, bandwidth and latency: transmitting tens of thousands of high-resolution images over external networks creates severe bottlenecks. By engineering our pipeline to run within a single 16 GB Tesla V100 on PARAM Kamrupa, we deliver an entirely private, air-gapped, on-premise institutional solution."*
