# 03 — Feasibility, Risks & Evaluation

> The honest answer to *"can this be built, how well, and at what cost?"* — plus how we'd measure it and where it will hurt.

---

## 1. Feasibility verdict (one paragraph)

**A ~50–60% POC is very feasible now (D14); a high-accuracy production system is a longer iterative climb.** Our scope is strictly **handwriting → LaTeX conversion** — the professor reads the LaTeX and grades (D13), so we're not attempting the hardest part (auto-grading). For the near-term **proof-of-concept**, ~50–60% faithful conversion is a realistic, achievable bar given today's tools (single-equation recognition sits at ~62–80% exact-match on *clean* isolated expressions; real messy IITG pages will be lower, but ~50–60% overall is reasonable to demonstrate value). The **iterative climb afterward** is where the known hard limits bite: full handwritten-page→LaTeX has **no off-the-shelf solution**, errors compound across pipeline stages, and general VLMs **silently "correct" student errors 42–66% of the time** — which for a *faithful transcription* tool is the key thing to guard against. So the plan is: **prove a ~50–60% POC, then iterate toward higher fidelity + full end-to-end**, keeping the human (the prof) reading the output throughout.

---

## 2. What accuracy to realistically expect

| Content type | Realistic expectation (2026) | Basis |
|---|---|---|
| **Printed** math → LaTeX | ~90%+ (BLEU-4 ~93 on im2latex) | `01` — printed is solved |
| **Isolated, clean handwritten** equation | ~62–80% exact-match (ExpRate) | PosFormer/Uni-MuMER on CROHME/HME100K |
| **Isolated, messy/realistic** handwritten eq. | **~23–72%** (23–28% zero-shot VLM; ~72% fine-tuned specialist on HME100K) | `01` §3.2 |
| **Handwritten English prose** | High (mature HTR) | TrOCR-class |
| **Multi-line / nested / matrix** handwritten math | **Markedly worse** than isolated | HMER structured-expression results |
| **Full handwritten page → coherent LaTeX** | **No reliable number — open problem; page-level fidelity ≪ per-equation** | errors compound [B]→[D]→[E] |
| **Diagrams → TikZ** | **Not reliable** — draft quality at best | see §4.4 |
| **Autonomous grading** of handwritten math/proofs | **Not deployment-ready** | MAE ~7.66%, 46.67% human agreement |

> **Rule of thumb for planning:** assume **20–40% of equations** and a **larger fraction of full pages** will need human touch-up. Design the review UX around that, not around a hoped-for 95%.

---

## 3. Top risks (ranked)

| # | Risk | Severity | Why | Mitigation |
|---|---|---|---|---|
| 1 | **Over-correction** — model silently fixes student's wrong math | 🔴 Critical | 42–66% of VLM transcriptions; *destroys the grading signal*; worse with better models | Faithful recognizer backbone + **dual-recognizer disagreement flagging**; never LLM-"clean" math; human verifies fidelity (`02` §3.5) |
| 2 | **Error compounding across stages** | 🔴 High | page fidelity = product of stage accuracies | Confidence gating + HITL; keep display blocks as whole units; compile-check |
| 3 | **No IIT-level training/eval data** | 🔴 High | best analog (FERMAT) is grades 7–12, 2,244 samples | **Build a bespoke corpus** (`04` Phase 0); bootstrap from MathWriting/HME100K/IAM |
| 4 | **Layout/reading-order on handwriting** | 🟠 Med-High | detectors printed-trained; weak on HW math pages | VLM cross-check for layout; human reorder in UI; may need to fine-tune detector |
| 5 | **Diagrams** | 🟠 Med | sketch→TikZ unsolved | **Embed as images**; explicit non-goal for TikZ |
| 6 | **Student-data privacy** | 🟠 Med → **Low-Med** | exam sheets = sensitive PII; some APIs train on data / are offshore | **Self-hosting now feasible (D10: full GPU/intranet)** → can keep data on-campus; cloud only on anonymized sample under DPA. Both lanes carried (D5) |
| 7 | **Messy scans** (skew, pencil, crossed-out) | 🟡 Med → **Low** | degrades downstream stages | **Flatbed input (D11)** removes dewarp/shadow problems; just deskew + denoise. Still preserve struck-through work |
| 8 | **Cost creep at class/institute scale** | 🟡 Low-Med | per-page API + human-review time | Estimate early (§5); self-host to cap marginal cost |
| 9 | **Grader trust / adoption** | 🟡 Med | one hallucinated "correct" proof erodes trust fast | Side-by-side scan+render; never hide uncertainty; start opt-in |

---

## 4. The genuinely hard cases (know them up front)

### 4.1 Proofs & reasoning in prose
Free-form proof *reasoning* is the hardest to transcribe faithfully (prose+notation+structure) and the least useful to "auto-grade." UIUC finding: **87% of AI grading errors are transcription failures, not rubric misapplication** — i.e. the bottleneck is *reading*, not *judging*. Keep humans on proofs.

### 4.2 Multi-line aligned math, matrices, cases
Weak even for printed; assembly must infer `align`/`pmatrix`/`cases`. Prefer whole-block recognizers + compile validation.

### 4.3 Notation ambiguity
Handwritten $z$ vs $2$, $x$ vs $\times$, $\ell$ vs $1$, sub/superscript vs baseline. Concentrate confidence-flagging here.

### 4.4 Diagrams (the least-tractable link) — verified
- **Production standard: detect region → embed as raster image.** Nougat drops figures; Mathpix crops+embeds (OCR's only the labels via `include_diagram_text`). ([PubLayNet](https://arxiv.org/abs/1908.07836), [Mathpix docs](https://docs.mathpix.com/))
- **Sketch→TikZ is emerging but draft-quality:** **DeTikZify** (NeurIPS'24, [paper](https://arxiv.org/abs/2405.15306)) degrades from reference→sketch (DreamSim 80.5→74.6) even on *synthetic* sketches — real handwriting is worse. **IMGTikZ/SkeTikZ** (ICLR'25, [site](https://sketikz.github.io/)) — 3,231 real sketch↔TikZ pairs; beats GPT-4o (compile-success 0.80 vs 0.48) but authors say the task "remains largely unresolved."
- **No mature recognizer** for handwritten function graphs, number lines, axes, or commutative diagrams.
- **Recommendation:** embed as images + human-in-loop; optionally offer DeTikZify/IMGTikZ as an experimental "draft TikZ" button with a compile check. Do not depend on it.

---

## 5. Effort & cost (feasibility-stage estimates)

> Rough order-of-magnitude, for planning only. Refine after Phase-1 bake-off (`04`).

### 5.1 Engineering effort
| Approach (`02`) | To a working prototype | To a robust service |
|---|---|---|
| A — VLM-first | ~days–2 weeks | not recommended as sole system |
| B — Specialized pipeline | ~2–4 months | 6–12 months + data work |
| **C — Hybrid (rec.)** | **~3–6 weeks** | ~3–6 months + ongoing eval |

Plus **data collection/annotation** (Phase 0) running in parallel: weeks of effort to assemble even a few hundred annotated IITG answer pages.

> **Production-scale note (DECISIONS.md D4):** the intent is a real service for profs, so cost must be modeled at **class/department volume**, not per-demo. Use DECISIONS.md Q3 (pages/student × students × classes) to project. Detailed component/throughput design: [`06-architecture-spec.md`](06-architecture-spec.md).

### 5.2 Running cost (illustrative, per 1,000 exam pages)
| Component | Cost basis | ~Cost / 1,000 pages |
|---|---|---|
| Mathpix (PDF-rate) | ~$0.005/page (`01` §4.1) | **~$5** |
| Frontier VLM (per page, if used) | model-dependent | ~$5–30 (varies) |
| Self-hosted open-weight (Qwen2.5-VL) | GPU amortized | marginal ≈ compute only |
| **Human review** | grader/TA time on flagged 20–40% | **dominant cost** — model this explicitly |

> **Human-review time, not API fees, is the real cost.** A HITL grading deployment showed ~23% grading-time reduction *with* mandatory review — so the value case is "faster + more legible grading," not "zero human effort."

### 5.3 Data-privacy cost
Self-hosting (privacy-optimal) trades API fees for GPU + MLOps effort. If cloud is allowed, budget legal review + DPA. This is a **decision gate**, not a line item (`04`).

---

## 6. Evaluation: how we'll measure success

### 6.1 Metrics (and why raw ones mislead for math)
| Metric | Measures | Limitation |
|---|---|---|
| **ExpRate** (exact expression match) | % expressions 100% correct | All-or-nothing; punishes LaTeX non-uniqueness (`\frac{1}{2}` vs `1\over2`) |
| **CER** (token-level) | edit distance / length, LaTeX-token as unit | token error ≠ semantic/render error |
| **BLEU** | n-gram overlap | high BLEU can still be wrong/non-compiling |
| **Edit distance** | edits to fix | distance ≠ severity |
| **Compile rate** | does the LaTeX compile? | necessary, not sufficient |
| **Render-match** (compile → compare images) | semantic-ish equality, handles non-uniqueness | brittle to font/spacing; needs successful compile |

**Method:** always **normalize LaTeX before scoring** (MathWriting-style: canonical sub/superscript order, `\sin`→`sin`, unify matrix envs, strip font/size) and prefer **render-match over string metrics**.

### 6.2 The metric that actually matters for us — **Fidelity**
Standard metrics measure "closeness to a reference." We need **"did it transcribe what the student *actually wrote*, including errors?"** So define:
- **Transcription Fidelity Rate** — % of regions where the LaTeX render matches the student's writing *including any mistakes* (human-judged on a sample).
- **Over-correction Rate** — % of regions where the model produced *different, "more correct"* math than written. **Target: ≈0.** This is the primary safety metric.
- **Flag Precision/Recall** — of regions the system flagged low-confidence, how many truly needed fixing (precision) and did it catch the ones that did (recall). Drives HITL efficiency.
- **Reviewer effort** — median seconds/edits per page to reach grader-acceptable output. The real productivity metric.

### 6.3 Grading-task metrics (only if we later attempt grading)
Error-detection/localization/correction rates (à la FERMAT), and human-agreement (exact-match %, MAE, Pearson r) — with the baseline reminder that GPT-4 hit only **46.67% agreement / MAE 7.66%** on real exams. Grading stays human-decided.

---

## 7. Feasibility conclusion → roadmap
- **Build the *assistive* transcription tool (Approach C), not autonomous grading.**
- **The de-risking experiment is the recognizer bake-off on real IITG scripts, scored by Fidelity + Over-correction Rate** — do this before anything else (`04` Phase 1).
- **Assume a bespoke dataset and a human-in-the-loop** are part of the product.
- **Diagrams = images. TikZ = non-goal.**
- **Pick the privacy lane early** (self-hosted vs. cloud) — it constrains model choice.
