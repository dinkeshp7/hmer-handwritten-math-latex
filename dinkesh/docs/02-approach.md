# 02 — Technical Approach: Candidate Architectures

> How we could actually build "handwritten answer sheet → LaTeX", the trade-offs, and the pipeline stages. Grounded in the landscape findings (`01`). No code — this is the design space.

---

## 0. The core insight that shapes everything

There is **no single model** that takes a full handwritten page (prose + multi-line equations + proofs + diagrams) and emits coherent LaTeX. That is an **open research problem** (`01` §7). So every realistic approach is a **staged pipeline**, and the central design tension is:

> **Faithfulness vs. fluency.** LLM/VLM approaches produce beautiful, fluent LaTeX — but *silently correct the student's mistakes 42–66% of the time* (`01` §3.1). For grading, the mistake is the signal. **Faithful transcription is the product; fluent-but-wrong is a failure.**

Everything below is organized around preserving faithfulness while getting usable LaTeX.

---

## 1. The pipeline stages (common to all approaches)

Whatever the model choices, a full sheet must pass through these stages. The approaches in §2 differ in *which stages are one model vs. many*.

```
   Scanned/photographed answer sheet (image / PDF page)
        │
   [A] Pre-processing        deskew · denoise · dewarp · shadow removal · binarize
        │
   [B] Layout analysis       segment page into regions:
        │                     prose · display-equation · inline-math · diagram · table · crossed-out
        │
   [C] Reading order         order regions the way a human reads them
        │
   [D] Per-region recognition
        │   ├─ prose region      → handwriting text recognition (HTR)  → text (+ inline math)
        │   ├─ equation region   → HMER / math-OCR                     → LaTeX math
        │   ├─ table region      → table structure recognition          → LaTeX tabular
        │   └─ diagram region    → (do NOT vectorize) crop              → embedded image
        │
   [E] Assembly              stitch regions into one LaTeX document
        │                     (align environments, matrices, \section, figure floats)
        │
   [F] Confidence + validation  compile check · per-region confidence · flag low-confidence
        │
   [G] Human-in-the-loop review  side-by-side scan vs. rendered LaTeX; correct; feed back
        │
   Faithful, compilable LaTeX document
```

**Maturity of each stage for HANDWRITING** (from `01` §, document-pipeline research):

| Stage | Printed | Handwritten | Notes |
|---|---|---|---|
| [A] Pre-process | Solved | Mostly solved | thin pencil strokes & bleed-through are the risk |
| [B] Layout | Solved | **Partly / weak** | detectors are printed-trained; math-region detection on HW barely studied |
| [C] Reading order | Solved | **No HW-specific model** | LayoutReader/Surya are printed-trained |
| [D] Prose (HTR) | Solved | Mostly solved | TrOCR-class |
| [D] Equation (HMER) | Solved | **~62–80% exact (isolated)** | the frontier; worse on multi-line/nested |
| [D] Table | Solved | Partly | |
| [E] Assembly | Partly (align/matrix weak) | **Open** | errors compound across [B]→[D]→[E] |
| [F] Confidence | — | Mathpix has it; VLM logprobs approximate | key to making the system *usable* |
| [G] HITL | — | — | **mandatory** given the above |

> **Error compounding is the crux:** if [B] is ~85% right, [D] equations ~75%, [E] assembly ~90%, the *page-level* "fully correct" rate is the product — far below any single stage. This is why page-level fidelity ≪ the headline single-equation numbers.

---

## 2. The three candidate architectures

### Approach A — **VLM-first (single frontier model does [B]–[E])**
Feed the whole page to a frontier VLM (GPT-4o / Claude / Gemini / fine-tuned Qwen2.5-VL) and prompt for LaTeX.

- ✅ Fastest to prototype (days); handles prose+math+layout jointly; best fluency.
- ✅ Gemini 2.x is the strongest *faithful* transcriber in fidelity-aware tests.
- ❌ **Over-correction: 42–66% of transcriptions silently fix student errors** — *disqualifying for grading as-is* (`01` §3.1).
- ❌ Only **~23–28% of expressions fully correct** on realistic handwriting (zero-shot, HME100K).
- ❌ Weak, unreliable confidence signal; hallucinates confident wrong LaTeX; 2-D structure breaks.
- ❌ Cloud privacy concerns for student PII (unless self-hosted open-weight).
- **Verdict:** great for a **quick feasibility probe** and as *one voter* in an ensemble; **unsafe as the sole transcriber** for grading.

### Approach B — **Specialized pipeline (distinct model per stage)**
Classic pipeline: preprocess → layout detector → route prose to HTR (TrOCR-class) and equation crops to HMER (PosFormer/ICAL/Uni-MuMER-class) → assemble.

- ✅ **Faithful by construction** — HMER models transcribe strokes, they don't "reason" the math into correctness.
- ✅ Each stage independently improvable, testable, and swappable (good engineering boundaries).
- ✅ Can be **fully self-hosted** (open weights) → privacy-optimal for exam data.
- ❌ Most engineering effort; **the weak links are [B] layout + [E] assembly on handwriting** (both under-served).
- ❌ HMER ceiling ~62–80% exact on *isolated* equations; multi-line proofs worse.
- ❌ Errors compound across stages.
- **Verdict:** the most **defensible long-term architecture**; heaviest build; needs custom data for [B]/[E].

### Approach C — **Hybrid: specialized transcription + VLM assist, confidence-gated (RECOMMENDED direction)**
Use a **faithful recognizer** as the backbone, a **VLM as a structural/layout assistant and second opinion**, and **confidence gates** everywhere routing hard cases to humans.

Concretely:
- **[B] Layout + [C] reading order:** Surya/PP-StructureV3-style detector, optionally cross-checked by a VLM ("list the regions and their reading order") — VLMs are decent at *layout description* even when unreliable at *math transcription*.
- **[D] Recognition:**
  - Prose → HTR (or VLM, low over-correction risk on prose).
  - Equations → **Mathpix** (commercial, confidence scores) *or* **fine-tuned Qwen2.5-VL / PosFormer** (self-hosted).
  - **Faithfulness guard:** run *two* recognizers on each equation; **disagreement ⇒ flag for human** (catches over-correction — a VLM "fixing" an error will diverge from the faithful recognizer).
- **[E] Assembly:** deterministic template + LLM only for environment formatting (never for content).
- **[F/G]:** compile-check + confidence + disagreement → human review queue.

- ✅ Backbone stays faithful; VLM adds fluency/layout without being trusted for content.
- ✅ **Disagreement detection directly attacks the over-correction failure mode.**
- ✅ Confidence gating makes the human's time go to the hard 20–30%.
- ✅ Tunable privacy (swap Mathpix ↔ self-hosted).
- ❌ More moving parts than A; needs orchestration + a good review UI.
- **Verdict:** best balance of faithfulness, usability, and effort. **This is the recommended direction for a prototype.**

### Side-by-side
| Dimension | A: VLM-first | B: Specialized pipeline | C: Hybrid (rec.) |
|---|---|---|---|
| Faithfulness (no over-correction) | ❌ Poor | ✅ Strong | ✅ Strong (disagreement guard) |
| Prototype speed | ✅ Days | ❌ Weeks–months | 🟡 Weeks |
| Handwritten fluency of output | ✅ Best | 🟡 Assembly-dependent | ✅ Good |
| Confidence / HITL routing | ❌ Weak | 🟡 Per-stage | ✅ Strong |
| Privacy (self-hostable) | 🟡 only if open-weight | ✅ Yes | ✅ Tunable |
| Build/maintenance cost | ✅ Low | ❌ High | 🟡 Medium |
| Weakest link | over-correction | layout+assembly on HW | orchestration complexity |

---

## 3. Handling the hard sub-problems

### 3.1 Prose vs. math disambiguation
"Let $x$ be…" mixes both. Options: (a) layout model tags inline-math spans; (b) VLM segments prose/math; (c) HTR-with-math-tokens. **Recommendation:** treat inline math as a tagged span within prose regions; keep display equations as separate regions.

### 3.2 Multi-line / aligned equations, matrices, cases
The weak point even for printed (`01`). Assembly must infer `align`/`aligned`, `pmatrix`, `cases`. **Recommendation:** keep each *display block* as a unit; prefer a recognizer that outputs the whole multi-line block (Mathpix / Uni-MuMER) over stitching single lines; validate by **compiling**.

### 3.3 Diagrams & figures — **embed as images, do NOT generate TikZ**
Sketch→TikZ (DeTikZify/AutomaTikZ) is immature and needs heavy manual fixing (`01` §, document research). **Recommendation:** detect the diagram region, crop it, embed as `\includegraphics`, and flag "diagram — not transcribed" for the grader. TikZ generation is an explicit **non-goal**.

### 3.4 Crossed-out work, margin notes, arrows, non-linear order
Real exam artifacts. **Recommendation:** detect-and-preserve as annotated image regions rather than forcing into linear LaTeX; surface to the reviewer. Do not silently drop struck-through work (it may carry partial credit).

### 3.5 Faithfulness guardrails (the most important engineering)
- **Dual-recognizer disagreement flagging** (§2 Approach C).
- **Never** post-process math through an LLM "cleanup" that could change meaning.
- **Compile + render** the LaTeX and show it **side-by-side with the original scan** to the human — the human verifies *fidelity to what was written*, not correctness of the math.
- Log per-region confidence; set thresholds empirically (Phase-1 eval, `04`).

---

## 4. Output format & data model
- **Output:** a compilable `.tex` per answer sheet + a **structured JSON** side-car (regions, types, bounding boxes, per-region confidence, recognizer(s) used, flags). JSON is what powers the review UI, analytics, and re-training.
- **Provenance:** every region links back to its pixel box in the source image, so a grader can click LaTeX ↔ scan.
- **Faithful markup:** encode uncertainty explicitly (e.g. a `\uncertain{…}` macro or JSON flag) rather than guessing.

---

## 5. Recommended direction (feeds `03`/`04`)
1. **Start with Approach C (hybrid), scoped down** for the feasibility prototype.
2. **Recognizer bake-off first:** Mathpix vs. fine-tuned Qwen2.5-VL vs. frontier VLM, on **real IITG scripts**, scored for *fidelity* (not BLEU) — this single experiment de-risks the whole project.
3. **Diagrams = embedded images** from day one.
4. **Confidence-gated human review** is part of the product, not an afterthought.
5. **Privacy decision gate:** if student PII can't leave campus, bias toward self-hosted open-weight (Qwen2.5-VL/PosFormer) over Mathpix/cloud VLMs.
