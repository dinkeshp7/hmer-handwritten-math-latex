# 06 — Architecture Specification

> Concrete, component-level design for the production-intent system (recall D4: **real scale, for college profs**). Expands the hybrid pipeline of `02`. Still design-stage — implementation follows Phase-1 GO. Where a choice depends on the privacy lane (DECISIONS.md Q1), **both lanes are specified**.

> **Plain-language note:** this is the "detailed blueprint" of the assembly line from `08-primer.md` §6. Each component below is one station on that line.

---

## 1. System context (production intent)

```
                         ┌─────────────────────────────────────┐
   Prof/TA uploads       │         Handwriting→LaTeX Service      │
   scanned sheets  ──────▶  Ingest → Pipeline → Review UI → Store  │──────▶ Clean LaTeX + PDF
   (batch, per class)    │                                        │        + structured JSON
                         └─────────────────────────────────────┘
                                    │            │
                              Recognizer(s)   Object storage
                            (cloud or self-hosted)  (sheets, outputs, corrections)
```

**Non-functional requirements (because production):**
- **Throughput:** a class batch (see DECISIONS.md Q3 for volume) processed in reasonable time — async/queue-based.
- **Reliability:** a failed page never blocks the batch; retries; nothing silently dropped.
- **Auditability:** every output traces to its source pixels + which recognizer produced it (integrity — see `02` over-correction risk).
- **Multi-user:** profs/TAs, per-course isolation, auth.
- **Privacy:** governed by Q1 lane (§7).

---

## 2. Components (the stations)

| ID | Component | Job | Build/Buy | Notes |
|---|---|---|---|---|
| **INGEST** | Upload & normalize | Accept PDF/images, split into pages, assign IDs, anonymize-check | Build | Batch-oriented; per-course/exam grouping |
| **PREP** | Pre-processing | Deskew, denoise, binarize | Buy/OSS (OpenCV, ScanTailor) | **Flatbed input (D11) → light work**; dewarp/shadow-removal not needed |
| **LAYOUT** | Layout + reading order | Segment page into typed regions; order them | OSS (Surya / PP-StructureV3) + VLM cross-check | Weak link on handwriting — may need fine-tuning |
| **ROUTER** | Region router | Send each region to the right recognizer | Build | prose→HTR, equation→HMER/Mathpix, diagram→crop, table→table-recognizer |
| **REC-PROSE** | Prose recognizer | Handwritten English → text (+ inline-math tags) | Buy/OSS (TrOCR / VLM) | Low over-correction risk on prose |
| **REC-MATH** | Math recognizer (backbone) | Equation image → faithful LaTeX + confidence | **Phase-1 winner** (Mathpix or Qwen2.5-VL) | The critical component |
| **REC-MATH-2** | Second math opinion | Independent equation read for disagreement check | 2nd model | Powers the over-correction guard |
| **REC-TABLE** | Table recognizer | Grid → LaTeX tabular | OSS | Lower priority |
| **DIAGRAM** | Diagram handler | Crop region, embed as image | Build | **No TikZ** (D7); optional draft-only later |
| **ASSEMBLE** | Document assembler | Stitch regions → one .tex (align/matrix/sections/figures) | Build (deterministic templates) | LLM only for *formatting*, never content |
| **VALIDATE** | Validation | Compile-check, per-region confidence, dual-recognizer disagreement flags | Build | Produces the review queue |
| **REVIEW-UI** | Human review | Side-by-side scan vs render; edit; reorder; approve | Build | The actual product surface for profs |
| **STORE** | Persistence | Sheets, outputs, JSON, corrections, audit log | Build (DB + object store) | Corrections feed retraining |
| **FEEDBACK** | Retraining loop | Turn corrections into fine-tuning data | Build (later) | Improves REC-MATH over time |

---

## 3. The core data model — Region JSON (single source of truth)

Every page becomes a structured document. This JSON is the **SSOT** (per your engineering guidelines): the .tex is *derived* from it, the review UI *edits* it, analytics *read* it, retraining *consumes* it.

```jsonc
{
  "sheet_id": "CS101-midsem-2026-stuXYZ",   // anonymized
  "course": "MA101",
  "page_index": 3,
  "source_image": "s3://.../page3.png",
  "preprocessing": { "deskew_deg": 1.2, "dewarped": true },
  "regions": [
    {
      "region_id": "p3-r1",
      "type": "prose",                        // prose | display_math | inline_math | table | diagram | crossed_out
      "bbox": [x, y, w, h],                    // pixel box in source_image (click-to-locate)
      "reading_order": 1,
      "recognizer": "trocr-handwritten",
      "output_latex": "Let $f$ be continuous on $[a,b]$.",
      "confidence": 0.91,
      "flags": [],                             // e.g. ["low_confidence", "disagreement", "uncompilable"]
      "alternatives": []                        // other recognizers' outputs (for disagreement)
    },
    {
      "region_id": "p3-r2",
      "type": "display_math",
      "bbox": [x, y, w, h],
      "reading_order": 2,
      "recognizer": "mathpix",
      "output_latex": "\\int_0^1 x^2\\,dx = \\frac{1}{3}",
      "confidence": 0.67,
      "flags": ["disagreement"],
      "alternatives": [
        { "recognizer": "qwen2.5vl", "output_latex": "\\int_0^1 x^2\\,dx = \\frac{1}{2}" }
      ]
      // ↑ Mathpix and Qwen disagree on the answer → flagged for human. This is the over-correction guard in action.
    },
    {
      "region_id": "p3-r3",
      "type": "diagram",
      "bbox": [x, y, w, h],
      "reading_order": 3,
      "recognizer": "none",
      "output_latex": "\\includegraphics{p3-r3.png}",
      "confidence": null,
      "flags": ["diagram_not_transcribed"]
    }
  ],
  "assembled_tex": "s3://.../page3.tex",
  "compile_status": "ok",
  "review_status": "pending"                    // pending | in_review | approved
}
```

**Why this schema matters:**
- `bbox` links every piece of LaTeX back to the exact pixels → the reviewer clicks LaTeX and sees the source (trust + speed).
- `alternatives` + `flags:["disagreement"]` **operationalize the over-correction safety guard** — the dangerous case becomes a visible, routable flag.
- `confidence` drives the review queue ordering.
- `crossed_out` type ensures struck-through work is **preserved, not dropped** (may carry partial credit).

---

## 4. The over-correction safety guard (detailed)

This is the architecture's most important safety feature (see `02` §3.5, `08` §7).

```
equation region
   ├─▶ REC-MATH   (faithful backbone, e.g. Qwen/PosFormer) → latex_A + conf_A
   └─▶ REC-MATH-2 (second opinion, e.g. Mathpix/VLM)        → latex_B + conf_B
        │
   normalize(latex_A) == normalize(latex_B) ?
        ├── yes → high trust, low review priority
        └── no  → FLAG "disagreement" → human reviews, sees both, picks/fixes
```

Rationale: a model that *over-corrects* a student's error produces different LaTeX than a faithful reader → disagreement → caught. Cheap insurance against the integrity risk that matters most at production scale.

**Cost control:** run the second recognizer only on regions where the first is low-confidence or where the region "looks" like an answer/result line — tune in Phase 2/3 to balance cost vs. safety.

---

## 5. Review UI (the product surface for profs)

Minimum viable design:

```
┌───────────────────────────┬───────────────────────────┐
│   ORIGINAL SCAN (page)     │   RENDERED LATEX (live)    │
│                            │                            │
│  [region boxes overlaid,   │  [compiled math, regions   │
│   color = confidence]      │   clickable]               │
│                            │                            │
│  click a box ──────────────┼──▶ jumps to its LaTeX      │
├───────────────────────────┴───────────────────────────┤
│  REVIEW QUEUE: ⚠ p3-r2 disagreement · ⚠ p5-r1 low-conf │
│  [edit LaTeX] [accept] [reorder] [mark diagram]         │
└─────────────────────────────────────────────────────────┘
```

Principles: **fidelity-first** (reviewer checks "does this match what was written?", not "is the math right?"), uncertainty is **visible** (never hidden), edits are **fast** (keyboard-driven), and corrections are **captured** for retraining (FEEDBACK).

---

## 6. Assembly rules (region JSON → .tex)
- Deterministic template: preamble → regions in `reading_order` → prose as text, `display_math` in `\[...\]` or `align`, tables as `tabular`, diagrams as `\includegraphics`.
- Multi-line/aligned math: prefer whole-block recognizer output; wrap in `align`/`aligned`; **compile-check**, and on failure, fall back to verbatim + flag.
- **LLM is allowed only to format** (e.g. choose `align` vs `cases`), **never to alter mathematical content.** Any LLM-touched region keeps the original as `alternatives`.

---

## 7. Deployment — both privacy lanes (carry BOTH — D5/D12)

Per D5, we don't pre-commit a lane; we evaluate both. **D10 (full GPU/CPU + intranet) makes the self-hosted lane genuinely feasible** — so we can keep student data fully on-campus *and* benchmark cloud on an anonymized sample.

| Concern | Lane A: **Self-hosted** (data stays on-campus) ✅ *feasible now* | Lane B: **Cloud + agreement** |
|---|---|---|
| REC-MATH | Qwen2.5-VL / PosFormer on **our GPUs** | Mathpix API (on-prem option) / frontier VLM (paid, no-training) |
| Prose | TrOCR self-hosted | TrOCR or VLM |
| Data location | **Never leaves campus network** (can be air-gapped via intranet) | Leaves to vendor under signed DPA |
| Infra | GPU server(s), container orchestration, MLOps — **we have the hardware (D10)** | Mostly API calls; lighter infra |
| Cost shape | Capex already owned + ops effort (2-person team, D10) | Opex (per-page fees) |
| Legal | Minimal | **DPA + no-training + FERPA-equivalent review** |
| Best when | Privacy strict / high volume / GPUs available ← **our situation** | Fast benchmark / when self-host quality lags |

> **Design principle (locked):** put every recognizer behind a **swappable interface** so REC-MATH can be Mathpix *or* Qwen *or* PosFormer without touching the rest of the pipeline. This directly serves the "lay out all alternatives / decide per-step on merit" rule (D12) — Phase 1 tests them head-to-head and the winner drops in.

> **Leaning (not a commitment):** given D10, **self-hosted as the production default** (privacy-optimal, hardware already in hand) with **cloud APIs as the Phase-1 benchmark and a fallback** if self-hosted quality lags. Confirmed only after Phase-1 numbers (`05`).

---

## 8. Tech-choice alternatives per layer (D12 — decide each on merit; Phase 1 tests recognizers)

Every layer lists **multiple researched options** rather than one prescribed pick. Trade-offs noted; final choice per-layer after evaluation.

| Layer | Option 1 | Option 2 | Option 3 | Notes / how to choose |
|---|---|---|---|---|
| **Preprocessing** | OpenCV (deskew, adaptive threshold, denoise) | ScanTailor Advanced (batch) | vendor-integrated (PP-StructureV3 built-in) | **Flatbed (D11) → light**: deskew + denoise usually enough; dewarp/shadow tools *not* needed |
| **Layout + reading order** | Surya | PaddleOCR PP-StructureV3 | fine-tuned detector on IITG data | Off-the-shelf first; fine-tune only if handwritten-region detection is weak (likely) |
| **Prose (HTR)** | TrOCR-handwritten (self-host) | Frontier VLM (cloud) | Qwen2.5-VL (self-host) | Low over-correction risk on prose; pick by fidelity + lane |
| **Math (REC-MATH)** | Mathpix (cloud, confidence) | Qwen2.5-VL fine-tuned (self-host) | PosFormer/ICAL (self-host, eq-only) | **Phase-1 bake-off decides**; loser(s) become REC-MATH-2 second opinion |
| **Second opinion (REC-MATH-2)** | whichever of the above is *not* primary | — | — | Powers over-correction disagreement guard (§4) |
| **Table recognition** | PaddleOCR table | Surya table | VLM | Lower priority |
| **Assembly** | deterministic templates + LaTeX compile | + LLM for *formatting only* | — | LLM never alters content (§6) |
| **Compile/render** | full LaTeX (TeX Live) | KaTeX (fast, web) | MathJax | KaTeX for live UI preview; TeX Live for final PDF |
| **Store** | Postgres (JSONB) + object store (MinIO on-prem) | SQLite (prototype) | — | MinIO keeps blobs on-campus (self-host lane) |
| **Queue** | Redis/RQ or Celery | Postgres-based queue | — | Async batch for whole-class volumes |
| **Review UI** | web app + KaTeX render, side-by-side | existing tool (Label Studio adapted) | — | Build — it's the prof-facing surface |

> Self-host-friendly picks (MinIO, TeX Live, Postgres, TrOCR/Qwen) are highlighted because D10 gives us the hardware to run everything on-campus if the privacy lane lands there.

---

## 9. What this spec deliberately defers
- Exact framework/language choices (until builder & lane known — Q2).
- Retraining loop details (FEEDBACK) — Phase 4.
- Autonomous grading — out of scope (D3).
- TikZ diagram generation — non-goal (D7).

**Open dependencies:** DECISIONS.md **Q1** (privacy → REC-MATH + deployment), **Q2** (builder/GPU → self-host feasibility), **Q3** (input format/volume → PREP difficulty + throughput sizing).
