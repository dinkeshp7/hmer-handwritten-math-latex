# 04 — Phased Roadmap & Decision Gates

> A staged plan. Each phase ends in a **decision gate** — a go/no-go with explicit criteria — so we never over-invest before the previous step de-risks the next.

> **⭐ Strategy (D14): POC first, then iterate.** The near-term goal is a **~50–60%-accurate proof-of-concept** that converts handwriting → LaTeX — *not* a production-grade system on day one. Prove the concept works, then improve toward full end-to-end. This lowers the early bar (realistic for handwritten math today) and front-loads learning. Scope is strictly **conversion**; the professor reads the LaTeX and grades (D13).

---

## Guiding principles
1. **POC before polish (D14).** First target ~50–60% faithful conversion; iterate afterward. Don't build production infra before the POC shows promise.
2. **De-risk the biggest unknown first.** The binding constraint is *transcription fidelity on real IITG handwriting*. So the first real experiment is a recognizer bake-off, not infrastructure.
3. **Cheapest experiment that resolves a decision, first.** Prefer a 20-page manual eval over a 6-month build.
4. **Faithfulness is the acceptance criterion**, not BLEU (see `03` §6.2) — even a 50% POC must be faithful about the 50% it gets, not "fluently wrong."
5. **Human-in-the-loop** — the prof reads/grades the LaTeX; conversion assists (D13).
6. **Carry both privacy lanes (D5/D12)** — decide per-step on merit; self-hosting is feasible (D10).

---

> **Detailed procedures:** Phase 0 → [`07-data-collection-guide.md`](07-data-collection-guide.md); Phase 1 → [`05-phase1-protocol.md`](05-phase1-protocol.md); build design → [`06-architecture-spec.md`](06-architecture-spec.md). Decisions & open questions → [`../DECISIONS.md`](../DECISIONS.md).

## Phase 0 — Framing & data (Weeks 1–3)
**Goal:** know the target precisely and have real material to test on. **Full guide: [`07`](07-data-collection-guide.md).**

- [ ] Collect **50–100 real IITG math answer pages** (varied courses, handwriting, scan quality) — anonymized. This is the single most valuable asset.
- [ ] Hand-transcribe **~20 pages to gold LaTeX** (the eval set). Include mistakes *as written* (fidelity ground truth).
- [ ] Characterize the content mix: % prose / display eq / inline math / diagrams / tables; typical proof length; common notation.
- [x] **Real sample data available (D9)** — past exams/assignments obtainable. Phase 0 unblocked.
- [x] **Infra ready (D10)** — full GPU/CPU + intranet; self-hosting feasible.
- [x] **Input = flatbed scans (D11)** — light preprocessing.
- [ ] **Privacy lane (D5): carry both.** Not pre-committed — both cloud and self-hosted go into the Phase-1 bake-off and we decide on merit (D12). Both specified in [`06` §7](06-architecture-spec.md). *Leaning self-hosted default given D10, cloud as benchmark.*

**🚦 Gate 0:** Do we have ≥20 faithful gold pages? (Privacy no longer blocks — both lanes carried.) → then Phase 1.

---

## Phase 1 — Recognizer bake-off (Weeks 3–6) ← **the key de-risking experiment**
**Goal:** measure how well the best available recognizers actually transcribe *our* handwriting, faithfully.

- [ ] Run the 20 gold pages (cropped equations + full pages) through:
  - **Mathpix** (use the $29 free credit)
  - **Fine-tuned open VLM** — Qwen2.5-VL / Uni-MuMER-class (self-hosted)
  - **Frontier VLM** — Gemini 2.x and/or Claude (paid tier, if privacy lane allows)
  - *(optional)* a **PosFormer/ICAL** HMER model on cropped equations
- [ ] Score with the metrics from `03` §6: **Transcription Fidelity Rate**, **Over-correction Rate** (the safety metric), Compile rate, Render-match, CER (normalized).
- [ ] Qualitatively log failure modes (structure breaks, notation ambiguity, over-correction examples).

**🚦 Gate 1 (POC go/no-go — reframed per D14):**
- The bar is a **~50–60%-accurate proof-of-concept**, *not* production quality. Question: **does the best recognizer show enough promise (~50–60% faithful conversion) to justify iterating?**
- Is **over-correction** at least *visible* (flaggable)? For a POC a simple flag is enough; full dual-recognizer infra can wait for later iterations.
- If a recognizer clears ~50–60% faithful → **GO, build the POC** (Phase 2). If nothing comes close even at POC level → document and reconsider.

---

## Phase 2 — The POC (Weeks 6–12)
**Goal:** a working proof-of-concept — page in → LaTeX out — at ~50–60% accuracy, end-to-end but minimal. Prove it works before polishing.

- [ ] Pipeline: preprocess (light — flatbed, D11) → layout → route prose(HTR)/equations(Phase-1 winner) → assemble → compile-check.
- [ ] **Diagrams → embed as cropped images** (no TikZ).
- [ ] **Simple over-correction flag** (start basic; full dual-recognizer guard is a later iteration, N10).
- [ ] **Minimal review view:** scan and rendered LaTeX **side-by-side**, click-to-edit. (Prof reads/grades the LaTeX — D13.)
- [ ] Structured JSON side-car (regions, boxes, flags) — see `06` §3.

**🚦 Gate 2:** Does the POC convert a real page to *mostly-right* LaTeX (~50–60%) that a prof finds useful to read? If yes → iterate toward higher accuracy + full end-to-end (Phase 3+).

---

## Phase 3 — Evaluation, hardening & pilot (Weeks 12–20)
**Goal:** know if this helps a real grader, on a real (small) exam batch.

- [ ] Expand eval set; tune confidence thresholds (flag precision/recall).
- [ ] Robustness pass: messy scans, crossed-out work, margin notes, multi-page.
- [ ] **Shadow pilot** with 1–2 friendly IITG graders on a real assignment; measure grading-time delta and trust.
- [ ] If cloud APIs used: finalize DPA / legal review; else finalize self-hosting.
- [ ] Cost model from real usage (API + GPU + review time, `03` §5).

**🚦 Gate 3:** Does it measurably help (time saved, grader trust, acceptable cost)? → decide: productionize / iterate / shelve.

---

## Phase 4 — Productionization (the stated intent — DECISIONS.md D4)
The goal is a **real service for college professors at scale**, so this phase is planned-for, not hypothetical. Only begin once Gate 3 passes. Adds: batch/queue processing for whole-class volumes, auth + per-course isolation, storage/archival, monitoring & runbooks, the **retraining loop** from reviewer corrections ([`06` FEEDBACK](06-architecture-spec.md)), and a growing **bespoke IIT-level dataset** from accumulated corrections. Full component design in [`06-architecture-spec.md`](06-architecture-spec.md). Autonomous *grading* remains out of scope (D3) unless separately justified.

> **Production raises the stakes on over-correction:** at scale, silently "fixing" a student's error could cause **unfair grades** — an integrity issue. The dual-recognizer disagreement guard ([`06` §4](06-architecture-spec.md)) is a required safety feature, not optional polish.

---

## What we are explicitly NOT doing (per `00` non-goals)
- ❌ Autonomous grading / correctness judgment.
- ❌ TikZ generation for diagrams (embed as images; optional draft-only experiment).
- ❌ Non-English, online/stroke capture (offline scans only).
- ❌ Trusting a single VLM's transcription without a faithfulness guard.

---

## Timeline at a glance
| Phase | Weeks | Output | Gate |
|---|---|---|---|
| 0 Framing & data | 1–3 | gold eval set (real IITG pages) | data ready? |
| 1 Recognizer bake-off | 3–6 | fidelity numbers per recognizer | **POC go/no-go (~50–60%?)** |
| 2 **POC** | 6–12 | page→LaTeX, end-to-end, minimal | POC useful to a prof? |
| 3 Iterate + pilot | 12–20 | higher accuracy + grader pilot | does it help at scale? |
| 4 Full end-to-end / productionize | conditional | real service | — |

---

## The one-line strategy
**Build a faithful handwriting→LaTeX *POC* (~50–60%) on real IITG scripts first — prove the concept, then iterate toward a full end-to-end system. The prof always reads the LaTeX and grades.**
