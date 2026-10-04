# Handwritten Math → LaTeX (for assisting exam grading)

**Research / feasibility study.** Can we turn scanned **handwritten** IIT Guwahati mathematics answer sheets (theory + equations + proofs, English) into clean, compilable **LaTeX** — so professors grade familiar typeset math instead of deciphering handwriting?

> **Stage:** research only — no implementation yet. Everything lives in `docs/`.

---

## The one-paragraph answer

An **assistive** system is feasible today; a fully-automatic one is not. Printed-math OCR is solved, but **handwritten** math→LaTeX is not: the best models hit only **~62–80% exact-match on clean isolated equations**, no off-the-shelf system does full **handwritten page → LaTeX**, and general VLMs (GPT-4o/Claude/Gemini) **silently "correct" students' wrong math 42–66% of the time** — the one thing a grading tool must never do. So the realistic product is **faithful transcription + confidence-gated human review** (a hybrid pipeline), *not* automated grading. The single most important next step is a **recognizer bake-off on real IITG scripts, scored for fidelity** — that one experiment decides whether the project is viable.

---

## 🧭 First time here? → [MASTER.md](MASTER.md)
The **whole study in one read**, with pointers into the detailed docs. Start there.

## 📄 Pitching this? → [EXECUTIVE-SUMMARY.md](EXECUTIVE-SUMMARY.md)
A single skimmable page for professors / the department.

## ⚠️ If you read only one file besides this

**[DECISIONS.md](DECISIONS.md)** — this replaces chat as the source of truth. It holds every decision made, the **open questions I need you to answer** (some block Phase 1), and my running notes. Check it first.

## Read in this order

| # | Doc | What's in it |
|---|---|---|
| — | [**DECISIONS.md**](DECISIONS.md) | **Decisions, open questions for you, running notes — start here** |
| 08 | [Primer](docs/08-primer.md) | **Plain-language "how it all works" — read this if anything is confusing** |
| 00 | [Overview](docs/00-overview.md) | Problem, users, target content (IITG math), scope, constraints, glossary |
| 01 | [Landscape](docs/01-landscape.md) | Models, tools, datasets, commercial services — cited, with the 5 key findings |
| 02 | [Approach](docs/02-approach.md) | The pipeline stages + 3 candidate architectures (VLM-first / specialized / **hybrid**) |
| 03 | [Feasibility](docs/03-feasibility.md) | Realistic accuracy, ranked risks, hard cases, cost, **evaluation metrics** |
| 04 | [Roadmap](docs/04-roadmap.md) | Phased plan with go/no-go **decision gates** |
| 05 | [Phase 1 Protocol](docs/05-phase1-protocol.md) | The recognizer bake-off — the experiment that decides viability |
| 06 | [Architecture Spec](docs/06-architecture-spec.md) | Component design, region JSON schema, review UI, both privacy lanes |
| 07 | [Data Collection Guide](docs/07-data-collection-guide.md) | How to gather + faithfully label real IITG scripts (Phase 0) |
| 09 | [Resources](docs/09-resources.md) | **All models, tools, datasets, papers, services — links in one place** |

**Intent:** production system for real use by college professors (not a toy) — an *assistive transcription* tool, humans still decide marks.

**Confirmed context (see [DECISIONS.md](DECISIONS.md)):** real past-exam data available · 2-person team with full GPU/CPU + intranet (self-hosting feasible) · **flatbed scans** (easy preprocessing) · privacy lane kept open — **both cloud and self-hosted evaluated at every step**, all alternatives laid out (no premature single choice).

**Strategy (D14):** **POC first at ~50–60% accuracy, then iterate** toward a full end-to-end system. Scope is strictly **handwriting → LaTeX conversion** — the professor reads the LaTeX and grades (D13). We do not grade or compare to an answer key.

---

## Five findings that shape everything

1. **Printed math is solved; handwritten math is not** — specialized tools (pix2tex, Nougat, Surya) are printed-first; real handwriting lives in the HMER research line, **isolated equations only, ~62–80% exact**.
2. **No off-the-shelf full-page handwritten→LaTeX exists** — it's an open research problem; any build is a **staged pipeline**.
3. **VLMs over-correct student errors 42–66% of the time** (worse with better models) — **disqualifying** for naive "VLM reads and grades."
4. **Mathpix is the most deployable recognizer** (real handwriting, LaTeX, confidence scores, ~$0.005/page, on-prem option) — but no rigorous third-party handwritten benchmark; SimpleTex is China-hosted (privacy blocker); self-hosted Qwen2.5-VL is the privacy-optimal route.
5. **The IIT-level dataset we need doesn't exist** — all datasets are isolated expressions; full-solution sets (FERMAT, VEHME) are school-grade and small. **Plan to build our own.**

---

## Recommended direction

**Hybrid pipeline (Approach C):** a *faithful* recognizer backbone + VLM as layout/second-opinion assistant + **dual-recognizer disagreement flagging** (to catch over-correction) + **confidence-gated human-in-the-loop** review. Diagrams → **embedded images**, not TikZ. Prove it on real IITG scripts in **Phase 1** before building anything bigger.

---

## Deeper working references (docs 05–08)
- **[05 Phase-1 Protocol](docs/05-phase1-protocol.md)** — step-by-step recognizer bake-off + scoring templates. The go/no-go experiment.
- **[06 Architecture Spec](docs/06-architecture-spec.md)** — production component design, region JSON (the SSOT), over-correction safety guard, review UI, both privacy lanes.
- **[07 Data Guide](docs/07-data-collection-guide.md)** — how to build the gold set faithfully (Phase 0, the #1 dependency).
- **[08 Primer](docs/08-primer.md)** — first-principles, analogy-driven explanation of everything.

## Caveats on the research
Compiled 2026 from direct arXiv / GitHub / vendor-doc sources. The three 2025–2026 papers once flagged for verification (FERMAT, VEHME, the over-correction/PINK paper) have been **confirmed against arXiv** with exact titles (see `01`/`09`). Remaining items to reconfirm at citation time: **vendor pricing, SimpleTex specifics, and some dataset licenses** (these change). Vendor "accuracy" claims are treated as marketing unless a methodology is cited.
