# MASTER DOCUMENT — Handwritten Math → LaTeX

> **One document, whole picture.** This synthesizes the entire study so a first-time reader gets the full idea in one read. Every section points to a detailed doc if you want to go deeper. If you read nothing else, read this.
>
> **Project:** convert scanned **handwritten** IIT Guwahati mathematics answer sheets into clean **LaTeX**, so professors grade legible typeset math instead of deciphering handwriting.
> **Stage:** research / feasibility study — **no code yet**. Deliverable = these documents.
> **Date:** July 2026

---

## Table of contents
1. [The idea in 60 seconds](#1-the-idea-in-60-seconds)
2. [Why this matters](#2-why-this-matters)
3. [Scope — what it is and isn't](#3-scope--what-it-is-and-isnt)
4. [How it would work (the pipeline)](#4-how-it-would-work-the-pipeline)
5. [What the research found](#5-what-the-research-found)
6. [The one big danger: over-correction](#6-the-one-big-danger-over-correction)
7. [Technical approaches considered](#7-technical-approaches-considered)
8. [Feasibility & realistic accuracy](#8-feasibility--realistic-accuracy)
9. [The plan (roadmap & gates)](#9-the-plan-roadmap--gates)
10. [Decisions locked in](#10-decisions-locked-in)
11. [Key resources](#11-key-resources)
12. [Document map](#12-document-map)

---

## 1. The idea in 60 seconds

A professor faces a stack of handwritten math exams. Reading messy handwriting is slow, and every student writes symbols differently. This tool reads each page and **retypes it as LaTeX** — the clean, typeset language mathematicians use. The professor then reads neat, familiar math and grades it.

```
Scanned answer sheet  →  [ convert to LaTeX ]  →  Clean LaTeX  →  Professor reads & grades
```

The tool's **only** job is faithful conversion. It transcribes **exactly what the student wrote — mistakes included** — and never decides whether the math is right. The human grades.

> 📖 Plain-language explainer with analogies: [`docs/08-primer.md`](docs/08-primer.md)

---

## 2. Why this matters

- **LaTeX is mathematics' native language.** Professors read, write, and think in it; typeset answers are faster and less error-prone to grade than variable handwriting.
- **Fairness.** Hard-to-read handwriting can unfairly cost marks or grader patience. A faithful typeset rendering levels that.
- **Downstream reuse.** Once answers are LaTeX: side-by-side with a model solution, partial-credit annotation, archival, and analytics on where students struggle.

> 📖 Full context, users, glossary: [`docs/00-overview.md`](docs/00-overview.md)

---

## 3. Scope — what it is and isn't

| ✅ In scope | ❌ Out of scope |
|---|---|
| Handwriting → LaTeX **conversion** | **Auto-grading** / deciding correctness |
| Theory + equations + proofs (IITG math) | Non-English scripts |
| English, offline **scanned** pages (flatbed) | Live pen-stroke / tablet capture |
| Diagrams **embedded as images** | Diagram → TikZ vector generation |
| Faithful transcription (errors preserved) | "Improving" or completing student work |

The professor always reads the LaTeX and assigns marks. We assist legibility; we don't judge.

> 📖 Decisions behind scope: [`DECISIONS.md`](DECISIONS.md) (D3, D7, D13)

---

## 4. How it would work (the pipeline)

No single tool does "full handwritten page → LaTeX" — it's an open research problem. So the design is a **staged pipeline**, each stage one job:

```
Scan → [A] clean up → [B] find regions → [C] reading order →
        [D] read each region (prose / equation / diagram) →
        [E] reassemble into LaTeX → [F] check & flag → [G] human review → Clean LaTeX
```

- **Prose** → handwriting text recognition. **Equations** → math recognizer. **Diagrams** → cropped and embedded as images (not vectorized).
- Because input is **flatbed scans** (clean, flat), the "clean up" stage is light — no dewarping/shadow removal needed.
- A **human reviews** at the end: the machine gets most of it right, the human fixes the rest — far faster than transcribing from scratch.

> 📖 Component-level design, data schema, review-UI sketch: [`docs/06-architecture-spec.md`](docs/06-architecture-spec.md)
> 📖 Approaches & trade-offs: [`docs/02-approach.md`](docs/02-approach.md)

---

## 5. What the research found

Five findings shape everything (full survey with citations in [`docs/01-landscape.md`](docs/01-landscape.md)):

1. **Printed math → LaTeX is solved; handwritten is not.** Polished tools (pix2tex, Nougat, Surya) are printed-first. Genuine handwriting lives in the academic **HMER** line — and only for **single isolated equations at ~62–80% exact-match** on clean benchmarks.
2. **No off-the-shelf system does the full task** (handwritten page → coherent LaTeX). Any build is a staged pipeline.
3. **General AI models (GPT-4o/Claude/Gemini) silently "correct" students' wrong math 42–66% of the time** — worse with more capable models. Disqualifying for a naive "AI reads and grades" approach. *(Confirmed: arXiv 2604.22774, the PINK-metric paper.)*
4. **Mathpix** is the most deployable commercial recognizer (real handwriting, LaTeX output, confidence scores, ~$0.005/page, on-prem option). **Self-hosted Qwen2.5-VL** is the privacy-optimal alternative. SimpleTex is China-hosted (residency blocker).
5. **The IIT-level dataset we'd want doesn't exist.** All public datasets are isolated expressions; full-solution sets (FERMAT, VEHME — both confirmed) are school-grade and small. Expect to build our own.

> 📖 Every model/tool/dataset/service with links: [`docs/09-resources.md`](docs/09-resources.md)

---

## 6. The one big danger: over-correction

**This is the most important idea in the project.** Powerful AI models are *so* good at math that when they see a student's **wrong** step, they often **silently rewrite it into the correct one** — their "math brain" overrides their eyes. This happens in **42–66% of transcriptions**.

For grading, that's a disaster: if a student wrote `2+2=5` and the tool "helpfully" writes `2+2=4`, the professor grades a mistake that isn't there. **Unfair marks.**

**Our defense — the "two witnesses" trick:** run **two different recognizers** on each equation. If they **disagree**, flag it for the human. A model that over-corrected will differ from a faithful reader, so disagreement catches it. The backbone recognizer is chosen for faithfulness; the clever AI is treated with suspicion.

**The rule, everywhere in this project:** transcribe *what was written*, never "improve" it.

> 📖 Detail: [`docs/06-architecture-spec.md`](docs/06-architecture-spec.md) §4 · plain-language: [`docs/08-primer.md`](docs/08-primer.md) §7

---

## 7. Technical approaches considered

| Approach | Summary | Verdict |
|---|---|---|
| **A — VLM-first** | One big model does everything | ❌ Fast to prototype but over-corrects; unsafe as sole transcriber |
| **B — Specialized pipeline** | Distinct model per stage | ✅ Faithful, but heaviest to build; weak links on handwritten layout/assembly |
| **C — Hybrid (recommended)** | Faithful recognizer backbone + AI as assistant + disagreement flagging + human review | ✅ Best balance of faithfulness, usability, effort |

**Recommended direction: Approach C**, scoped down for a first proof-of-concept.

> 📖 Full comparison: [`docs/02-approach.md`](docs/02-approach.md)

---

## 8. Feasibility & realistic accuracy

- **A ~50–60% proof-of-concept is realistic now.** High production accuracy is a longer, iterative climb.
- Isolated clean handwritten equations: ~62–80% exact. Messy full pages: lower (errors compound across stages).
- **Success is measured by *fidelity*** (did it reproduce what was written, mistakes included?) — **not** BLEU or "is the math right?" The safety metric is **over-correction rate** (target ≈ 0 or reliably flagged).
- **Cost:** Mathpix ≈ $0.005/page; self-hosted ≈ compute only (we have the hardware). The real cost is **human review time** — but even assisted review beats deciphering raw handwriting.

> 📖 Risks, metrics, cost, hard cases: [`docs/03-feasibility.md`](docs/03-feasibility.md)

---

## 9. The plan (roadmap & gates)

Staged, with a go/no-go **gate** after each phase so we never over-invest.

| Phase | What | Gate |
|---|---|---|
| **0. Data** | Gather ~50–100 real pages; hand-transcribe ~20 as a faithful answer key | data ready? |
| **1. Test (bake-off)** | Compare recognizers on real IITG handwriting; measure fidelity + over-correction | **~50–60% achievable? → project go/no-go** |
| **2. POC** | Minimal page → LaTeX prototype + side-by-side review screen | useful to a prof? |
| **3. Pilot** | Try with a real professor on a small batch | does it save time? |
| **4. Scale** | Iterate toward full production service | — |

**Strategy: POC first (~50–60%), then iterate** — prove the concept before promising perfection. The single decisive experiment is **Phase 1**: *can any recognizer read our handwriting well enough?*

> 📖 Detailed roadmap: [`docs/04-roadmap.md`](docs/04-roadmap.md) · Phase-0 how-to: [`docs/07-data-collection-guide.md`](docs/07-data-collection-guide.md) · Phase-1 protocol: [`docs/05-phase1-protocol.md`](docs/05-phase1-protocol.md)

---

## 10. Decisions locked in

| # | Decision |
|---|---|
| D3 | Assistive **conversion only** — humans grade |
| D4 | Built for **production scale** for college profs (not a toy) |
| D5/D12 | Privacy lane open — **carry both cloud + self-hosted**, lay out all alternatives per step |
| D7 | Diagrams **embedded as images**; TikZ is a non-goal |
| D9 | **Real past-exam data available** → Phase 0 unblocked |
| D10 | 2-person team + full GPU/CPU/intranet → **self-hosting feasible** (on-campus, private) |
| D11 | Input = **flatbed scans** → light preprocessing |
| D13 | Scope strictly handwriting → LaTeX; model solution optional, never required |
| D14 | **POC first at ~50–60% accuracy**, then iterate |

All open questions resolved. Next real-world step is Phase 0 (gather sample sheets).

> 📖 Full decision log + notes: [`DECISIONS.md`](DECISIONS.md)

---

## 11. Key resources

- **Recognizers:** [Mathpix](https://mathpix.com) · [Qwen2.5-VL](https://huggingface.co/Qwen) · [PosFormer](https://github.com/SJTU-DeepVisionLab/PosFormer) · [Uni-MuMER](https://arxiv.org/abs/2505.23566) · [TrOCR](https://huggingface.co/microsoft/trocr-base-handwritten)
- **Directly relevant papers (verified):** [FERMAT — "Can VLMs Evaluate Handwritten Math?"](https://arxiv.org/abs/2501.07244) · [VEHME](https://arxiv.org/abs/2510.22798) · [Over-correction / PINK](https://arxiv.org/abs/2604.22774)
- **Datasets:** [CROHME](https://www.cs.rit.edu/~crohme2019/) · [MathWriting](https://arxiv.org/abs/2404.10690) · [HME100K](https://arxiv.org/abs/2207.11463)

> 📖 Full linked list (all types): [`docs/09-resources.md`](docs/09-resources.md)

---

## 12. Document map

| File | Purpose |
|---|---|
| **[EXECUTIVE-SUMMARY.md](EXECUTIVE-SUMMARY.md)** | One page to pitch to profs / department |
| **MASTER.md** (this) | Whole study in one read, with pointers |
| [README.md](README.md) | Entry point / index |
| [DECISIONS.md](DECISIONS.md) | Decisions, resolved questions, running notes |
| [docs/00-overview.md](docs/00-overview.md) | Problem, users, scope, glossary |
| [docs/01-landscape.md](docs/01-landscape.md) | Models, tools, datasets, services (cited) |
| [docs/02-approach.md](docs/02-approach.md) | Candidate architectures & trade-offs |
| [docs/03-feasibility.md](docs/03-feasibility.md) | Accuracy, risks, metrics, cost |
| [docs/04-roadmap.md](docs/04-roadmap.md) | Phased plan with decision gates |
| [docs/05-phase1-protocol.md](docs/05-phase1-protocol.md) | The recognizer bake-off (decisive experiment) |
| [docs/06-architecture-spec.md](docs/06-architecture-spec.md) | Component design, data schema, review UI |
| [docs/07-data-collection-guide.md](docs/07-data-collection-guide.md) | Gathering + faithfully labeling data |
| [docs/08-primer.md](docs/08-primer.md) | First-principles explainer (analogies) |
| [docs/09-resources.md](docs/09-resources.md) | All links in one place |

---

**Bottom line:** the concept is sound, the risks are known and manageable, and the realistic path is *prove a modest proof-of-concept, then improve*. The main input needed now is real sample sheets to run the Phase-1 test.
