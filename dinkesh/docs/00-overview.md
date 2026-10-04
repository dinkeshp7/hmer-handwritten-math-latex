# 00 — Problem & Context Overview

> **Project:** Handwritten mathematics answer sheets → LaTeX, to assist grading.
> **Stage:** Research / feasibility study (no implementation yet).
> **Domain:** IIT Guwahati — Department of Mathematics. Theory math, equations, and worked solutions/proofs.
> **Language:** English.
> **Status:** Draft — this doc is stable; companion docs `01`–`04` depend on ongoing research.

---

## 1. The problem in one sentence

Professors grade stacks of **scanned handwritten** exam answer sheets. We want a service that turns each sheet into a **clean, compilable LaTeX document** that faithfully reproduces the student's written answer — so the grader reads familiar, legible, typeset math instead of deciphering handwriting.

## 2. Why this matters (the "why", not just the "what")

- **LaTeX is the native language of mathematics.** Professors read, write, and think in typeset math. A page of clean LaTeX is faster and less error-prone to grade than variable-quality handwriting.
- **Legibility ≠ correctness, but it helps fairness.** Hard-to-read handwriting can cost students marks unfairly, or cost graders time. A faithful typeset rendering levels that out.
- **Downstream reuse.** Once answers are LaTeX, you unlock: side-by-side comparison with a model solution, partial-credit annotation, plagiarism/similarity checks, archival, and analytics on where students struggle.
- **It is NOT auto-grading.** The goal (at least initially) is *faithful transcription* to LaTeX — a legibility/assistance tool — **not** deciding whether the math is correct. That distinction is a core design constraint (see §6).

## 3. Who uses it

| User | Need | Interaction |
|---|---|---|
| **Professor / grader** (primary) | Read answers as clean typeset math; grade faster and more fairly | Reviews generated LaTeX / rendered PDF alongside (or instead of) the scan |
| **TA / evaluator** | Same as above, at higher volume | Bulk processing of a class's sheets |
| **Student** (indirect) | Fairer, more legible evaluation of their work | Does not use the tool directly (initially) |
| **Department / admin** (future) | Archival, analytics, audit | Consumes stored LaTeX + metadata |

## 4. The target content: IIT Guwahati Mathematics

This domain is deliberately the **hardest** OCR case: not isolated equations, but **full handwritten solutions** mixing prose and dense notation. Representative course areas and the notation they bring:

- **Calculus / Real Analysis** — limits, integrals, ∑/∏, ε–δ arguments, sup/inf, sequences & series.
- **Linear Algebra** — matrices, determinants, vectors, eigen-notation, spans, bases.
- **Differential Equations (ODE/PDE)** — derivatives (Leibniz & prime & dot), partial ∂, boundary conditions, operators.
- **Abstract Algebra** — groups/rings/fields, quotient notation, homomorphisms, set-builder notation.
- **Complex Analysis** — contour integrals, residues, complex plane notation.
- **Probability & Statistics** — expectations, distributions, combinatorics, conditional notation.
- **Topology / Functional Analysis** — set notation, norms, quantifiers, mappings.
- **Numerical Analysis** — algorithms, iteration, tabular data.

### What makes a real answer sheet hard (beyond a single equation)
1. **Prose + math interleaved** ("Let $f$ be continuous on $[a,b]$. Then …") — needs both handwriting text recognition *and* math recognition, plus knowing which is which.
2. **Multi-line structure** — aligned equation chains, cases, matrices, systems.
3. **Proof structure** — "Claim / Proof / ∴ / QED", numbered steps, indentation carrying meaning.
4. **Diagrams & figures** — graphs, geometric figures, number lines, commutative diagrams. (Very hard to LaTeX-ify; likely embedded as images — see `03`.)
5. **Messy real-world scans** — skew, noise, phone photos, faint pencil, crossed-out work, margin notes, arrows, non-linear reading order.
6. **Idiosyncratic handwriting** — every student writes symbols differently; ambiguity (e.g. handwritten $z$ vs $2$, $x$ vs $\times$, $\ell$ vs $1$).

## 5. What "success" looks like (feasibility-stage criteria)

Since this is a **feasibility study**, success is *understanding what is achievable and at what cost*, not shipping. Concretely, the study succeeds if we can answer:

- **Achievability:** What transcription accuracy is realistically attainable on IITG-style handwritten math today?
- **Best approach:** Which technical path (specialized OCR pipeline vs VLM-first vs hybrid) is most promising — see `02`.
- **Effort/cost:** Rough effort, data needs, and running cost to reach a usable prototype — see `03`.
- **Data reality:** Do the datasets/models we need exist, or must we build data — see `01`.
- **Human-in-the-loop:** How much human correction is unavoidable, and how to design for it.

A *later* production system would add measurable targets (accuracy thresholds, throughput, turnaround, reviewer effort per sheet). Placeholder metrics are defined in `03`.

## 6. Constraints & non-negotiables

- **Faithfulness over "helpfulness."** The system must transcribe **what the student wrote**, including mistakes. It must **not** silently "correct" wrong math into plausible-looking right math. This is the single biggest risk of LLM/VLM approaches (they hallucinate correct-looking math) and is treated as a first-class design constraint — see `02`/`03`.
- **Compilable output.** Generated LaTeX should compile (or clearly flag where it can't), so graders get a rendered PDF.
- **Human review by default.** At feasibility stage we assume a human verifies/corrects output; the tool assists, it does not replace the grader.
- **Data privacy.** Student exam sheets are sensitive personal data. Any use of third-party cloud APIs (Mathpix, GPT-4V, etc.) must be weighed against privacy/consent — evaluated in `01`.
- **English only** (agreed scope).

## 7. Explicit non-goals (for this stage)

- ❌ Automatic **grading** / marks assignment.
- ❌ Detecting whether the student's math is **correct**.
- ❌ Non-English scripts.
- ❌ Real-time / online (stroke-based) capture — we assume **offline** scanned images. (Online data like CROHME/MathWriting is discussed in `01` only as a training resource.)
- ❌ Perfect reproduction of **diagrams** as vector LaTeX/TikZ (out of scope; embed as images).

## 8. Glossary

| Term | Meaning |
|---|---|
| **OCR** | Optical Character Recognition — image → text. |
| **HTR** | Handwriting Text Recognition — OCR specialized for handwriting (prose). |
| **HMER** | Handwritten Mathematical Expression Recognition — recognizing handwritten *math* → structured form (e.g. LaTeX). |
| **VLM** | Vision-Language Model — multimodal LLM that takes images + text (e.g. GPT-4V, Claude, Qwen-VL). |
| **Offline vs Online** | Offline = static image of writing. Online = pen-stroke trajectory data (timing/coordinates). Exams are offline. |
| **Isolated expression** | A single cropped equation, vs a full multi-line page. |
| **CER / ExpRate / BLEU** | Evaluation metrics (defined in `03`). |
| **TikZ** | LaTeX package for drawing vector diagrams in code. |
| **Human-in-the-loop (HITL)** | Workflow where a person reviews/corrects model output. |

---

### Document map
- **00 — Overview** (this file): problem, users, scope, constraints.
- **01 — Landscape:** existing models, tools, datasets, services (research-backed).
- **02 — Technical approach:** candidate architectures & trade-offs.
- **03 — Feasibility, risks, evaluation:** what's achievable, metrics, effort/cost.
- **04 — Roadmap:** phased feasibility plan & decision gates.
