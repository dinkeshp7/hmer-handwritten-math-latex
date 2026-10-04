# Handwritten Math → LaTeX: Executive Summary

**One-line pitch:** A tool that reads students' **handwritten** math exam sheets and retypes them as clean **LaTeX**, so professors grade legible, typeset math instead of deciphering handwriting.

**Prepared for:** IIT Guwahati, Department of Mathematics · **Stage:** research / feasibility study (no code yet) · **Date:** July 2026

---

## The problem
Grading stacks of handwritten answer sheets is slow and inconsistent — every student writes differently, and hard-to-read work costs both time and fairness. Professors already think in LaTeX; giving them typeset answers speeds grading and improves consistency.

## What we would build
A tool that does **one job: convert handwriting → LaTeX.** The professor still reads the LaTeX and assigns marks. We do **not** auto-grade or judge correctness.

```
Scanned answer sheet  →  [ convert ]  →  Clean LaTeX  →  Professor reads & grades
```

## What we found (the honest picture)
| | Finding |
|---|---|
| ✅ **Feasible** | A working proof-of-concept at **~50–60% accuracy** is realistic with today's tools. |
| ⚠️ **Not instant** | Printed-math OCR is solved; **handwritten** math is not — the best models get single equations exactly right only ~60–80% of the time, less on messy full pages. High accuracy is an *iterative* climb. |
| 🚫 **Key risk** | General AI models (GPT-4o, etc.) **silently "correct" a student's wrong math 42–66% of the time.** For grading, that's dangerous — it hides real mistakes. Our design specifically guards against this: it transcribes **exactly what was written, errors included.** |
| 🔒 **Privacy-safe option** | We have the hardware to run everything **on-campus** — student data need never leave IITG. |

## Why now / why us
- **Real data available** — past exam scripts to test on.
- **Infrastructure ready** — GPUs + intranet in place; can run fully on-campus (private) or benchmark against cloud services.
- **Easy input** — flatbed scans (clean, consistent), the best-case for recognition.
- **Directly relevant research exists** — incl. FERMAT (IIT Madras/AI4Bharat), a handwritten-math benchmark for Indian education.

## The plan (staged, low-risk)
| Phase | What | Gate |
|---|---|---|
| **0. Data** | Gather ~50–100 real pages; hand-transcribe ~20 as an answer key | data ready? |
| **1. Test** | Compare available recognizers on real IITG handwriting | **Is any ~50–60% good? → go/no-go** |
| **2. POC** | Build a minimal page → LaTeX prototype + review screen | useful to a prof? |
| **3. Pilot** | Try it with a real professor on a small batch | does it save time? |
| **4. Scale** | Iterate toward a full, production-grade service | — |

Each phase ends in a decision gate — **we never over-invest before proving the previous step.** The single experiment that decides everything is Phase 1: *does any recognizer read our handwriting well enough?*

## The ask
- Access to **anonymized past exam scripts** for testing.
- A **professor/TA** willing to try the pilot and give feedback.
- Agreement on **data-handling** (on-campus vs. cloud).

## Bottom line
The concept is sound and the hard risks are known and manageable. The realistic path is **prove a modest proof-of-concept first, then improve** — not promise perfection on day one. Low cost to reach the go/no-go decision; the main input needed is real sample sheets.

---
*Full study: `README.md` → `docs/00`–`09`. Decisions & open items: `DECISIONS.md`. Plain-language explainer: `docs/08-primer.md`.*
