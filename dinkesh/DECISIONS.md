# DECISIONS, OPEN QUESTIONS & NOTES

> **This file replaces chat as the source of truth.** Decisions you've made, questions I still need you to answer, and my running notes all live here. When you make a decision, it gets recorded here. Check the **⏳ OPEN QUESTIONS** section — those are blocking or shaping the work and I need your input.

_Last updated: 2026-07-17 (round 2)_

---

## ✅ DECISIONS MADE

| # | Decision | Value | Impact | Date |
|---|---|---|---|---|
| D1 | **Domain** | IIT Guwahati Mathematics — theory + equations + proofs | Hardest OCR case (prose + dense notation) | 2026-07-17 |
| D2 | **Language** | English only | Simplifies model/dataset choice | 2026-07-17 |
| D3 | **Purpose** | Assistive transcription to LaTeX for grading — **NOT auto-grading** | Faithfulness is the core constraint; humans decide marks | 2026-07-17 |
| D4 | **Scale/intent** | **Production — real scale, for professors in college** | Roadmap must plan for throughput, reliability, multi-user, archival — not a toy | 2026-07-17 |
| D5 | **Privacy lane** | **Open — "anything that helps"; not yet confirmed.** → **evaluate BOTH lanes at every step and lay out all alternatives** | Keep cloud + self-hosted both live; pick per-step on merit | 2026-07-17 |
| D6 | **Deliverables now** | Phase-1 protocol + architecture spec + data guide + beginner primer | Docs 05–08 created | 2026-07-17 |
| D7 | **Diagrams** | Embed as images; TikZ is a non-goal (optional draft-only experiment) | Removes the least-tractable sub-problem from critical path | 2026-07-17 |
| D8 | **Working reference** | The files in this repo, not chat | I write everything to disk; ask questions in chat, findings to files | 2026-07-17 |
| D9 | **Sample data** | ✅ **Real past exams/assignments available** (with permission) | Phase 0 unblocked; realistic difficulty | 2026-07-17 |
| D10 | **Team & infra** | **2-person team; full hardware — GPU + CPU + internet + intranet** | **Self-hosting fully feasible**; on-prem/air-gapped lane is real, not aspirational | 2026-07-17 |
| D11 | **Input format** | **Flatbed scanner** (clean, flat, consistent) | Easiest preprocessing; best-case input quality; dewarp/shadow largely unneeded | 2026-07-17 |
| D12 | **Standing rule** | **At every decision point, present all well-researched alternatives** with trade-offs — don't collapse to one option prematurely | Docs must be option-laying, not prescriptive | 2026-07-17 |
| D13 | **Scope & model solution** | Scope is **strictly handwriting → LaTeX conversion**; prof reads the LaTeX and grades. Model solution exists only **sometimes** → **fully optional**, never assumed or required. Never used to nudge a student's answer toward the key | Narrows scope cleanly; no dependency on answer keys | 2026-07-17 |
| D14 | **Accuracy target / staging** | **POC first at ~50–60% accuracy**, then iterate toward a full end-to-end system. Modest initial bar; prove the concept before polishing | Reframes roadmap as POC → iterate; Phase-1 bar is "does a ~50–60% POC show promise?" not "is it production-grade?" | 2026-07-17 |

---

## ⏳ OPEN QUESTIONS (need your input — these shape or block work)

> Answer any of these by telling me; I'll record the answer in the table above and update the affected docs. Ranked by how much they change the plan.

### ✅ Q1 — Privacy lane — ANSWERED (→ D5)
"Anything that helps; not confirmed yet." **Decision: keep both cloud and self-hosted live and evaluate both at every step (D12).** Because full on-prem infra exists (D10), the self-hosted lane is genuinely viable — so we're not forced onto cloud. Both stay in the Phase-1 bake-off.

### ✅ Q2 — Builder / infra — ANSWERED (→ D10)
**2-person team; full GPU + CPU + internet + intranet.** Self-hosting is feasible. No hard deadline stated (assumed relaxed pacing — correct me if there's an exam-cycle deadline).

### ✅ Q3 — Input format — ANSWERED (→ D11)
**Flatbed scanner.** Preprocessing is best-case (deskew/light denoise; dewarp & shadow-removal largely unnecessary). Still open (minor): single vs double-sided, one PDF per student vs per question, and rough volumes (pages/student × class size × #classes) — useful for the throughput model but not blocking. *Assumed: ~5–15 pages/student, classes 50–200; correct if wrong.*

### ✅ Q4 — Scope & model solution — ANSWERED (→ D13)
Our job is **only** the handwriting → LaTeX conversion. The **professor reads the resulting LaTeX and grades** — we don't grade, compare, or score. A model solution **sometimes** exists but is **fully optional** (nice-to-have for future side-by-side; never required, never used to alter a student's answer).

### ✅ Q5 — Accuracy target / staging — ANSWERED (→ D14)
**Build a POC first at ~50–60% accuracy**, then iterate toward a full end-to-end system. So the Phase-1 question becomes *"does a ~50–60%-accurate POC show enough promise to keep going?"* — not "is it production-ready?" Polishing (confidence gating, higher fidelity, throughput) comes in later iterations.

**→ All open questions resolved. This remains a research/planning study (no code, no data scaffolding — per your scope). Deliverables = the docs. When you're ready to act on it, Phase 0 (`docs/07`) is the first real-world step.**

### Q6 — Real sample sheets — ANSWERED (→ D9) ✅
Yes — **real past exams/assignments available.** Phase 0 is unblocked. Next concrete step: assemble ~50–100 anonymized pages + ~20 faithfully-transcribed gold pages (see `docs/07`).

---

## 📝 RUNNING NOTES (my observations, for you to read anytime)

- **N1 — The project's whole viability rests on one number.** Fidelity of transcription on *real IITG handwriting*. Everything in docs 05–08 exists to get to that number cheaply (Phase 1). Don't let infrastructure work start before that number is known.
- **N2 — "Production for profs" raises the stakes on the over-correction risk.** At scale, a system that silently "fixes" a student's wrong step into a right one could cause **unfair grades** — an integrity problem, not just a bug. The dual-recognizer disagreement guard (doc 02/06) is therefore not optional; it's a safety feature. I've written the architecture so grading stays human-decided.
- **N3 — Privacy stays open by design (D5/D12).** Rather than picking a lane, we carry both and let the Phase-1 bake-off + per-step trade-off tables decide. Every doc now lays out cloud vs self-hosted alternatives side by side.
- **N4 — Self-hosting is now clearly feasible (D10).** With full GPU/CPU + intranet, the on-prem / air-gapped lane is real. This is a strong position: we can keep student data entirely on-campus *and* still compare against cloud APIs on an anonymized sample. Leans the eventual default toward self-hosted for privacy, cloud as benchmark/backup.
- **N5 — Flatbed input is the easy case (D11) — good news.** Clean, flat scans mean preprocessing is mostly deskew + light denoise; the hard dewarp/shadow problems (which plague phone photos) largely disappear. More of the difficulty budget can go to recognition, where it's actually needed.
- **N6 — Beginner-friendly by request.** Per your profile, doc 08 explains everything from first principles with analogies. If any other doc is too dense, tell me and I'll add a plain-language version.
- **N7 — Sources verified (update).** The three 2025–2026 papers once flagged (FERMAT arXiv:2501.07244, VEHME 2510.22798, over-correction/PINK 2604.22774) are now **confirmed against arXiv** with exact titles. Only vendor pricing, SimpleTex specifics, and a couple of dataset licenses remain "reconfirm at citation time" (they change over time).
- **N8 — Standing rule now in effect (D12):** every step's docs present *all* researched alternatives with trade-offs, not a single prescribed choice. If any doc reads too prescriptive, that's a bug — tell me and I'll add the alternatives.
- **N9 — Phase 0 can start now (D9).** The gold-set build (docs/07) is the immediate next action; it's the only thing standing between us and the Phase-1 go/no-go number.
- **N10 — POC-first changes the bar (D14).** The goal is a ~50–60%-accurate proof-of-concept, then iterate — not production quality up front. This *lowers* the Phase-1 gate: we're asking "is this promising enough to continue?", not "is it done?". Good news, because ~50–60% is realistic for handwritten math today (see doc 01/03), whereas production-grade is not yet. The over-correction guard still matters, but for a POC it can start as a simple flag rather than full dual-recognizer infra.
- **N11 — Scope is narrow and clean (D13).** We convert handwriting → LaTeX and stop. No grading, no comparison to a key. This keeps the build focused and sidesteps the hardest/riskiest territory (auto-grading). The prof is always the grader.

---

## How to use this file
1. Skim **OPEN QUESTIONS** — answer what you can; each answer unblocks or sharpens a doc.
2. **RUNNING NOTES** is where I'll flag anything important I notice, so you never have to reconstruct it from chat.
3. When something changes, this file changes. It is always current as of the date at the top.
