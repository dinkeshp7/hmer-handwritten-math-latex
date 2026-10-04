# 07 — Data Collection & Annotation Guide

> How to gather and label the real IITG handwriting we need. This is **Phase 0**, and it is the **#1 dependency** — the whole study is blocked without it (DECISIONS.md Q6). No public dataset covers IIT-level handwritten proofs (see `01` §5.2), so **we must build our own.**

> **Plain-language note:** we need two things — (1) a pile of real handwritten pages to test on, and (2) a smaller set where we've *perfectly hand-typed* the correct LaTeX, to grade the machine against (the "answer key"). See `08-primer.md` §9 step 1.

---

## 1. What we're collecting, and why

| Asset | Size (Phase 0) | Purpose |
|---|---|---|
| **Raw corpus** | 50–100 anonymized answer pages | Realistic variety for qualitative failure-mode review |
| **Gold set** | ~20 pages, transcribed to faithful LaTeX | The scored answer key for Phase 1 (`05`) |
| **Region labels** (optional but valuable) | bounding boxes + type per region on gold pages | Trains/评 layout; enables per-region scoring |

Later (Phase 4) a much larger corpus (hundreds–thousands of pages) would support fine-tuning — but **Phase 0 only needs ~20 gold + ~50–100 raw**. Start small; the goal is a go/no-go signal, not a training set.

---

## 2. Sourcing the pages

Prefer, in order:
1. **Past exams/assignments already graded** — lowest friction, real difficulty. Get instructor permission.
2. **Volunteer students** re-writing sample solutions — controllable variety.
3. **Faculty/TA-written mock answers** — fastest, but *less realistic handwriting* (use only to bootstrap, not as the main gold set).

**Aim for deliberate variety** (this makes Phase 1 meaningful):
- [ ] Multiple **courses** (calculus, linear algebra, analysis, algebra, ODE/PDE…).
- [ ] Multiple **writers** (≥10 different hands — handwriting varies enormously).
- [ ] A spread of **content types**: pure prose, single equations, multi-line/aligned derivations, matrices, a full proof, a page with a diagram, a table.
- [ ] A spread of **scan quality**: clean flatbed, phone photo, faint pencil, some crossed-out work.

> **Tie to DECISIONS.md Q3:** collect pages in the **same format students will actually submit** (scan vs phone photo). Testing on clean scans then deploying on phone photos would invalidate results.

---

## 3. Anonymization (mandatory, before anything else)

Required **regardless of privacy lane** — and essential before any cloud API touches the data.

- [ ] Remove/black-out **names, roll numbers, signatures** (crop headers or redact).
- [ ] Rename files to **opaque IDs** (`MA101-mid-2026-p03`, not the student's name).
- [ ] Keep a **separate, access-controlled** mapping if you ever need to trace back (store offline, not with the images).
- [ ] Get **consent/permission** appropriate to your institution's policy (involve whoever owns data governance — this also informs Q1).
- [ ] If phone photos: strip **EXIF metadata** (can contain location/device).

---

## 4. Creating the gold LaTeX (the answer key) — the critical craft

This is where fidelity is defined. **The golden rule:**

> ✍️ **Transcribe EXACTLY what the student wrote — including every mistake. Do NOT fix, simplify, or "clean up" the math.**

If the student wrote `∫₀¹ x² dx = 1/2` (wrong), the gold LaTeX is `\int_0^1 x^2\,dx = \frac{1}{2}` — the wrong answer, faithfully. This is the entire point: it's how we later measure whether a machine **over-corrects** (`08` §7).

### Conventions (keep gold consistent so scoring is fair)
- **Normalize your own style** so two annotators produce comparable LaTeX: agree on `\frac` vs `\dfrac`, `\int_0^1` spacing, `\,` usage, matrix env (`bmatrix` vs `pmatrix`), etc. Write these in a short `gold-conventions.md`.
- **Prose:** transcribe as plain text with inline math in `$...$`. Preserve wording and obvious spelling as written.
- **Structure:** use `align`/`aligned` for multi-line derivations, `cases` for case-splits, matching the student's layout.
- **Diagrams:** do **not** transcribe — mark `% [DIAGRAM: brief description]` and note the region. (Consistent with embed-as-image, D7.)
- **Crossed-out work:** preserve it, marked (e.g. `% [CROSSED OUT: ...]`) — never silently drop.
- **Illegible bits:** mark `\text{[illegible]}` rather than guessing.
- **Uncertainty:** if *you* can't tell `z` from `2`, flag it — those are exactly the cases the machine will struggle with too.

### Double-annotation for quality
- Have **two people** transcribe an overlapping subset; compare. Disagreements reveal ambiguous handwriting and tighten conventions. This also gives a **human-vs-human fidelity ceiling** — the machine can't be expected to beat what two humans disagree on.

---

## 5. Region labeling (optional, higher value if time allows)

For each gold page, draw **bounding boxes** and tag each region's **type** (`prose | display_math | inline_math | table | diagram | crossed_out`) and **reading order**. This enables:
- Per-region scoring in Phase 1 (isolate recognition vs layout errors).
- A small eval set for the LAYOUT component (`06`).

**Tools:** Label Studio, CVAT, or Roboflow (all support bbox + label export to JSON/COCO). Export to match the region JSON schema in `06` §3.

---

## 6. Storage & structure

```
data/
  raw/                     # anonymized source images (the 50–100)
    MA101-mid-2026-p03.png
  gold/                    # the ~20 with faithful LaTeX
    MA101-mid-2026-p03.tex
    MA101-mid-2026-p03.json    # regions + bboxes (if labeled)
  manifest.csv             # page_id, course, writer_id(anon), types, scan_quality, has_gold
  gold-conventions.md      # your LaTeX style rules
  consent/                 # permissions (access-controlled)
```

- `manifest.csv` is the index for Phase 1 — one row per page with its attributes.
- Keep `data/` **out of any cloud sync** unless the privacy lane (Q1) permits.

---

## 7. How much is enough?

| Purpose | Minimum | Comfortable |
|---|---|---|
| Phase 1 go/no-go signal | ~20 gold pages | ~40 gold + 100 raw |
| Layout eval | ~20 labeled pages | ~50 |
| Fine-tuning (Phase 4 only) | — | hundreds–thousands |

> Don't over-collect for Phase 1. Twenty faithful gold pages spanning the variety in §2 is enough to decide viability. Scale data **after** a GO.

---

## 8. Checklist (Phase 0 done when all ✅)
- [ ] 50–100 raw pages gathered, varied per §2
- [ ] All pages anonymized (§3) + consent handled
- [ ] ~20 pages transcribed to **faithful** gold LaTeX (§4), mistakes preserved
- [ ] `gold-conventions.md` written; double-annotation on a subset
- [ ] (optional) regions labeled (§5)
- [ ] `manifest.csv` complete
- [ ] Privacy lane (Q1) decided or gold data confirmed safe for planned candidates
- [ ] → **Ready for Phase 1 (`05`)**

---

## 9. Notes
- **Faithfulness in the gold set is non-negotiable.** If annotators "tidy up" the math, we lose the ability to measure over-correction — the project's central risk. Brief annotators on this explicitly.
- **Variety beats volume** at this stage. Ten writers × two pages each teaches more than one writer × twenty.
- This guide's gold set doubles as the **Phase 1 eval set** and later seeds the **retraining loop** (`06` FEEDBACK).
