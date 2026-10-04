# 05 — Phase 1 Protocol: The Recognizer Bake-off

> **The single most important experiment in the project.** It answers: *"On real IITG handwriting, can any available recognizer transcribe faithfully enough — without over-correcting — to make human-assisted grading worthwhile?"* That answer is the **project go/no-go gate**. Do this before building anything.

> **Plain-language version:** we test the best "readers" on real IITG pages and measure two things: (1) do they type what the student *actually wrote*, and (2) how often do they secretly "fix" mistakes. If the answers are "yes" and "rarely," we proceed. See `08-primer.md` §7 if "over-correction" is unfamiliar.

---

## 1. Objective & success definition

**Objective:** quantify, on real IITG scripts, each candidate recognizer's:
- **Fidelity** — does it reproduce what was written (including errors)?
- **Over-correction rate** — how often does it silently "fix" student errors? (the safety metric — must be low or catchable)
- **Compile rate** — does the LaTeX compile?
- **Structural accuracy** — multi-line/matrix/fraction handling.
- **Confidence usefulness** — do its confidence scores actually predict errors (so we can route to humans)?

> **POC framing (D14):** the target is a **~50–60%-accurate proof-of-concept**, not production quality. Phase 1 asks *"is any recognizer promising enough (~50–60% faithful) to build a POC on?"* — a deliberately modest, realistic bar for handwritten math today.

**Phase 1 succeeds (GO — build the POC)** if **at least one** candidate achieves:
- **~50–60%+ faithful conversion** on real IITG pages (faithful = reproduces what was written; see §5.2), **AND**
- Over-correction is at least **visible/flaggable** (for a POC a simple flag suffices; full dual-recognizer guard is a later iteration), **AND**
- The output is something a professor would find **useful to read** vs. raw handwriting.

**Phase 1 fails (NO-GO / iterate)** if no candidate even reaches POC-level (~50–60%) faithful conversion, or all over-correct so pervasively that the LaTeX misrepresents student work. Document and reconsider before over-investing.

> Higher bars (≥70% no-edit regions, <5% over-correction, confidence-gated routing) become targets for **later iterations** toward the full end-to-end system — not the POC gate.

---

## 2. Prerequisites (from Phase 0 — see `07-data-collection-guide.md`)

- [ ] **≥20 real IITG answer pages** transcribed to **gold LaTeX** (faithful — mistakes preserved). This is the answer key.
- [ ] A **superset of ~50–100 raw pages** (un-transcribed) for qualitative failure-mode review.
- [ ] Pages **anonymized** (no student names/roll numbers) — required regardless of privacy lane.
- [ ] **Privacy lane decided** (DECISIONS.md Q1) — determines which candidates are even allowed. *If undecided: run only self-hosted + Mathpix-on-anonymized-sample-with-consent, defer cloud VLMs.*
- [ ] A mix in the gold set: pure-prose, single-equation, multi-line/aligned, matrix, a proof, a diagram page, a messy/low-quality scan.

---

## 3. Candidates to test

| # | Candidate | Type | Privacy lane | Notes |
|---|---|---|---|---|
| C1 | **Mathpix** | Commercial API | Cloud (has on-prem option) | Use $29 free credit; capture `confidence`/`confidence_rate` |
| C2 | **Fine-tuned open VLM** — Qwen2.5-VL (or Uni-MuMER weights) | Self-hosted | Either | The privacy-optimal bet; needs a GPU |
| C3 | **Frontier VLM** — Gemini 2.x and/or Claude | Cloud API | Cloud only | Paid tier (no-training); expect strong fluency, watch over-correction |
| C4 | **HMER specialist** — PosFormer/ICAL | Self-hosted | Either | On *cropped equations only* — a faithfulness baseline |
| C5 *(optional)* | **Surya 2** | Self-hosted | Either | Full-page; newer |

> Test C1–C3 on **full pages** and on **cropped equations**; C4 on cropped equations only. Not all candidates are viable in all privacy lanes — the allowed set follows DECISIONS.md Q1.

---

## 4. Procedure

### Step 4.1 — Prepare inputs
- Two input sets: **(a) full pages**, **(b) cropped single equations** (cut from the same pages). Cropped set isolates *pure recognition* from *layout* difficulty.
- Keep a manifest (CSV/JSON): `page_id, region_id, type, source_image_path, gold_latex`.

### Step 4.2 — Run each candidate
- For each candidate × each input, save: **raw LaTeX output**, **confidence scores** (if any), **latency**, **cost**.
- Use identical, minimal prompts for VLMs (no "fix errors," no "make it correct" — prompt explicitly: *"Transcribe exactly what is written, including any mathematical errors. Do not correct."*).
- Record prompt verbatim in results (reproducibility).

### Step 4.3 — Normalize before scoring
LaTeX is non-unique (`\frac{1}{2}` = `\tfrac12` visually). Before any string metric, **normalize** both gold and output (MathWriting-style: canonical sub/superscript order, `\sin`→`sin`, unify matrix envs, strip font/size/spacing). Document the normalizer.

### Step 4.4 — Score (see §5)
- Automatic metrics on normalized LaTeX.
- **Human fidelity judgment** on every region (the metric that matters most).
- Compile each output; record pass/fail and errors.

### Step 4.5 — Dual-recognizer disagreement test
- For each equation, compare the **faithful backbone** (C2/C4) vs the **fluent VLM** (C3) outputs.
- Measure: when they disagree, how often is the VLM the one that over-corrected? This validates the core safety mechanism (`02` §2C).

---

## 5. Metrics & scoring rubric

### 5.1 Automatic (on normalized LaTeX)
| Metric | How | Target |
|---|---|---|
| **Compile rate** | fraction that compiles | high; failures are review flags |
| **Exact match (ExpRate)** | normalized string equality per region | context only (all-or-nothing) |
| **CER** | token-level edit distance / length | lower better |
| **Render-match** | compile both, compare images (SSIM/pixel) | best semantic proxy |

### 5.2 Human judgment (the decisive metrics)
Two reviewers independently label each region (adjudicate disagreements):

| Label | Definition |
|---|---|
| **Faithful-perfect** | Renders exactly what the student wrote, mistakes included |
| **Faithful-minor** | Small fix needed (spacing, one symbol), still faithful |
| **Unfaithful-garbled** | Wrong/unreadable — needs redo |
| **⚠️ OVER-CORRECTED** | Output is "more correct" than what was written — **the dangerous case** |

Derived rates:
- **Fidelity Rate** = (perfect + minor) / total regions.
- **No-edit Rate** = perfect / total (proxy for reviewer speed).
- **Over-correction Rate** = over-corrected / total. **This is the safety gate.**

### 5.3 Confidence usefulness
- Plot confidence vs. actual error. Compute **flag precision/recall** at candidate thresholds. Good confidence = we can auto-route the bad 25% to humans and trust the rest.

### 5.4 Operational
- **Latency/page**, **cost/page** (record for the production cost model, `03` §5).

---

## 6. Results template (fill this in — write to `results/phase1-results.md`)

```
## Phase 1 Results — <date>

### Setup
- Gold pages: N = __   | Cropped equations: N = __
- Candidates run: [C1 Mathpix, C2 Qwen, C3 Gemini, ...]
- Prompt used (VLMs): "________"
- Normalizer version: ____
- Privacy lane at test time: ____

### Headline table
| Candidate | Fidelity Rate | No-edit Rate | OVER-CORRECTION | Compile % | CER | Latency | Cost/pg |
|-----------|--------------:|-------------:|----------------:|----------:|----:|--------:|--------:|
| Mathpix   |               |              |                 |           |     |         |         |
| Qwen2.5VL |               |              |                 |           |     |         |         |
| Gemini    |               |              |                 |           |     |         |         |
| PosFormer |               |              |                 |     (eq)  |     |         |         |

### Dual-recognizer disagreement
- Disagreements: __/__ equations. Of these, VLM over-corrected in __%.
- Verdict: does disagreement-flagging catch over-correction reliably? Y/N

### Confidence usefulness
- Flag precision/recall at threshold __: __ / __

### Failure modes observed (qualitative)
- Structure breaks: ...
- Notation ambiguity: ...
- Over-correction examples (screenshots/latex): ...

### GATE DECISION
- [ ] GO   [ ] NO-GO   [ ] ITERATE
- Best candidate: ____   Rationale: ____
- If GO → proceed to Phase 2 with recognizer = ____
```

---

## 7. Decision gate (Gate 1)

Record the outcome in **DECISIONS.md** and the results file.

- **GO** → pick the winning recognizer; proceed to Phase 2 (thin prototype, `06`).
- **ITERATE** → try fine-tuning C2 on more IITG data, better preprocessing, or prompt changes; re-run.
- **NO-GO** → conclude "not feasible at acceptable quality yet" with evidence. This saves months. Revisit when models improve.

---

## 8. Effort & timeline
- **~2–4 weeks**, assuming gold data exists (Phase 0 done).
- Bulk of effort: **human fidelity labeling** (the decisive metric) and building the normalizer + scoring scripts.
- Cheap in money (Mathpix free credit; one GPU rental for C2; small VLM API spend for C3).

---

## 9. Common pitfalls (avoid these)
- ❌ Scoring only with BLEU/ExpRate — they miss over-correction entirely. **Human fidelity labeling is mandatory.**
- ❌ Prompting VLMs to "produce correct LaTeX" — that *induces* over-correction. Prompt for faithful transcription.
- ❌ Testing on clean/printed samples — must be **real messy IITG handwriting** or results are meaningless.
- ❌ Skipping normalization — inflates error rates on visually-identical LaTeX.
- ❌ Sending non-anonymized data to any cloud API.
