# 08 — Beginner's Primer: How This All Works

> Plain-language, first-principles explanation of the whole project. No jargon without an analogy. Read this if any other doc feels too dense. If you understand this file, you understand the project.

---

## 1. What are we actually trying to do?

Imagine a professor with a stack of 200 handwritten exam papers. Reading messy handwriting is slow and tiring, and every student writes symbols differently. We want a machine that reads each handwritten page and **retypes it in LaTeX** — the clean, typeset language mathematicians use — so the professor reads neat, familiar math instead of squinting at handwriting.

**Analogy:** it's like an audio-typist who listens to a recording and types it up — except our "typist" *looks* at handwriting and types **LaTeX**. And crucially, a good typist types **exactly what was said**, even the mistakes. That "exactly what was written" rule is the heart of this project (more in §7).

---

## 2. What is LaTeX, and why do professors love it?

**LaTeX** (say "LAH-tek") is a language for writing documents, especially math. Instead of clicking buttons to make a fraction, you *type instructions* and LaTeX renders beautiful math.

You write:
```
\frac{a+b}{2} = \int_0^1 x^2 \, dx
```
LaTeX renders:  (a+b)/2 = the integral from 0 to 1 of x² — but perfectly typeset, like in a textbook.

**Analogy:** LaTeX is to math what a recipe is to a cake. The recipe (code) is just text; the cake (rendered PDF) is what you get when you "compile" it. Professors think in this recipe language, so giving them LaTeX is speaking their native tongue.

**"Compile"** = turn the LaTeX recipe into the finished PDF. If the recipe has a typo, it "won't compile" — an error instead of a cake. We use "does it compile?" as a basic quality check.

---

## 3. The family of technologies (and how they differ)

These acronyms appear everywhere. Here's each one as a person with a job:

| Term | Plain meaning | Analogy |
|---|---|---|
| **OCR** | Optical Character Recognition — turning a picture of text into text | A person who reads a printed sign and types what it says |
| **HTR** | Handwriting Text Recognition — OCR but for *handwriting* | The same person, but now reading messy handwriting (harder!) |
| **HMER** | Handwritten Math Expression Recognition — reading handwritten *equations* into LaTeX | A specialist who reads *only* handwritten formulas and writes them as LaTeX recipes |
| **VLM** | Vision-Language Model — a big AI that looks at images and talks about them (GPT-4o, Claude, Gemini) | A brilliant but overconfident intern who can look at anything and describe it — but sometimes "fixes" what they see to what they *think* it should be |
| **LaTeX** | The math typesetting language | The recipe language (§2) |

**Why math is harder than normal text:** normal text is a straight line of letters. Math is **2-D** — exponents float up, fractions stack, matrices form grids. Reading it means understanding *position*, not just *symbols*. That's why we need HMER specialists, not just a text reader.

---

## 4. Why is this hard? (The honest version)

Three layers of difficulty, stacked:

1. **Handwriting is ambiguous.** Is that a `z` or a `2`? An `x` or a `×`? A poorly-written `∫` or an `f`? Even humans guess from context.
2. **Math is 2-D and structured.** A tiny position change (`x2` vs `x²`) means something completely different. The machine must get the *layout* right, not just the symbols.
3. **A full page is many problems at once.** A real answer has prose ("Let f be continuous..."), display equations, multi-line proofs, crossed-out work, arrows, and diagrams — all mixed. The machine must first figure out *what each region is* before reading it.

**Analogy:** reading one handwritten equation is like translating one sentence. Reading a full exam page is like translating a messy handwritten letter that also contains diagrams, footnotes, and scribbles in the margin — while never "improving" the author's grammar.

---

## 5. The state of the world (what's solved, what's not)

- **Printed math → LaTeX: basically solved.** Machines read textbook-quality typeset math very well.
- **Handwritten single equation → LaTeX: partly solved.** The best specialist models get it *exactly* right about **62–80% of the time** on clean examples — meaning roughly **1 in 4 is wrong**. On messy real handwriting, worse.
- **Full handwritten page → LaTeX: not solved.** No ready-made tool does this. We have to **build a pipeline** — a chain of steps, each doing one job.

**Analogy:** we can't buy a finished machine. We're assembling an assembly line from parts, and some parts (reading handwritten equations) are themselves still imperfect.

---

## 6. How our "assembly line" (pipeline) works

A page flows through stations, each with one job:

```
Scan of page
   ↓
[A] Clean up      straighten, remove shadows, sharpen  (like photocopying it nicely first)
   ↓
[B] Find regions  "here's prose, here's an equation, here's a diagram"  (like a highlighter marking sections)
   ↓
[C] Reading order "read this box first, then that one"  (like numbering the boxes)
   ↓
[D] Read each box  prose→text reader, equation→math specialist, diagram→just crop the picture
   ↓
[E] Reassemble    glue the pieces into one LaTeX document
   ↓
[F] Check         does it compile? how confident are we per box?
   ↓
[G] Human review  professor/TA sees scan + result side-by-side, fixes anything wrong
   ↓
Clean LaTeX
```

**Why the human at the end?** Because the machine is ~75% right on equations. The human fixes the rest. The goal isn't "no human" — it's "**much faster human**." A professor correcting a mostly-right transcription is far quicker than deciphering raw handwriting.

**Analogy:** the machine is a fast junior assistant who does 75% of the typing; the professor just proofreads and fixes. That still saves enormous time at 200 papers.

---

## 7. The one danger you must understand: **over-correction**

This is the most important idea in the whole project.

The big AI models (VLMs) are *so* good at math that when they see a student's **wrong** step, they often **silently rewrite it into the correct step** — because their internal "math brain" overrides their eyes. Research shows this happens in **42–66% of transcriptions**, and *smarter models do it more*.

**Why this is a disaster for grading:** if a student wrote `2+2=5` and the machine "helpfully" transcribes `2+2=4`, the professor grades a mistake that isn't there — or misses one that is. The student gets the **wrong marks**. That's an unfairness/integrity problem.

**Analogy:** imagine a court transcript typist who "fixes" what witnesses say to sound more sensible. The transcript becomes useless — even dangerous — because it's no longer *what actually happened*.

**Our defense (the "two witnesses" trick):** we run **two different readers** on each equation. If they **disagree**, we flag it for the human. A model that "corrected" an error will differ from a faithful reader — so disagreement catches the lie. This is why the design uses a faithful specialist as the backbone and treats the clever VLM with suspicion.

**The rule:** the system must transcribe **what the student wrote, mistakes and all.** Never "improve" it.

---

## 8. The build vs. buy choices (in plain terms)

To read the equations, we have two realistic options:

- **Buy: Mathpix** — a paid service (~half a cent per page) that's good at handwritten math and tells you how confident it is. Fast to start. But student data leaves our building (privacy question), and it's a running cost.
- **Build/host: Qwen2.5-VL** — a free, open AI model we run on our *own* computers (GPUs). Student data never leaves. But it needs powerful hardware and technical upkeep.

**The deciding question (still open — see DECISIONS.md Q1):** *Are we allowed to send student exam images to an outside company?* If no → we self-host. If yes (with a legal agreement) → Mathpix is the quick path. This one answer changes a lot.

---

## 9. The plan in one breath

1. **Get real IITG handwritten pages** and hand-type ~20 of them perfectly (our "answer key" for testing).
2. **Test the readers** (Mathpix vs self-hosted AI) on those pages — measure how faithful they are and how often they over-correct. *This one test decides if the project is worth doing.*
3. If good enough → **build the thin pipeline** + a review screen.
4. **Pilot with a real professor** on a real (small) batch. Did it save time? Do they trust it?
5. Only then → **scale up** for production.

Each step has a **go/no-go gate** so we never over-build before proving the previous step.

---

## 10. Expanded glossary

| Term | Plain meaning |
|---|---|
| **OCR / HTR / HMER / VLM** | See §3 |
| **LaTeX** | Math typesetting "recipe" language (§2) |
| **Compile** | Turn LaTeX code into the finished PDF |
| **Render** | The visual result of compiling — the "cake" |
| **Pipeline** | A chain of processing steps, each doing one job (§6) |
| **Region / segmentation** | Dividing the page into boxes (prose/equation/diagram) |
| **Layout analysis** | The step that finds the regions |
| **Reading order** | Which region to read first, second, ... |
| **Offline vs online** | Offline = a still image of writing (our case). Online = live pen strokes on a tablet (has timing data; easier, but not what we have) |
| **Isolated expression** | A single cropped equation vs a whole page |
| **Over-correction** | The AI silently "fixing" a student's mistake (§7) — the key danger |
| **Fidelity** | How faithfully the output matches *what was actually written*, mistakes included — our main success measure |
| **ExpRate** | "Expression Recognition Rate" — % of equations the machine got *exactly* right |
| **CER** | "Character Error Rate" — how many small edits to fix the output |
| **BLEU** | A text-similarity score borrowed from translation; imperfect for math |
| **Confidence score** | The model's own "how sure am I?" number, used to flag shaky bits for humans |
| **Human-in-the-loop (HITL)** | A person reviews/corrects the machine's output |
| **Fine-tuning** | Taking a general AI and training it more on *our* kind of data to specialize it |
| **Ground truth / gold set** | The correct answers we compare the machine against |
| **TikZ** | A LaTeX way to *draw* diagrams in code (we're NOT doing this — too hard; we embed diagram photos instead) |
| **Self-hosted** | Running the AI on our own computers so data stays private |
| **VLM hallucination** | When the AI confidently outputs something that isn't really there |

---

**If any other document confuses you, come back here first — then tell me which term or step to expand, and I'll add it.**
