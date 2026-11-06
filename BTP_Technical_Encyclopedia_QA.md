# The B.Tech Project Technical Master Encyclopedia & Viva Defense Guide
## *End-to-End Full-Page Handwritten Mathematical Script Transcription to \LaTeX: A Comparative Study of Modular Hybrid and Multimodal Vision-Language Architectures*

**Course:** MA4998P Project – I (Semester 1 / Phase 1)  
**Department:** Department of Mathematics, Indian Institute of Technology Guwahati  
**Students:** Dinkesh Pal (Roll No. 230123021) & Rithvik Ponnapalli (Roll No. 230123052)  
**Supervisor:** Dr. K. V. Srikanth  
**Evaluation Committee:** Dr. K. V. Srikanth, Dr. P. A. S. Sree Krishna, Dr. Vinay Wagh  

---

# Table of Contents
1. [Part I: Foundational Machine Learning, Computer Vision & NLP Primer (Absolute Zero)](#part-i-foundational-primer)
2. [Part II: Mathematical Typesetting, Document Representation & First Principles](#part-ii-mathematical-typesetting)
3. [Part III: The Institutional Context & The IITG MA102 Examination Dataset](#part-iii-dataset-and-context)
4. [Part IV: The First Research Failure — Pivot 1: Generative Font Normalization](#part-iv-pivot-1-gan)
5. [Part V: Paradigm A — The Modular Dual-Head Hybrid Pipeline](#part-v-paradigm-a-modular)
6. [Part VI: Paradigm B — The Multimodal Vision-Language Frontier (Qwen2.5-VL)](#part-vi-paradigm-b-vlm)
7. [Part VII: Supercomputing Systems Engineering on PARAM Kamrupa](#part-vii-supercomputing)
8. [Part VIII: Quantitative Benchmarks, Failure Case Studies & Qualitative Diagnosis](#part-viii-benchmarks-and-case-studies)
9. [Part IX: Project Staging — Phase 1 Completion vs. Phase 2 Roadmap](#part-ix-phase-staging)
10. [Part X: Master Viva Voce Defense Playbook (Top 25 Anticipated Questions)](#part-x-viva-playbook)

---

<a name="part-i-foundational-primer"></a>
# Part I: Foundational Machine Learning, Computer Vision & NLP Primer (Absolute Zero)

*This section assumes you know nothing about AI or neural networks. Every concept is explained with simple visual analogies first, followed by mathematical definitions.*

---

### Q1: What is a digital image to a computer? What do $H$, $W$, and $C$ mean?
**Simple Analogy:** Think of a mosaic made of colored glass tiles on a grid.  
**Technical Explanation:** To a computer, an image is a 3-dimensional array of numbers (a tensor) with shape $(H, W, C)$:
* $H$ (Height): The number of rows of pixels from top to bottom (e.g., 2338 pixels for an A4 page).
* $W$ (Width): The number of columns of pixels from left to right (e.g., 1653 pixels for an A4 page).
* $C$ (Channels): The depth of color information:
  * For a **grayscale** image, $C = 1$ (each pixel is an integer from $0$ for pure black to $255$ for pure white).
  * For an **RGB color** image, $C = 3$ (Red, Green, Blue intensity values).

### Q2: What is DPI (Dots Per Inch)? Why does it matter for handwritten exams?
**Explanation:** DPI measures spatial scanning density. 
* At **72 DPI** (standard old computer screen), letters are pixelated and thin pencil strokes disappear.
* At **200–300 DPI** (our dataset standard), an A4 sheet ($8.27 \times 11.69$ inches) digitizes into approximately $2338 \times 1653$ pixels. At this resolution, subtle pen details—such as a minus sign ($-$), a decimal point ($.$), or a fraction vinculum—are represented by several continuous pixels rather than disappearing into image noise.

### Q3: What is a CNN (Convolutional Neural Network)? What is a Kernel and a Feature Map?
**Simple Analogy:** Imagine holding a magnifying glass with a specific stencil pattern (like a curve, a cross, or a corner) and sliding it across an entire page. Whenever the magnifying glass passes over ink that matches the stencil, a bell rings loudly.  
**Technical Definition:**
* **Kernel (or Filter):** A small learnable matrix of weights (typically $3 \times 3$ pixels).
* **Convolution Operation:** The kernel slides across the input image, computing the element-wise dot product with the underlying pixel patch:
  $$F(x, y) = \sum_{i=-1}^{1} \sum_{j=-1}^{1} I(x+i, y+j) \cdot K(i, j)$$
* **Feature Map:** The resulting output grid. Early layers detect primitive edges and lines; deeper layers combine edges into character loops, square roots, and brackets.
* **Receptive Field:** The spatial area of the original input image that directly influences a specific neuron in a deep layer.

### Q4: What is DenseNet (Densely Connected Convolutional Network)? Why did we choose it?
**Simple Analogy:** In standard networks (like VGG), Layer 1 passes its output only to Layer 2, Layer 2 only to Layer 3, etc. If early layers make a mistake, later layers cannot fix it. In a DenseNet, **every layer is connected directly to every subsequent layer**.  
**Technical Definition:** In a DenseNet block, layer $\ell$ receives the concatenated feature maps of all preceding layers:
$$\mathbf{x}_\ell = H_\ell([\mathbf{x}_0, \mathbf{x}_1, \dots, \mathbf{x}_{\ell-1}])$$
**Why it matters for Math OCR (HMER):**
1. **Feature Reuse:** Mathematical notation contains both low-level geometric features (thin horizontal lines for minus signs and fraction bars) and high-level conceptual features (complete matrices). DenseNet preserves fine, early-layer pixel strokes all the way to the final classification layer without them being blurred out by multiple downsampling layers.
2. **Gradient Flow:** It eliminates the vanishing gradient problem during backpropagation, allowing stable training from scratch.

### Q5: What is an RNN (Recurrent Neural Network)? What are LSTMs and GRUs?
**Simple Analogy:** When reading a sentence, your brain does not reset its memory after every word. The meaning of the 5th word depends on the first 4 words. An RNN possesses a memory state that carries forward in time.  
**Technical Definition:** An RNN processes sequential data $(x_1, x_2, \dots, x_T)$ step-by-step, updating its internal hidden state $\mathbf{h}_t$:
$$\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b})$$
* **The Vanishing Gradient Problem:** Standard RNNs forget information after 10–15 steps because backpropagated gradients shrink exponentially over time.
* **LSTM (Long Short-Term Memory) & GRU (Gated Recurrent Unit):** Neural memory cells equipped with internal **gates** (forget gate, input gate, update gate) that selectively choose what information to preserve or discard over hundreds of tokens.

### Q6: What is the Attention Mechanism? What are Query, Key, and Value ($Q, K, V$)?
**Simple Analogy:** Imagine searching a library database. 
* **Query ($Q$):** What you are looking for right now (e.g., *"What is the denominator of this fraction?"*).
* **Key ($K$):** The shelf labels on all the books in the library.
* **Value ($V$):** The actual content written inside the books.
* You compare your Query against every Key to calculate relevance scores (attention weights). Then you take a weighted sum of the Values.

**Mathematical Formulation (Scaled Dot-Product Attention):**
$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V$$
1. $Q K^T$: Computes dot-product similarity between every query and every key.
2. $\sqrt{d_k}$: Scaling factor preventing dot-products from exploding in high dimensions.
3. $\text{softmax}$: Converts similarities into positive probabilities that sum to $1.0$ (the attention map $\alpha$).
4. Multiply by $V$: Extracts a context vector focused on the most relevant spatial regions.

### Q7: What is a Transformer? What is the difference between Self-Attention and Cross-Attention?
**Explanation:** 
* A **Transformer** (Vaswani et al., 2017) completely eliminates recurrent loops. It processes all tokens in parallel using stacked multi-head attention layers.
* **Self-Attention:** Tokens attend to *each other within the same sequence*. (e.g., Words in a sentence attending to other words in that same sentence).
* **Cross-Attention:** Tokens in the *decoder* attend to representations in the *encoder*. (e.g., A \LaTeX\ token like `\frac` attending to the visual pixels of a fraction bar in the input image).

### Q8: What is a Vision Transformer (ViT)?
**Explanation:** Transformers were originally built for 1D text strings. To apply them to 2D images, the **Vision Transformer (ViT)** (Dosovitskiy et al., 2020):
1. Takes an image $\mathbf{I} \in \mathbb{R}^{H \times W \times C}$ and cuts it into a grid of non-overlapping square patches (typically $16 \times 16$ pixels).
2. Flattens each patch into a vector and linearly projects it into an embedding dimension $D$.
3. Treats each image patch exactly like a **word token** in NLP!
4. Passes these patch tokens through standard Transformer encoder blocks.

### Q9: What is a GAN (Generative Adversarial Network)? What are U-Net and PatchGAN?
**Simple Analogy:** An art forger trying to paint a fake Rembrandt, and an art detective trying to spot whether the painting is real or fake. They train against each other until the forger's work is indistinguishable from reality.
* **Generator ($G$):** A neural network that generates synthetic images.
* **Discriminator ($D$):** A neural network that outputs a probability ($0$ to $1$) of whether an image is real or fake.
* **U-Net:** A network architecture with an encoder-decoder path connected by **skip connections**. Skip connections pass high-resolution pixel features directly across layers, preserving edges.
* **PatchGAN:** A discriminator that classifies whether localized $70 \times 70$ pixel patches are real or fake, rather than scoring the entire image as a single number.

### Q10: What is a Multimodal Vision-Language Model (VLM)?
**Explanation:** A neural network that natively takes **both image pixels and text prompts** as inputs, and generates text tokens as output. It connects a high-resolution Vision Transformer encoder (which "sees" the image) to a Large Language Model decoder (which "reasons" and writes text) via a cross-modal projection layer. Examples include Qwen2.5-VL, GPT-4o, and Claude 3.5 Sonnet.

---

<a name="part-ii-mathematical-typesetting"></a>
# Part II: Mathematical Typesetting, Document Representation & First Principles

---

### Q11: What is Donald Knuth’s \TeX\ engine? What is the "Boxes and Glue" model?
**Explanation:** In 1978, legendary computer scientist Donald Knuth created \TeX\ because digital typography looked terrible. He formulated typesetting not as drawing pixels, but as physical architecture:
* **Boxes:** Rigid rectangular blocks that have three dimensions: **height** (above baseline), **depth** (below baseline), and **width**. Every character glyph, fraction, and matrix is a box.
* **Glue:** Elastic connectors placed between boxes that have a natural width, an ability to stretch, and an ability to shrink. The engine balances glue to justify lines perfectly.

### Q12: Why is \LaTeX\ an Abstract Syntax Tree (AST) rather than plain text?
**Explanation:** 
* In plain text, reading order is a 1D sequence of characters: $c_1, c_2, c_3 \dots$
* In \LaTeX, mathematical meaning is expressed as a **nested hierarchy (AST)**.
* **Concrete Example:** When you compile:
  ```latex
  \frac{\alpha + \beta}{\det(\mathbf{A})} = \lambda_1
  ```
  The compiler parses it internally as:
  ```
          [Equation]
          /    |    \
     [\frac]  [=]  [\lambda_1]
      /   \
  [Num]   [Den]
   |        |
 (\alpha+\beta) (\det(\mathbf{A}))
  ```
  The parent command `\frac` dictates the physical coordinate placement of two separate child trees: the **numerator box** placed strictly above the vinculum, and the **denominator box** placed strictly below it.

### Q13: What is the Fundamental Dimensionality Mismatch: 1D Text OCR vs. 2D Math HMER?
**Comparison Table:**

| Property | Printed/Handwritten Text OCR (1D) | Handwritten Math HMER (2D) |
| :--- | :--- | :--- |
| **Spatial Reading Order** | Strictly linear left-to-right. | Non-linear 2D planar graph; tree hierarchies. |
| **Baseline Dependency** | Characters sit on a shared baseline. | **Displacement changes meaning:** $x^2$ (superscript) vs. $x_2$ (subscript). |
| **Scope Delimiters** | Periods, commas, quotation marks. | Multi-line bounding brackets ($[ \dots ]$), square roots ($\sqrt{\dots}$), vinculums. |
| **State of the Art** | Solved ($>99\%$ accuracy on printed text). | **Open research problem** ($\approx 50$--$65\%$ on full exam pages). |

### Q14: What is a Spatial Relationship Tree (SRT)?
**Explanation:** In mathematical expression recognition, an SRT is a formal tree structure where nodes represent detected mathematical symbols and edges represent spatial grammatical relationships:
* `Above` / `Below` (for fraction bars, limits, summations)
* `Superscript` (exponent placed top-right)
* `Subscript` (index placed bottom-right)
* `Inside` (radicands inside a square root)
* `Matrix Cell` (row $i$, column $j$)

### Q15: What is the difference between OCR, HTR, and HMER?
* **OCR (Optical Character Recognition):** Converting images of **machine-printed text** (books, newspapers) into digital text.
* **HTR (Handwritten Text Recognition):** Converting images of **handwritten linear natural language prose** (letters, notes) into text strings.
* **HMER (Handwritten Mathematical Expression Recognition):** Converting images of **handwritten mathematics** (formulas, matrices, proofs) into structural markup like \LaTeX\ or MathML.

### Q16: What is the "Expression-to-Page Generalization Gap"? Why do CROHME models fail on real exams?
* **CROHME (Competition on Recognition of Online Handwritten Mathematical Expressions):** The global research benchmark for HMER.
* **The Fatal Generalization Gap:**
  1. **Isolated Crops vs. Full Pages:** CROHME consists *only* of single, isolated formulas ($\frac{a}{b} + c = d$) tightly cropped on white backgrounds.
  2. **No English Prose:** In real exams, students write continuous cursive English explanations (*"Let $W$ be a subspace..."*) mixed with equations. Models trained on CROHME have **zero English vocabulary**—they attempt to parse English words as broken math symbols!
  3. **Resolution Collapse:** Downsampling an entire A4 exam page ($2338 \times 1653$ pixels) to CROHME input size ($256 \times 512$ pixels) shrinks pen strokes to sub-pixel noise, destroying all information.

---

<a name="part-iii-dataset-and-context"></a>
# Part III: The Institutional Context & The IITG MA102 Examination Dataset

---

### Q17: What are the exact demographics of the MA102 Dataset?
* **Target Course:** MA102 (Mathematics-II: Linear Algebra & Multivariable Calculus).
* **Institution:** Department of Mathematics, Indian Institute of Technology Guwahati.
* **Student Cohort:** **939 enrolled candidates**.
* **Physical Booklet:** Standard unruled 14-page physical answer booklet per candidate.
* **Total Volume:** **13,146 scanned pages** from a single complete examination cycle.
* **Scanning Specifications:** Digitized on flatbed optical scanners at the Computer & Communication Centre (CCC), IIT Guwahati, at 200–300 DPI, yielding $2338 \times 1653$ pixel images in 24-bit sRGB and 8-bit grayscale.

### Q18: What is the Header ($y \le 0.20H$) vs. Canvas ($y > 0.20H$) geometric split?
* **Printed Header ($y \le 0.20H$):** The top $15\%$ to $20\%$ of the scanned page contains institutional typography: course code (`MA102`), student metadata (Name, Roll No.), question label (*"Space for answering Question-X only"*), and horizontal dividing rules.
* **Student Canvas ($y > 0.20H$):** The remaining $80\%$ of the page is unruled white bond paper where students write without guidelines, producing baseline drift, margin annotations, and scratch-outs.

### Q19: What mathematical structures appear in MA102?
1. **$4 \times 4$ Real Decimal Matrices:** Matrices containing real numbers, decimals, and negative signs (e.g., student script `page-210103096-01`):
   $$\mathbf{A} = \begin{bmatrix}
   -1.0 & 8.5 & -2.0 & 0.5 \\
   4.5 & 2.5 & -3.0 & 3.0 \\
   -0.5 & -1.5 & 3.0 & 1.0 \\
   0.0 & 2.0 & -1.0 & 4.0
   \end{bmatrix} \in M_{4 \times 4}(\mathbb{R})$$
2. **Elementary Row Operations (EROs):** Transition operators written between matrices:
   $$R_2 \to R_2 + 4.5 R_1, \qquad R_3 \to R_3 - \frac{1}{2} R_1$$
3. **Deductive Vector Subspace Proofs:** Step-by-step verification of zero vector ($\mathbf{0} \in S$), additive closure ($\mathbf{u}+\mathbf{v} \in S$), and scalar multiplication closure ($c\mathbf{u} \in S$).
4. **Multivariable Calculus Operators:** Gradients ($\nabla f$), double integrals ($\iint_D f(x, y) \, dx dy$), kernels ($\ker(T)$), and rank-nullity calculations.

### Q20: What is "Ascender-Descender Interlocking"? Why does it break simple bounding boxes?
* **Ascenders:** Parts of letters that extend upwards (e.g., $d, t, f, \lambda, \int$).
* **Descenders:** Parts of letters that hang downwards (e.g., $y, g, p, q$).
* **The Interlocking Phenomenon:** On unruled paper, the descenders of line 1 hang directly into the horizontal band where ascenders of line 2 reach up. Their vertical bounding boxes overlap by **$30\%$ to $45\%$** without their ink strokes touching.
* **Why it breaks simple bounding boxes:** A horizontal projection cut slices directly through the descenders and ascenders, breaking letters in half.

### Q21: What is "Handwriting Style Leakage"? How does Student-Disjoint Partitioning prevent it?
* **The Flaw (Style Leakage):** If you randomly shuffle cropped equation images across train and test sets, crops written by the *same student* appear in both sets. The neural network simply memorizes that individual student's cursive slant and pen pressure, producing unrealistically high, fake benchmark scores.
* **Our Solution (Student-Disjoint Partitioning):** We split the dataset strictly by **Student Roll Number**:
  $$\mathcal{S}_{\text{total}} = \mathcal{S}_{\text{train}} \cup \mathcal{S}_{\text{val}} \cup \mathcal{S}_{\text{test}}, \quad \text{where } \mathcal{S}_i \cap \mathcal{S}_j = \emptyset$$
  * **Train Set:** 750 students ($\approx 10,500$ pages)
  * **Validation Set:** 89 students ($\approx 1,246$ pages)
  * **Held-Out Test Set:** 100 students ($\approx 1,400$ pages)
  The test set contains handwriting from students the model has never encountered.

### Q22: What is the "Verbatim Ink Protocol"?
* When transcribing gold-standard benchmark pages (e.g., Roll 210103096 and Roll 240104105), human annotators transcribing into \LaTeX\ must transcribe **100% literal ink with zero error correction**.
* If a student wrote an arithmetic error (e.g., $(-2) \times 3 = 6$ or $-2.0$ as $+2.0$), the ground truth records the mistake verbatim. We **never** alter reference \LaTeX\ to match an instructor's answer key.

---

<a name="part-iv-pivot-1-gan"></a>
# Part IV: The First Research Failure — Pivot 1: Generative Font Normalization

---

### Q23: What was our initial hypothesis in July 2026?
* **The Thought Process:** Recognizing *printed* textbook math (Computer Modern font) is already solved with high accuracy ($>95\%$) using open-source tools like `pix2tex`.
* **The Idea:** Build a **Two-Stage GAN Pipeline**:
  * **Stage 1 (Handwriting $\to$ Printed Font Image):** Train a GAN to transform messy handwriting crops into synthetic, clean Computer Modern printed images.
  * **Stage 2 (Printed Font Image $\to$ \LaTeX):** Run standard printed OCR on the synthetic image.

```
[Handwritten Crop] ──► [ U-Net Generator + PatchGAN ] ──► [ Synthetic "Printed" Image ] ──► [ pix2tex ] ──► LaTeX Code
```

### Q24: What was the composite loss function used to train the GAN?
$$\mathcal{L}_G = \lambda_1 \|\mathbf{I}_{\text{font}} - \hat{\mathbf{I}}_{\text{font}}\|_1 + \lambda_{\text{perc}} \mathcal{L}_{\text{perceptual}} + \lambda_{\text{adv}} \mathcal{L}_{\text{adv}}$$
* $\|\dots\|_1$: Mean Absolute Error across pixels.
* $\mathcal{L}_{\text{perceptual}}$: Feature-matching loss comparing deep feature representations from a pretrained VGG network.
* $\mathcal{L}_{\text{adv}}$: Adversarial loss from the PatchGAN discriminator trying to differentiate real printed math from generator output.

### Q25: Why did the GAN Font Normalization approach fail catastrophically?
When evaluated on real student handwriting, three fatal failure modes emerged:
1. **Perceptual Realism $\neq$ Semantic Correctness:** A GAN is trained to fool a discriminator into believing an image looks like a printed font. It has no mathematical understanding. To make handwriting look like serif typography, it hallucinated phantom serifs: turning cursive `$l$` into the number `$1$`, and curling a handwritten `$v$` into an `$r$`.
2. **Erasure of Minus Signs and Exponents:** A minus sign ($-$) or decimal point ($.$) occupies less than $0.1\%$ of the pixels in an equation box. In an $\mathcal{L}_1$ pixel loss, omitting a tiny dot incurs almost zero penalty compared to misdrawing a large capital letter $M$. The GAN routinely smoothed over or completely erased minus signs and exponents ($x^2 \to x$).
3. **Merging Matrix Columns:** Slanted cursive handwriting caused the GAN to blur adjacent matrix column numbers into unsegmentable ink blobs.

### Q26: What was the Pivot 1 Decision?
In August 2026, we made our **First Strategic Pivot (Pivot 1)**:  
**Completely abandon intermediate image-to-image synthesis.** Generating an intermediate bitmap creates an unnecessary information bottleneck where subtle symbolic distinctions are lost to generative hallucination. All subsequent architectures must map continuous visual feature representations directly into discrete symbolic \LaTeX\ tokens using end-to-end differentiable sequence modeling.

---

<a name="part-v-paradigm-a-modular"></a>
# Part V: Paradigm A — The Modular Dual-Head Hybrid Pipeline

---

### Q27: What is the high-level philosophy of Paradigm A?
**Philosophy:** Divide-and-conquer. Deconstruct full-page transcription into 5 transparent, mathematically governed, decoupled stages:
$$\mathbf{I}_{\text{raw}} \xrightarrow{\text{Stage 1}} \mathbf{I}_{\text{deskew}} \xrightarrow{\text{Stage 2}} \{B_1, \dots, B_K\} \xrightarrow{\text{Stage 3}} \{\text{Type}_k\} \xrightarrow{\text{Stage 4}} \{\hat{\mathbf{y}}_k\} \xrightarrow{\text{Stage 5}} \mathbf{Y}_{\text{doc}}$$

### Q28: How does Projection Variance Deskewing work?
* To straighten a tilted page without brittle Hough line transforms, we rotate the image across candidate angles $\theta \in [-15^\circ, +15^\circ]$ and evaluate the horizontal projection histogram $H_\theta(y)$.
* **Mathematical Invariant:** When lines are tilted, ink is smeared across vertical coordinates. When the page is rotated to the true baseline angle $\theta^*$, horizontal text lines align perfectly, creating tall peaks separated by empty white-space valleys. This **maximizes projection variance**:
  $$\theta^* = \arg\max_{\theta \in [-15^\circ, +15^\circ]} \text{Var}\left( H_\theta(y) \right)$$
* The image is rotated by $-\theta^*$ using bicubic interpolation.

### Q29: What is Sauvola Adaptive Local Binarization? How does it differ from Otsu?
* **Otsu's Thresholding:** Computes a single global threshold for the whole page. It fails when there are scanner shadows or faint ballpoint pens.
* **Sauvola Binarization (Sauvola & Pietikainen, 2000):** Computes an independent threshold $T(x, y)$ for each pixel using a local $W \times W$ window ($31 \times 31$ pixels):
  $$T(x, y) = m(x, y) \cdot \left[ 1 + k \cdot \left( \frac{s(x, y)}{R} - 1 \right) \right]$$
  * $m(x, y)$: Mean grayscale intensity in the local window.
  * $s(x, y)$: Standard deviation of pixel intensities in the local window.
  * $R = 128$: Maximum standard deviation for 8-bit images.
  * $k = 0.20$: Control parameter determining how sensitive the threshold is to faint pen strokes.
* If a pixel is darker than $T(x, y)$, it is classified as student ink.

### Q30: How does 1D Vertical IoU Line Segmentation work?
* Let candidate line strips be projected onto the vertical $y$-axis as intervals $J_a = [y_{a1}, y_{a2}]$ and $J_b = [y_{b1}, y_{b2}]$.
* Compute their **1D Vertical Intersection-over-Union**:
  $$\text{IoU}_{1D}(J_a, J_b) = \frac{\max\left(0, \ \min(y_{a2}, y_{b2}) - \max(y_{a1}, y_{b1})\right)}{\max(y_{a2}, y_{b2}) - \min(y_{a1}, y_{b1})}$$
* If $\text{IoU}_{1D}(J_a, J_b) \ge \theta_{\text{IoU}} = 0.35$, the two regions interlock (e.g., descenders of line 1 entering ascenders of line 2). They are merged into a single line container, preventing sliced characters.

### Q31: How does the Algorithmic Matrix Parser (`MatrixGridParser`) work?
1. **Bracket Detection:** Uses connected-component analysis to search for tall, slender vertical strokes that have an aspect ratio $h/w > 3.0$ and span more than two standard line heights.
2. **Container Tagging:** When matching left and right brackets are identified ($[ \dots ]$), the entire 2D area between them is tagged as a **Matrix Container** *before* standard line segmentation cuts the page.
3. **Grid Extraction:** Horizontal and vertical projection cuts inside the container locate rows and columns.
4. **Assembly:** Wraps individual cell recognitions into a \LaTeX\ `\begin{bmatrix} ... \end{bmatrix}` environment, using `&` for column separation and `\\` for row line breaks.

### Q32: What is the Dual-Head Recognition Dispatcher?
Exam scripts contain both cursive English sentences and mathematical formulas. Sending text to math models produces crashes; sending math to text models produces garbage.
* **Classifier:** Evaluates stroke density, aspect ratio, and symbol heuristics.
* **Text Head (Microsoft TrOCR):** Bounding boxes classified as prose proofs are dispatched to `microsoft/trocr-base-handwritten`.
* **Math Head (Custom DenseNet-Coverage HMER):** Bounding boxes classified as mathematical formulas are dispatched to our custom 23.1M DenseNet model.

### Q33: How does the 2D Spatial Reading-Order DAG work?
* Full-page layouts are modeled as a **Directed Acyclic Graph (DAG)** $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \prec)$ where each vertex is a recognized bounding box.
* Ordering relation:
  $$u \prec v \iff \begin{cases}
  y_{u, \text{bottom}} < y_{v, \text{top}} & \text{(Block $u$ is physically above block $v$)} \\
  x_{u, \text{right}} < x_{v, \text{left}} \quad \text{when vertical overlap} > 50\% & \text{(Block $u$ is to the left of block $v$)}
  \end{cases}$$
* A standard **Topological Sort** reconstructs the reading sequence into `output_solution.tex`.

---

<a name="part-vi-paradigm-b-vlm"></a>
# Part VI: Paradigm B — The Multimodal Vision-Language Frontier (Qwen2.5-VL)

---

### Q34: What is Qwen2.5-VL-7B-Instruct?
* A state-of-the-art native Multimodal Vision-Language Model developed by Alibaba Cloud, containing approximately **$7.61 \times 10^9$ parameters**.
* Processes full-page images end-to-end without requiring external binarization, deskewing, or line segmentation modules.

### Q35: What is Dynamic Resolution ViT? Why is square resizing bad for math?
* **The Problem:** Earlier Vision Transformers resize all images to fixed square dimensions (e.g., $448 \times 448$). An A4 exam scan is tall and narrow ($2338 \times 1653$). Resizing it to a square squishes the vertical axis, crushing fractions, exponents, and minus signs into single blurry pixel lines.
* **Dynamic Resolution:** Qwen2.5-VL splits the image into visual patch tokens dynamically according to its **native aspect ratio**, ensuring that character proportions remain undistorted.

### Q36: What is 2D-RoPE (2D Rotary Position Embeddings)?
* Standard language models use 1D Rotary Position Embeddings (RoPE) to encode word order along a 1D sequence.
* **2D-RoPE (Su et al., 2024):** Splits attention query and key channels into two orthogonal halves:
  * Channels $1 \dots D/2$: Rotated by horizontal spatial coordinate $x$.
  * Channels $D/2+1 \dots D$: Rotated by vertical spatial coordinate $y$.
* **The Mathematical Effect:** The inner product between two tokens depends directly on their **2D Euclidean displacement $(\Delta x, \Delta y)$**. Self-attention natively perceives whether two numbers belong to the same matrix row (same $y$) or the same matrix column (same $x$).

### Q37: What is Conversational Mode Collapse?
* Off-the-shelf instruction-tuned VLMs are trained via RLHF to act as conversational assistants.
* When presented with an exam scan, instead of generating raw \LaTeX\ code, the model generates friendly dialogue:
  ```text
  "Certainly! Below is the transcription of the linear algebra exam script shown in the image..."
  ```
  Or it interprets math prompts as coding homework, generating executable Python scripts:
  ```python
  import numpy as np
  A = np.array([[-1.0, 8.5], [4.5, 2.5]])
  ```
* Once conversational text begins, attention decouples from image pixels and the output is uncompilable.

### Q38: What is Assistant Prefill Locking (`continue_final_message=True`)?
* **Our Solution to Conversational Collapse:**
1. We prepopulate the assistant's context buffer with the formal \LaTeX\ document preamble before generation begins:
   $$\mathbf{Y}_{\text{prefill}} = \big[ \texttt{\textbackslash documentclass[12pt]\{article\}}, \dots, \texttt{\textbackslash begin\{document\}}, \dots, \texttt{\textbackslash section*\{Solution\}\textbackslash n\textbackslash n} \big]$$
2. In Hugging Face Transformers, we set `continue_final_message=True`.
3. The model is **not allowed** to generate an assistant start token or say *"Certainly!"*.
4. Its very first predicted token must be the immediate continuation after `\section*{Solution}\n\n`.
5. The probability of conversational chatter drops to $0.0\%$, forcing the model directly into literal \LaTeX\ transcription.

### Q39: What is the "NLP Repetition Penalty Paradox in Mathematical Transcription"?
* **Standard NLP Practice:** Apply a high repetition penalty ($\rho > 1.2$) or strict n-gram blocking (`no_repeat_ngram_size = 15`) to prevent chatbots from repeating themselves.
* **The Paradox in Math:** In mathematical derivations, **repetition across consecutive lines is authentic**:
  1. *Consecutive Coordinate Equations:* In subspace proofs, students verify equations for each coordinate:
     $$\sum_{j=1}^4 a_{1j} = 0, \quad \sum_{j=1}^4 a_{2j} = 0, \quad \sum_{j=1}^4 a_{3j} = 0, \quad \sum_{j=1}^4 a_{4j} = 0$$
  2. *Sequential Row Operations:* Students write row reductions with identical syntax:
     $$R_2 \to R_2 - 2R_1, \qquad R_3 \to R_3 - 2R_1$$
* **The Consequence:** When standard n-gram blocking was active, the model generated the first line, but **artificially blocked subsequent valid equations** because their syntax matched the preceding line! This caused the omission of 4 coordinate equations on Page 1 of Roll 210103096.

### Q40: How did we resolve the Repetition Penalty Paradox?
1. Lowered global token repetition penalty to a conservative threshold ($\rho = 1.05$--$1.10$) and **completely disabled multi-line n-gram blocking**, allowing valid repetitive equations to generate freely.
2. Built a **Targeted Algorithmic Matrix Regex Collapser**: A deterministic post-processor that operates strictly inside matrix environments (`\begin{bmatrix} ... \end{bmatrix}`). If consecutive repeating zeros in any row exceed the column rank ($N > 6$), the script trims the row back to the true matrix dimension.

### Q41: What is the Pushdown Automaton Stack Balancer? How does it guarantee 100% compilation?
* **The Problem:** Autoregressive generation has a maximum token limit ($T_{\text{max}} = 2560$). If generation truncates midway through an environment (e.g., inside `\begin{bmatrix}`), running `pdflatex` produces a fatal syntax crash.
* **The Pushdown Automaton (in `src/latex/generator.py`):**
  1. Maintains an environment stack $S = []$.
  2. For every `\begin{env}` encountered $\implies$ `Push(S, env)`.
  3. For every `\end{env}` encountered $\implies$ if $\text{Top}(S) == \text{env}$, `Pop(S)`.
  4. When generation stops: while $S \neq \emptyset$, pop each remaining open environment and append its closing tag:
     $$\hat{\mathbf{Y}} \leftarrow \hat{\mathbf{Y}} \ \oplus \ \texttt{\textbackslash end\{} e \texttt{\}\textbackslash n}$$
  5. If `\end{document}` is absent, append it.
* Guarantees every generated file is syntactically closed, achieving a **$100.0\%$ Compilation Pass Rate**.

---

<a name="part-vii-supercomputing"></a>
# Part VII: Supercomputing Systems Engineering on PARAM Kamrupa

---

### Q42: What is the hardware topology of PARAM Kamrupa?
* **Platform:** BullSequana X1000 Blade Platform, deployed by C-DAC and IIT Guwahati under the National Supercomputing Mission.
* **GPU Node Accelerators:** **NVIDIA Tesla V100-SXM2 (Volta architecture, Compute Capability `sm_70`)**.
* **GPU Memory:** **16 GB HBM2** per GPU (900 GB/s bandwidth).
* **Host CPU & RAM:** Dual-socket Intel Xeon Gold 6248 (40 Cores / 80 Threads), 192–384 GB DDR4 ECC RAM.
* **Storage:** High-Performance Parallel **Lustre Filesystem** (`/scratch/p.dinkesh/hmer_project/`).
* **Interconnect:** Mellanox InfiniBand EDR (100 Gbps).
* **OS:** CentOS 7.9 / Red Hat Enterprise Linux 7.9.
* **Orchestrator:** SLURM (Simple Linux Utility for Resource Management).

### Q43: How did we handle Air-Gapped execution without internet access?
* Compute nodes have **zero outbound internet access** (cannot connect to Hugging Face or PyPI).
* Personal home directories have a strict 20 GB quota.
* **Our Workflow (`download_vlm_weights.py`):**
  1. Login nodes (`login01`--`login04`) have internet proxy access. We executed snapshot downloads on login nodes using `huggingface_hub.snapshot_download`.
  2. Serialized raw weight shards, configs, and tokenizers directly into Lustre scratch (`/scratch/p.dinkesh/hmer_project/models/qwen_vl`).
  3. Created an isolated virtual environment on Lustre scratch (`/scratch/p.dinkesh/hmer_project/venv/`).
  4. Compute nodes load weights locally across InfiniBand at over **$2.5\,\text{GB/s}$**, completely offline.

### Q44: What was the CentOS 7 CXXABI dynamic linking bug? How did we solve it without root access?
* **The Bug:** CentOS 7 ships with GCC 4.8.5. Its C++ library (`/lib64/libstdc++.so.6`) only exports symbols up to `GLIBCXX_3.4.19` and `CXXABI_1.3.7`. Modern PyTorch 2.0+ and `torchvision.ops` require **`CXXABI_1.3.9`** and **`GLIBCXX_3.4.21`** (introduced in GCC 5.1+). Importing `torchvision` threw a fatal runtime error:
  ```text
  ImportError: /lib64/libstdc++.so.6: version `CXXABI_1.3.9' not found (required by .../torchvision/image.so)
  ```
* **Our Fix:** We located a modern `libstdc++.so.6` inside the institutional Anaconda module at `/opt/ohpc/pub/apps/Anaconda3-2023.03/lib/libstdc++.so.6`.
* We added a dynamic linking hook using Python's `ctypes` library at the entry point of our scripts:
  $$\text{ctypes.CDLL}\left( \texttt{"/opt/ohpc/pub/apps/Anaconda3-2023.03/lib/libstdc++.so.6"}, \ \text{mode} = \text{ctypes.RTLD\_GLOBAL} \right)$$
* By preloading the modern library into the global symbol table (`RTLD_GLOBAL`) before importing PyTorch, all subsequent requests for `CXXABI_1.3.9` resolved against the modern Anaconda runtime, fixing the crash without needing root access.

### Q45: Why does FlashAttention-2 fail on NVIDIA Tesla V100? What was the fallback?
* **Why FlashAttention Fails:** FlashAttention-2 relies on hardware-specific asynchronous copy instructions (`cp.async`) physically present only on NVIDIA Ampere (`sm_80`), Ada (`sm_89`), and Hopper (`sm_90`) architectures. The Tesla V100 is Volta (`sm_70`), which lacks `cp.async`. Compiling FlashAttention on Volta triggers immediate kernel failure.
* **The Fallback:** We configured PyTorch's native Scaled Dot-Product Attention (SDPA) to use **Cutlass Memory-Efficient Attention**:
  ```python
  torch.backends.cuda.enable_flash_sdp(False)
  torch.backends.cuda.enable_mem_efficient_sdp(True)
  torch.backends.cuda.enable_math_sdp(True)
  ```
  Cutlass computes exact attention without constructing the $N \times N$ attention matrix in GPU memory, bounding memory to $\mathcal{O}(N)$ on Volta hardware.

### Q46: How did we budget 16GB VRAM for a 7.6B parameter model?
* Qwen2.5-VL-7B has **$7.61 \times 10^9$ parameters**.
* In 16-bit half-precision (FP16), storing model weights consumes **$13.67\,\text{GiB}$**.
* Hardware limit is **$16.00\,\text{GiB}$** on Tesla V100.
* Remaining headroom: $16.00 - 13.67 = 2.33\,\text{GiB}$ ($\approx 1.83\,\text{GiB}$ free after PyTorch CUDA context allocation).
* **Dynamic Resolution Capping:** Passing a full $2338 \times 1653$ image generates $>2800$ visual tokens, instantly causing a CUDA Out-of-Memory (OOM) crash.
* We capped the maximum image dimension to **$1200\,\text{px}$** ($N_{\text{tokens}} \le 1280$, typically $\approx 1016$ tokens).
* This stabilized peak VRAM at **$15.12\,\text{GB}$**, leaving a safe $900\,\text{MB}$ safety buffer that allowed autoregressive generation to finish cleanly.

---

<a name="part-viii-benchmarks-and-case-studies"></a>
# Part VIII: Quantitative Benchmarks, Failure Case Studies & Qualitative Diagnosis

---

### Q47: What are the exact evaluation metrics used in the report?
1. **Exact Match Rate (EMR / $\text{ExpRate}_0$):** Percentage of expressions transcribed with zero token errors:
   $$\text{EMR} = \text{ExpRate}_0 = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left( \hat{\mathbf{Y}}^{(i)} = \mathbf{Y}_{\text{ref}}^{(i)} \right)$$
   $\text{ExpRate}_1$ and $\text{ExpRate}_2$ allow $\le 1$ and $\le 2$ token errors.
2. **Token Error Rate (TER):** Normalized Levenshtein distance computed over discrete \LaTeX\ tokens:
   $$\text{TER} = \frac{\sum_{i=1}^N \mathcal{D}_{\text{Lev}}\left( \hat{\mathbf{Y}}^{(i)}, \mathbf{Y}_{\text{ref}}^{(i)} \right)}{\sum_{i=1}^N |\mathbf{Y}_{\text{ref}}^{(i)}|} \times 100\%$$
3. **Normalized Edit Distance (NED):** Bounded similarity score within $[0, 1]$:
   $$\text{NED}\left( \hat{\mathbf{Y}}, \mathbf{Y}_{\text{ref}} \right) = 1 - \frac{\mathcal{D}_{\text{Lev}}\left( \hat{\mathbf{Y}}, \mathbf{Y}_{\text{ref}} \right)}{\max\left( |\hat{\mathbf{Y}}|, |\mathbf{Y}_{\text{ref}}| \right)}$$

### Q48: What are the performance numbers of our custom 23.1M DenseNet HMER model?
* $\text{ExpRate}_0$ (Exact Match): **$75.0\%$** (target $\ge 70\%$)
* $\text{ExpRate}_1$: **$100.0\%$** (target $\ge 85\%$)
* $\text{ExpRate}_2$: **$100.0\%$** (target $\ge 90\%$)
* Token Error Rate (TER): **$2.13\%$** (target $\le 5\%$)
* Numerical Stability: **0 NaNs** across all training epochs.

### Q49: What are the full-page benchmark comparison results?
* **Tesseract 5.0 (Classical OCR):** TER = $88.4\%$, NED = $0.12$, Compilable = $0.0\%$. (Slices matrices; discards fractions).
* **Pivot 1 (GAN Font Normalization):** TER $> 85.0\%$, NED = $0.18$, Compilable = $0.0\%$. (Erased minus signs and exponents).
* **Meta Nougat (`nougat-small`):** TER = $52.1\%$, NED = $0.48$, Compilable = $33.3\%$. (Repetition loops on cursive ink).
* **Pivot 3 (Modular Hybrid Pipeline):** TER = $38.5\%$, NED = $0.65$, Compilable = **$100.0\%$**. (TrOCR historical vocabulary drift).
* **Naive Qwen2-VL-2B (Zero-Shot):** TER = $71.3\%$, NED = $0.29$, Compilable = $0.0\%$. (Conversational mode collapse).
* **Pivot 4 (Qwen2.5-VL-7B with Prefill Lock):** **TER = $28.4\%$**, **NED = $0.72$**, **Compilable = $100.0\%$**. (Best zero-shot full-page baseline).

### Q50: What happened in Case Study 1 (TrOCR Prose Drift)?
* When our Modular Pipeline dispatched cursive English proofs to Microsoft TrOCR, it generated:
  ```text
  a member of the House of Representatives from the Housewives of Representatives from the House . 
  LFOGET L GO J. COUNTY I Celebrity matrix ... Chamberlain v.
  ```
* **Why:** TrOCR was pretrained on 18th/19th century historical legal manuscripts (the IAM dataset). Faced with rapid cursive handwriting on mathematics proofs, its visual confidence dropped, and the language decoder defaulted to its historical training prior!

### Q51: What happened in Case Study 2 (Formula OCR Stroke Collapse)?
* When math lines were dispatched to `pix2tex`, it output uncompilable noise:
  ```latex
  \vert\Im\rangle\ \ \left(\frac{\gamma_{\mathrm{\tiny~\lost,~}}}{\cdot} ... \right)\stackrel{\sim}{\bf b}_{\infty}
  ```
* **Why:** Formula recognizers are trained exclusively on clean, printed vector \LaTeX\ equations from arXiv. Wavy handwritten strokes caused attention to scatter across arbitrary math glyphs.

### Q52: What happened in Case Study 3 (VLM Semantic Solving Hazard)?
* When Qwen2.5-VL ran without negative constraints, it output:
  ```text
  A) To show S is a subspace of M_n(R), we need to verify that it satisfies the 
  following conditions: 1. The zero matrix must be in S. Let's check each condition... 
  Therefore, S is not a subspace of M_n(R).
  ```
* **Why:** The model tried to **solve** the math as an AI tutor rather than transcribing the ink verbatim. This directly motivated our strict negative prompts.

### Q53: What were the findings of the in-depth diagnosis on Roll 210103096?
* **Page 01 ($\approx 50\%$ Overall Accuracy):** $100\%$ accuracy on the large $4 \times 4$ decimal matrix, but skipped 4 consecutive coordinate equations due to the repetition penalty paradox and $1200\,\text{px}$ downscaling.
* **Page 02 (Skipped Operation):** Skipped an explicit row operation $R_1 \to -R_1$ and hallucinated intermediate matrix steps.
* **Pages 03 & 04 (Lecture-Mode Divergence):** On long cursive theoretical proofs, the zero-shot model entered "lecture mode", writing textbook explanations instead of copying handwritten ink.
* **Core Conclusion:** Zero-shot inference represents the performance ceiling ($\approx 50$--$60\%$ fidelity) of off-the-shelf foundation models. Supervised domain fine-tuning (LoRA) is strictly required to reach production grading accuracy.

---

<a name="part-ix-phase-staging"></a>
# Part IX: Project Staging — Phase 1 Completion vs. Phase 2 Roadmap

---

### Q54: What did we accomplish in Phase 1 (Semester 1 / MA4998P)?
This report marks the completion of Phase 1 at a **$95\%$ completion milestone**:
1. Curated the MA102 dataset (13,146 scanned pages across 939 students).
2. Engineered the Modular Dual-Head Hybrid Pipeline (Sauvola, 1D IoU NMS, `MatrixGridParser`, 2D DAG).
3. Trained a custom 23.1M DenseNet-Coverage HMER model from scratch on PARAM Kamrupa ($75.0\%$ ExpRate0, $2.13\%$ TER, 0 NaNs).
4. Deployed Qwen2.5-VL-7B on Tesla V100 GPUs in air-gapped PARAM Kamrupa, fixing CentOS 7 library bugs via `ctypes` preloading and configuring Cutlass SDPA.
5. Budgeted 16GB VRAM, stabilizing peak memory at $15.12\,\text{GB}$ via dynamic $1200\,\text{px}$ bounding.
6. Engineered Assistant Prefill Locking with a pushdown stack balancer, achieving a **$100.0\%$ \LaTeX\ Compilation Pass Rate**.
7. Diagnosed baseline failure modes (TrOCR prose drift, repetition penalty paradox, VLM problem-solver prior).

### Q55: What is the Phase 2 Engineering Roadmap (Semester 2 / MA4999P)?
1. **Supervised Domain Fine-Tuning via LoRA:** Fine-tune Qwen2.5-VL's vision encoder directly on handwriting crops from the 13,146 MA102 pages to suppress the autonomous "problem-solver" prior.
2. **INT4 / FP8 Quantization for Native 2000px Resolution:** Quantize the 7.6B model to 4-bit weights ($\approx 4.5\,\text{GiB}$), liberating over $11\,\text{GiB}$ of VRAM so we can process full $2000\,\text{px}+$ images without CUDA OOM crashes.
3. **2D Spatial Layout Guidance:** Inject explicit bounding-box coordinates into prompts to ensure the cross-attention decoder reads every spatial line in topological order.
4. **Faculty Grading Interface:** Finalize a lightweight web interface displaying typeset \LaTeX\ side-by-side with original scanned ink for instructor evaluation.

---

<a name="part-x-viva-playbook"></a>
# Part X: Master Viva Voce Defense Playbook (Top 25 Questions & Model Answers)

*Practice these 25 rapid-fire questions to ace your presentation.*

1. **Q: Why didn't you build an automated grading system?**  
   **A:** Grading introduces severe risks of hallucination and unfairness; in university exams, grading authority must remain strictly with human faculty. Our assistive transcription tool eliminates eye fatigue by providing clean typeset \LaTeX\ side-by-side with original ink, keeping instructors in control.

2. **Q: Why can't we use commercial cloud APIs like GPT-4o or Mathpix?**  
   **A:** Answer booklets are confidential university records; uploading unencrypted scans violates institutional privacy policies. Furthermore, commercial LLMs silently rewrite student errors in $42\%$ to $66\%$ of cases (the PINK metric, arXiv:2604.22774), which would award unearned marks to wrong derivations.

3. **Q: What is Donald Knuth's "Boxes and Glue" model?**  
   **A:** Knuth modeled typesetting through physical boxes (rigid rectangular blocks with height, depth, width) and glue (elastic spacing that stretches and shrinks to balance lines).

4. **Q: Why is \LaTeX\ an Abstract Syntax Tree (AST)?**  
   **A:** Unlike 1D plain text, \LaTeX\ is governed by a context-free grammar where commands like `\frac` create hierarchical parent-child subtrees for numerators and denominators above and below the vinculum.

5. **Q: Why does classical OCR (Tesseract) fail on mathematics?**  
   **A:** Tesseract assumes 1D linear horizontal text streams. When it encounters 2D mathematical notation, it slices through tall matrix brackets, confuses fraction bars with underline noise, and scrambles superscripts into adjacent lines.

6. **Q: What was your First Strategic Pivot (Pivot 1)?**  
   **A:** In July 2026, we tested a Two-Stage GAN pipeline to convert handwriting into clean Computer Modern printed images before OCR. We abandoned it because the GAN hallucinated phantom serifs, erased minus signs and exponents, and merged matrix columns. We pivoted to direct continuous-to-discrete sequence translation.

7. **Q: What is the Attention Coverage Deficit?**  
   **A:** Standard spatial attention has no memory of where it has looked. It suffers from *under-parsing* (skipping faint symbols like minus signs and dots) and *over-parsing* (getting trapped in infinite repetition loops over identical symbols like zeros).

8. **Q: How does CoMER fix the coverage deficit?**  
   **A:** CoMER maintains a cumulative coverage map $\mathbf{C}_t = \sum_{\tau=1}^{t-1} \boldsymbol{\alpha}_\tau$ and uses an Attention Refinement Module (ARM) to penalize already-visited locations, forcing the decoder to seek unparsed strokes.

9. **Q: What are the specifications of the MA102 dataset?**  
   **A:** 939 students, 14-page unruled booklets, 13,146 scanned pages at 200–300 DPI ($2338 \times 1653$ pixels) from IIT Guwahati.

10. **Q: How did you prevent handwriting style leakage?**  
    **A:** We enforced strict Student-Disjoint Partitioning by Roll Number (750 train, 89 val, 100 test). Shuffling crops randomly allows the model to memorize a student's handwriting style, producing fake benchmark scores.

11. **Q: What is the Verbatim Ink Protocol?**  
    **A:** Gold-standard ground truth transcribes 100% literal ink without correcting student algebraic blunders. If a student wrote $-2.0$ instead of $+2.0$, ground truth records $-2.0$.

12. **Q: How does Projection Variance Deskewing work?**  
    **A:** It rotates the image across angles $\theta \in [-15^\circ, +15^\circ]$ and evaluates horizontal projection histograms. When baselines are perfectly level, peaks are sharpest and valleys are emptiest, maximizing projection variance $\text{Var}(H_\theta(y))$.

13. **Q: Why use Sauvola Binarization over Otsu's method?**  
    **A:** Otsu picks a single global threshold, failing under uneven scanner shadows or faint ballpoint pens. Sauvola computes an adaptive threshold for each pixel based on local window mean and standard deviation ($31 \times 31$ window, $k=0.20$).

14. **Q: How does 1D Vertical IoU prevent sliced characters?**  
    **A:** Descenders and ascenders overlap by $30$--$45\%$ vertically on unruled paper. If candidate line intervals overlap by $\text{IoU}_{1D} \ge 0.35$, our algorithm merges them into a single line container.

15. **Q: How does `MatrixGridParser` work?**  
    **A:** It detects tall bracket strokes ($h/w > 3.0$) using connected components, tags the 2D bounding area as a matrix container before line segmentation, projects internal row/column histograms, and formats it as `\begin{bmatrix} ... \end{bmatrix}`.

16. **Q: Why use a Dual-Head Dispatcher in Paradigm A?**  
    **A:** It routes English prose proofs to Microsoft TrOCR and mathematical formulas to our custom DenseNet HMER model, preventing cross-domain syntax crashes.

17. **Q: What is Conversational Mode Collapse in VLMs?**  
    **A:** RLHF alignment causes instruction-tuned models to generate conversational greetings (*"Certainly!..."*) or Python scripts rather than raw \LaTeX\ code.

18. **Q: How does Assistant Prefill Locking fix Conversational Collapse?**  
    **A:** Using `continue_final_message=True`, we prepopulate the assistant's buffer with the formal \LaTeX\ preamble ending in `\section*{Solution}\n\n`. The model's very first predicted token must continue immediately from the header, collapsing conversational probability to zero.

19. **Q: What is the NLP Repetition Penalty Paradox?**  
    **A:** High NLP repetition penalties ($\rho > 1.2$, `no_repeat_ngram_size = 15`) artificially delete authentic mathematical repetitions (consecutive coordinate equations in subspace proofs and sequential row reductions). We lowered the global penalty to $\rho = 1.10$ and used targeted matrix regex collapsers.

20. **Q: How did you guarantee a 100% compilation pass rate?**  
    **A:** We engineered a Pushdown Automaton (Stack Balancer) that tracks open environments on a stack. When generation terminates, it automatically pops and appends closing delimiters (`\end{...}`) in LIFO order.

21. **Q: How did you bypass CentOS 7 CXXABI library errors without root access?**  
    **A:** We preloaded the modern `libstdc++.so.6` bundled inside Anaconda using Python's `ctypes.CDLL("/opt/.../libstdc++.so.6", RTLD_GLOBAL)` before importing PyTorch.

22. **Q: Why couldn't you run FlashAttention-2 on Tesla V100?**  
    **A:** FlashAttention-2 requires asynchronous copy instructions (`cp.async`) present only on Ampere (`sm_80`)+ GPUs. The Tesla V100 is Volta (`sm_70`), which lacks these instructions. We fell back to Cutlass Memory-Efficient SDPA.

23. **Q: How did you fit Qwen2.5-VL-7B into 16GB VRAM?**  
    **A:** In FP16, weights consume 13.67 GiB, leaving 1.83 GiB headroom. We capped image resolution to $1200\,\text{px}$ ($N_{\text{tokens}} \le 1280$), stabilizing peak VRAM at $15.12\,\text{GB}$.

24. **Q: Why did TrOCR output "House of Representatives" on a math script?**  
    **A:** TrOCR was pretrained on 18th/19th century historical legal manuscripts. Facing ambiguous cursive math proofs, its visual encoder lost confidence and its decoder fell back onto its historical literary prior.

25. **Q: What is your primary roadmap for Phase 2 next semester?**  
    **A:** Supervised domain fine-tuning with LoRA on MA102 handwriting crops to eliminate the problem-solver prior, and INT4 quantization to enable native $2000\,\text{px}+$ high-resolution processing without VRAM limits.
