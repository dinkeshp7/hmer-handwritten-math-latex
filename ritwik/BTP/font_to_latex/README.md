# Stage 2: font-page image -> LaTeX

## Files
- `tokenizer.py` — LaTeX tokenizer that keeps structure/spacing commands (`\par`, `\\`, `\begin{...}`, whitespace runs) as atomic tokens
- `build_vocab.py` — scans a corpus of `.tex` files and builds `vocab.json`
- `models.py` — ViT-style patch encoder + autoregressive transformer decoder, with optional layout (bounding box) conditioning
- `dataset.py` — paired dataset loader (font image + LaTeX + optional line boxes)
- `train.py` — training loop
- `inference.py` — beam search decoding + automatic brace/environment repair

## 1. Build a vocabulary

```bash
pip install -r requirements.txt
python build_vocab.py --corpus_dir /path/to/tex/files --out vocab.json
```

Use the same LaTeX corpus you rendered your synthetic font-page images
from (see the earlier data-generation discussion), so the vocabulary
actually covers what your images depict.

## 2. Prepare paired data

```
data/
  images/  0001.png   0002.png  ...   (font-rendered page or line images)
  latex/   0001.tex   0002.tex  ...   (matching LaTeX source)
  boxes/   0001.json  0002.json ...   (optional: [[x0,y0,x1,y1], ...], normalized 0-1)
```

Include `boxes/` and pass `--use_boxes` to `train.py` if you have line
geometry available from Stage 1's segmentation step — this gives the
decoder an explicit signal for where lines break, rather than asking it
to infer that purely from pixel gaps.

## 3. Train

```bash
python train.py --data_root data/ --vocab vocab.json --epochs 50 --use_boxes
```

Checkpoints save to `stage2_runs/model_epochN.pt` and bundle the model's
architecture config alongside its weights, so inference doesn't need
matching CLI flags — it reconstructs the exact same architecture
automatically.

## 4. Run inference

```bash
python inference.py --checkpoint stage2_runs/model_epoch40.pt \
                     --vocab vocab.json --image font_page.png
```

Prints the predicted LaTeX. Beam search (`--beam_width`, default 5) is
followed by an automatic repair pass that balances unmatched braces and
`\begin{}`/`\end{}` pairs — a common decoder failure mode that's cheaper
to fix mechanically than to eliminate purely through training.

## Connecting to Stage 1

Feed Stage 1's composited "font-style" page image directly as this
model's input. If you pass along the same line boxes Stage 1 used for
compositing (via `--use_boxes`), Stage 2 gets the layout signal for free
without re-deriving it from pixels.

## Notes
- Image height/width are padded to a multiple of `--patch_size` (default
  16) automatically in both `dataset.py` and `inference.py`.
- `label_smoothing=0.1` is applied in the training loss — helps prevent
  the decoder from becoming overconfident on frequent tokens like braces
  and whitespace.
- Not included here, left as a future addition: an attention-coverage
  regularizer (penalizing the decoder for attending to the same encoder
  region twice or skipping regions), which can reduce repeated/dropped
  tokens on longer documents.
