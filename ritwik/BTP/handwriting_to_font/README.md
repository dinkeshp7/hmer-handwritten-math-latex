# Stage 1: handwriting -> font-style line translation

## Files
- `models.py` — U-Net generator + PatchGAN discriminator
- `losses.py` — L1 + perceptual (VGG) + adversarial + optional OCR-consistency
- `dataset.py` — paired dataset loader with padding/masking for variable-width crops
- `train.py` — training loop
- `inference.py` — run a trained generator on a real page, preserving spacing
- `check_segmentation.py` — visualize line detection with no model required

## Start here, with your real notes, right now

You mentioned testing with your own handwritten notes — since you don't
have a trained checkpoint yet, start with segmentation only. This is the
part responsible for spacing preservation, so it's worth validating first,
independent of the model:

```bash
pip install -r requirements.txt
python check_segmentation.py --image my_notes.jpg --out boxes.png
```

Open `boxes.png` and check that each detected red box matches one line of
your handwriting. If lines merge together, increase `--min_gap`. If it
picks up stray marks as fake lines, increase `--min_line_height`.

## Once you have paired training data

Build a directory like:
```
data/
  handwriting/  0001.png  0002.png  ...
  font/         0001.png  0002.png  ...   (matching filenames)
```
then:
```bash
python train.py --data_root data/ --epochs 100
```
Checkpoints land in `stage1_runs/checkpoints/`, sample grids (input |
generated | target) in `stage1_runs/samples/` so you can watch progress.

## Running on a real page once trained

```bash
python inference.py --checkpoint stage1_runs/checkpoints/generator_epoch50.pt \
                     --image my_notes.jpg --out font_style.png
```

This segments the page into lines, translates each line, and pastes every
result back at its original row position — so `font_style.png` has
identical line spacing to `my_notes.jpg`, by construction rather than by
the model guessing.

## Notes
- `losses.py`'s OCR-consistency term is optional and needs a pretrained
  OCR backbone plugged in (`ocr_model=None` skips it for now).
- Input/target crops are grayscale, normalized to `[-1, 1]` to match the
  generator's `Tanh` output.
- The generator requires width divisible by 16 — both `dataset.py`'s
  collate function and `inference.py`'s resizing handle this automatically.
