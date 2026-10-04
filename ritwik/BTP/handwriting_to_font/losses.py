"""
Combined loss for Stage 1 training:
L1 + perceptual (VGG) + adversarial + optional OCR-consistency.
"""
import torch
import torch.nn as nn
import torchvision.models as tv_models


class VGGPerceptualLoss(nn.Module):
    """
    Compares generated vs target images in VGG16 feature space rather than
    raw pixels -- catches structural similarity that pixel loss misses.
    Expects input in [-1, 1]; grayscale crops are repeated across channels.
    """
    def __init__(self, layers=(3, 8, 15, 22)):
        super().__init__()
        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.DEFAULT).features
        self.layers = set(layers)
        self.vgg = vgg.eval()
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _prep(self, x):
        x = (x + 1) / 2  # [-1, 1] -> [0, 1]
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return (x - self.mean) / self.std

    def forward(self, generated, target):
        gen, tgt = self._prep(generated), self._prep(target)
        loss = 0.0
        for i, layer in enumerate(self.vgg):
            gen, tgt = layer(gen), layer(tgt)
            if i in self.layers:
                loss = loss + nn.functional.l1_loss(gen, tgt)
            if i == max(self.layers):
                break
        return loss


class OCRConsistencyLoss(nn.Module):
    """
    Penalizes the generator for producing output a frozen, pretrained OCR
    model can no longer read correctly. `ocr_model` should map an image
    batch to per-timestep logits over a character vocabulary -- pick a
    backbone (e.g. a small CRNN trained on your font corpus) and plug it
    in once you have one; this loss term is optional until then.
    """
    def __init__(self, ocr_model, blank_idx=0):
        super().__init__()
        self.ocr_model = ocr_model.eval()
        for p in self.ocr_model.parameters():
            p.requires_grad = False
        self.ctc_loss = nn.CTCLoss(blank=blank_idx, zero_infinity=True)

    def forward(self, generated, target_text_indices, target_lengths):
        logits = self.ocr_model(generated)  # (T, B, num_classes)
        log_probs = nn.functional.log_softmax(logits, dim=2)
        input_lengths = torch.full(
            (logits.size(1),), logits.size(0), dtype=torch.long, device=logits.device
        )
        return self.ctc_loss(log_probs, target_text_indices, input_lengths, target_lengths)


class Stage1Loss(nn.Module):
    """
    Combined generator loss:
        total = w_l1 * L1 + w_perc * VGG + w_adv * adversarial (+ w_ocr * OCR)

    Pass ocr_model=None to skip the OCR-consistency term until you have a
    trained OCR backbone available.
    """
    def __init__(self, ocr_model=None, w_l1=100.0, w_perc=10.0, w_adv=1.0, w_ocr=5.0):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.perceptual = VGGPerceptualLoss()
        self.adv_loss = nn.BCEWithLogitsLoss()
        self.ocr = OCRConsistencyLoss(ocr_model) if ocr_model is not None else None
        self.w_l1, self.w_perc, self.w_adv, self.w_ocr = w_l1, w_perc, w_adv, w_ocr

    def generator_loss(self, generated, target, disc_fake_logits,
                        target_text_indices=None, target_lengths=None):
        l1 = self.l1(generated, target)
        perc = self.perceptual(generated, target)
        adv = self.adv_loss(disc_fake_logits, torch.ones_like(disc_fake_logits))

        total = self.w_l1 * l1 + self.w_perc * perc + self.w_adv * adv
        parts = {"l1": l1.item(), "perceptual": perc.item(), "adversarial": adv.item()}

        if self.ocr is not None and target_text_indices is not None:
            ocr = self.ocr(generated, target_text_indices, target_lengths)
            total = total + self.w_ocr * ocr
            parts["ocr"] = ocr.item()

        return total, parts

    def discriminator_loss(self, disc_real_logits, disc_fake_logits):
        real = self.adv_loss(disc_real_logits, torch.ones_like(disc_real_logits))
        fake = self.adv_loss(disc_fake_logits, torch.zeros_like(disc_fake_logits))
        return (real + fake) * 0.5
