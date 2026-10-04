"""
PyTorch Native Multi-Task Composite Loss Function for HMER Training.
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class CompositeHMERLoss(nn.Module):
    """
    Unified Multi-Task PyTorch Loss Layer for HMER Training.
    """
    def __init__(
        self,
        lambda_arm: float = 1.0,
        lambda_count: float = 0.5,
        pad_id: int = 0
    ):
        super().__init__()
        self.lambda_arm = lambda_arm
        self.lambda_count = lambda_count
        self.pad_id = pad_id
        
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="mean")
        self.mse_loss = nn.MSELoss(reduction="mean")

    def forward(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        coverage_sums: torch.Tensor,
        pred_counts: torch.Tensor,
        target_counts: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        b, seq_len, v_size = logits.shape
        logits_flat = logits.view(-1, v_size)
        targets_flat = target_ids.view(-1)
        
        valid_mask = (targets_flat != self.pad_id)
        if valid_mask.any():
            l_seq = self.ce_loss(logits_flat[valid_mask], targets_flat[valid_mask])
        else:
            l_seq = torch.tensor(0.0, device=logits.device, requires_grad=True)

        l_arm = torch.mean(coverage_sums)
        l_count = self.mse_loss(pred_counts, target_counts)

        l_total = l_seq + self.lambda_arm * l_arm + self.lambda_count * l_count

        return {
            "loss_total": l_total,
            "loss_seq": l_seq,
            "loss_arm": l_arm,
            "loss_count": l_count
        }
