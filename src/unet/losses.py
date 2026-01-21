from __future__ import annotations
import torch


def total_variation_3d(x: torch.Tensor) -> torch.Tensor:
    # x: (N,1,D,H,W)
    dx = torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :]).mean()
    dy = torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :]).mean()
    dz = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1]).mean()
    return dx + dy + dz


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # mask: (N,1,D,H,W) float {0,1}
    m = mask.bool()
    return torch.nn.functional.l1_loss(pred[m], target[m])


def hybrid_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                l1_alpha: float = 1.0, tv_alpha: float = 0.0) -> tuple[torch.Tensor, float, float]:
    l1 = masked_l1(pred, target, mask)
    tv = total_variation_3d(pred) if tv_alpha > 0 else pred.new_tensor(0.0)
    loss = l1_alpha * l1 + tv_alpha * tv
    return loss, float(l1.item()), float(tv.item())

