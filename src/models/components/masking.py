import torch


def mask_before_cross_attention(inputs: torch.Tensor, mask_probability: float) -> torch.Tensor:
    """
    Mask the inputs before the cross attention.
    """
    raise NotImplementedError


def mask_after_cross_attention(inputs: torch.Tensor, mask_probability: float) -> torch.Tensor:
    """
    Mask the inputs after the cross attention.
    """
    raise NotImplementedError