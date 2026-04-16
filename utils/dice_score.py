import torch
from torch import Tensor


def _apply_valid_mask(input: Tensor, target: Tensor, valid_mask: Tensor = None):
    if valid_mask is None:
        return input, target

    valid_mask = valid_mask.to(device=input.device)
    if valid_mask.dtype != torch.bool:
        valid_mask = valid_mask > 0
    valid_mask = valid_mask.reshape(-1)

    if input.dim() < 3:
        raise ValueError("valid_mask is only supported for batched 2D masks.")
    if input.shape[0] != valid_mask.shape[0]:
        raise ValueError(
            f"valid_mask length {valid_mask.shape[0]} does not match batch dimension {input.shape[0]}."
        )

    return input[valid_mask], target[valid_mask]


def dice_coeff(
    input: Tensor,
    target: Tensor,
    reduce_batch_first: bool = False,
    epsilon: float = 1e-6,
    valid_mask: Tensor = None,
):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    input, target = _apply_valid_mask(input, target, valid_mask=valid_mask)
    if input.numel() == 0:
        return input.new_tensor(0.0)

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(
    input: Tensor,
    target: Tensor,
    reduce_batch_first: bool = False,
    epsilon: float = 1e-6,
    valid_mask: Tensor = None,
):
    # Average of Dice coefficient for all classes
    input, target = _apply_valid_mask(input, target, valid_mask=valid_mask)
    if input.numel() == 0:
        return input.new_tensor(0.0)
    return dice_coeff(
        input.flatten(0, 1),
        target.flatten(0, 1),
        reduce_batch_first,
        epsilon,
    )


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False, valid_mask: Tensor = None):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True, valid_mask=valid_mask)


def seg_loss(input: Tensor, target: Tensor, valid_mask: Tensor = None,
             focal_weight: float = 0.5, gamma: float = 2.0, alpha: float = 0.75):
    """Combined Focal + Dice loss for sparse masks.

    Focal loss up-weights hard/rare examples (positive pixels in sparse masks),
    providing strong gradients even at initialization. Dice loss ensures global
    overlap optimization.

    Args:
        alpha: weight for positive class (0.75 = 3x more weight on sparse positives)
        gamma: focusing parameter (2.0 = standard focal loss)
    """
    input_flat, target_flat = _apply_valid_mask(input, target, valid_mask=valid_mask)
    if input_flat.numel() == 0:
        return input.new_zeros(())

    # Focal loss
    p = input_flat.clamp(1e-6, 1 - 1e-6)
    ce = -target_flat * torch.log(p) - (1 - target_flat) * torch.log(1 - p)
    p_t = target_flat * p + (1 - target_flat) * (1 - p)
    alpha_t = target_flat * alpha + (1 - target_flat) * (1 - alpha)
    focal = alpha_t * (1 - p_t) ** gamma * ce
    focal_loss = focal.mean()

    d_loss = 1 - dice_coeff(input_flat, target_flat, reduce_batch_first=True)
    return focal_weight * focal_loss + (1 - focal_weight) * d_loss
    # return 1 - fn(input, target, reduce_batch_first=False)
