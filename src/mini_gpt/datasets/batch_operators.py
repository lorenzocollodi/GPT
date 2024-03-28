from torch import Tensor, ones_like, triu, zeros_like


def expand_gt(input: Tensor, gt: Tensor) -> Tensor:
    B, _ = input.shape
    assert gt.dim() == 1
    assert gt.shape[0] == B
    gt_expanded = zeros_like(input)
    gt_expanded[:, :-1] = input[:, 1:]
    gt_expanded[:, -1] = gt
    return gt_expanded.view(-1)


def time_expand(
    inpt: Tensor,
) -> Tensor:
    B, C = inpt.shape
    return inpt.unsqueeze(1).repeat(1, C, 1)


def get_mask(
    inpt: Tensor,
) -> Tensor:
    B, T, C = inpt.shape
    mask = triu(ones_like(inpt).view(T, -1), 1).view(T, B, C).permute(1, 0, 2)
    return mask
