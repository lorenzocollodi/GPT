from torch import Tensor, zeros_like, Size


def expand_gt(input: Tensor, gt: Tensor) -> Tensor:
    B, _ = input.shape
    assert gt.dim() == 1
    assert gt.shape[0] == B
    gt_expanded = zeros_like(input)
    gt_expanded[:, :-1] = input[:, 1:]
    gt_expanded[:, -1] = gt
    return gt_expanded.view(-1)