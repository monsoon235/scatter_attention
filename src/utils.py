import torch


def zero_cost_repeat(tensor: torch.Tensor, dim: int, repeat: int):
    ndim = tensor.ndim
    assert -ndim - 1 <= dim <= ndim, f'dim must be in range [{-ndim - 1}, {ndim}]'
    if dim < 0:
        dim += ndim + 1
    stride = list(tensor.stride())
    size = list(tensor.size())
    stride.insert(dim, 0)
    size.insert(dim, repeat)
    return torch.as_strided(tensor, size=size, stride=stride)
