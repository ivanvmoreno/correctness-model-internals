import torch


def sample_list(l, n):
    """
    Sample n elements from list l.
    """
    return [l[i] for i in torch.randperm(len(l))[:n]]
