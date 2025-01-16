import torch


def sample_list_random(l, n):
    """
    Sample n elements from list l.
    """
    return [l[i] for i in torch.randperm(len(l))[:n]]


def sample_list_first_n(l, n):
    """
    Sample the first n elements from list l.
    """
    return l[:n]
