# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import torch
from . import dispatch, Torch

__all__ = []


@dispatch(Torch)
def shape(a):
    s = a.shape
    return tuple(s[i] for i in range(rank(a)))


@dispatch(Torch)
def shape_int(a):
    return shape(a)


@dispatch(Torch)
def rank(a):
    return len(a.shape)


@dispatch(Torch)
def length(a):
    return a.numel()


@dispatch(Torch)
def expand_dims(a, axis=0):
    return torch.unsqueeze(a, dim=axis)


@dispatch(Torch)
def diag(a):
    return torch.diag(a)


# ----

@dispatch(Torch)
def reshape(a, shape=(-1,)):
    return torch.reshape(a, shape=shape)
