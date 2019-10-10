#!/usr/bin/env python
"""
Extragradient Optimizer

Mostly copied from the extragrad paper repo.

MIT License
Copyright (c) Facebook, Inc. and its affiliates.
written by Hugo Berard (berard.hugo@gmail.com) while at Facebook.
"""

import torch
from torch.optim import Optimizer, SGD

def extragrad_step(optimizer, model, i):
    if i % 2 == 0:
        optimizer.extrapolate()
    else:
        optimizer.step()
        model.zero_grad()

class Extragradient(Optimizer):
    """Base class for optimizers with extrapolation step.
        Arguments:
        params (iterable): an iterable of :class:`torch.Tensor` s or
            :class:`dict` s. Specifies what Tensors should be optimized.
        defaults: (dict): a dict containing default values of optimization
            options (used when a parameter group doesn"t specify them).
    """
    def __init__(self, params, defaults):
        super(Extragradient, self).__init__(params, defaults)
        self.params_copy = []

    def update(self, p, group):
        raise NotImplementedError

    def extrapolate(self):
        """Performs the extrapolation step and save a copy of the current
          parameters for the update step.
        """
        # Check if a copy of the parameters was already made.
        is_empty = len(self.params_copy) == 0
        for group in self.param_groups:
            for param in group["params"]:
                u = self.update(param, group)
                if is_empty:
                    self.params_copy.append(param.data.clone()) # save w[t]

                param.data += u # w[t + .5] = w[t] - eta * F(w[t])

    def step(self):
        """Performs a single optimization step.
        """
        i = 0
        for group in self.param_groups:
            for param in group["params"]:

                if not param.grad:
                    continue

                u = self.update(param, group)
                param.data = self.params_copy[i] + u # w[t + 1] = w[t] - eta * F(w[t + .5])
                i += 1

        # Free the old parameters
        self.params_copy = []


class ExtraSGD(Extragradient):
    """Implements stochastic gradient descent with extrapolation step (optionally with momentum).
    Nesterov momentum is based on the formula from
    """
    def __init__(self, params, lr, weight_decay=0):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay
        }
        super(ExtraSGD, self).__init__(params, defaults)

    def update(self, param, group):
        grad = param.grad.data
        if group["weight_decay"]:
            grad += group["weight_decay"] * param.data

        return -group["lr"] * grad
