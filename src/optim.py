#!/usr/bin/env python
"""
Extragradient Optimizer

Mostly copied from the extragrad paper repo.

MIT License
Copyright (c) Facebook, Inc. and its affiliates.
written by Hugo Berard (berard.hugo@gmail.com) while at Facebook.
"""
import torch
import torch.optim as optim
from torch.optim import Optimizer


def get_optimizers(g, d, opts):
    optimizer = opts.train.optimizer

    if not isinstance(optimizer, str):
        # legacy use_extragradient_optimizer compatibility
        if opts.train.use_extragradient_optimizer:
            optimizer = "extrasgd"
        else:
            optimizer = "adam"

    if optimizer.lower() not in {"adam", "extraadam", "extrasgd"}:
        raise ValueError("unknown optimizer from train opts {}".format(opts.train))

    if optimizer.lower() == "adam":
        return (
            optim.Adam(g.parameters(), lr=opts.train.lr_g, betas=(0.5, 0.999)),
            optim.Adam(d.parameters(), lr=opts.train.lr_d, betas=(0.5, 0.999)),
        )
    if optimizer.lower() == "extraadam":
        return (
            ExtraAdam(g.parameters(), lr=opts.train.lr_g, betas=(0.5, 0.999)),
            ExtraAdam(d.parameters(), lr=opts.train.lr_d, betas=(0.5, 0.999)),
        )
    if optimizer.lower() == "extrasgd":
        return (
            ExtraSGD(g.parameters(), opts.train.lr_g),
            ExtraSGD(d.parameters(), opts.train.lr_d),
        )


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
                if param.grad is None:
                    continue
                u = self.update(param, group)
                if is_empty:
                    self.params_copy.append(param.data.clone())  # save w[t]

                param.data += u  # w[t + .5] = w[t] - eta * F(w[t])

    def step(self):
        """Performs a single optimization step.
        """
        i = 0
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    i += 1
                    continue
                u = self.update(param, group)
                param.data = (
                    self.params_copy[i] + u
                )  # w[t + 1] = w[t] - eta * F(w[t + .5])
                i += 1

        # Free the old parameters
        self.params_copy = []


class ExtraSGD(Extragradient):
    """Implements stochastic gradient descent with extrapolation step (optionally with momentum).
    Nesterov momentum is based on the formula from
    """

    def __init__(self, params, lr, weight_decay=0):
        defaults = {"lr": lr, "weight_decay": weight_decay}
        super(ExtraSGD, self).__init__(params, defaults)

    def update(self, param, group):
        grad = param.grad.data
        if group["weight_decay"]:
            grad += group["weight_decay"] * param.data

        return -group["lr"] * grad


class ExtraAdam(Extragradient):
    """Implements the Adam algorithm with extrapolation step.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad
        )
        super(ExtraAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(ExtraAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    def update(self, p, group):
        if p.grad is None:
            return None
        grad = p.grad.data
        if grad.is_sparse:
            raise RuntimeError(
                "Adam does not support sparse gradients, please consider SparseAdam instead"
            )
        amsgrad = group["amsgrad"]

        state = self.state[p]

        # State initialization
        if len(state) == 0:
            state["step"] = 0
            # Exponential moving average of gradient values
            state["exp_avg"] = torch.zeros_like(p.data)
            # Exponential moving average of squared gradient values
            state["exp_avg_sq"] = torch.zeros_like(p.data)
            if amsgrad:
                # Maintains max of all exp. moving avg. of sq. grad. values
                state["max_exp_avg_sq"] = torch.zeros_like(p.data)

        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
        if amsgrad:
            max_exp_avg_sq = state["max_exp_avg_sq"]
        beta1, beta2 = group["betas"]

        state["step"] += 1

        if group["weight_decay"] != 0:
            grad = grad.add(group["weight_decay"], p.data)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            # Use the max. for normalizing running avg. of gradient
            denom = max_exp_avg_sq.sqrt().add_(group["eps"])
        else:
            denom = exp_avg_sq.sqrt().add_(group["eps"])

        bias_correction1 = 1 - beta1 ** state["step"]
        bias_correction2 = 1 - beta2 ** state["step"]
        step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

        return -step_size * exp_avg / denom
