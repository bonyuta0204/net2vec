import math
import torch
from torch.optim import Optimizer 

required = object()

class Custom_Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
            l1_weight_decay=0, l2_weight_decay=0, lower_bound=None, upper_bound=None):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                                        l1_weight_decay=l1_weight_decay,
                                        l2_weight_decay=l2_weight_decay, 
                                        lower_bound=lower_bound,
                                        upper_bound=upper_bound)
        super(Custom_Adam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['l1_weight_decay'] != 0:
                    grad = grad.add(group['l1_weight_decay'], torch.sign(p.data))
                if group['l2_weight_decay'] != 0:
                    grad = grad.add(group['l2_weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

                # custom clamping
                if group['lower_bound'] is not None:
                    p.data.clamp_(min=group['lower_bound'])
                if group['upper_bound'] is not None:
                    p.data.clamp_(max=group['upper_bound'])

        return loss

class Custom_SGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v
        where p, g, v and :math:`\rho` denote the parameters, gradient, velocity, and
        momentum respectively.
        This is in constrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
             v = \rho * v + lr * g \\
             p = p - v
        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 l1_weight_decay=0, l2_weight_decay=0, nesterov=False, 
                 lower_bound=None, upper_bound=None):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        l1_weight_decay=l1_weight_decay, l2_weight_decay=l2_weight_decay, 
                        nesterov=nesterov, lower_bound=lower_bound, upper_bound=upper_bound)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(Custom_SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Custom_SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            l1_weight_decay = group['l1_weight_decay']
            l2_weight_decay = group['l2_weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                # L1 weight decay
                if l1_weight_decay != 0:
                    d_p.add_(l1_weight_decay, torch.sign(p.data))
                if l2_weight_decay != 0:
                    d_p.add_(l2_weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = d_p.clone()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

                # custom clamping
                if group['lower_bound'] is not None:
                    p.data.clamp_(min=group['lower_bound'])
                if group['upper_bound'] is not None:
                    p.data.clamp_(max=group['upper_bound'])

        return loss

