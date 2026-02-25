"""
Monkey-patch higher library's DifferentiableOptimizer.step() to support:
  - first_order MAML (detach updated params)
  - frozen parameters (no grad required)
  - grad_callback

From DoubleAdapt (KDD'23): https://github.com/SJTU-DMTai/DoubleAdapt
"""

from higher import optim, patch
from higher.optim import _OverrideType, _GradCallbackType
import typing as _typing

import torch as _torch

import sys

if "_forward_pre_hooks" in patch._internal_attrs:
    patch._internal_attrs.remove("_forward_pre_hooks")


class DifferentiableOptimizer(optim.DifferentiableOptimizer):
    def step(
        self,
        input: _torch.Tensor,
        params: _typing.Iterable[_torch.Tensor] = None,
        override: _typing.Optional[_OverrideType] = None,
        grad_callback: _typing.Optional[_GradCallbackType] = None,
        first_order=False,
        **kwargs
    ) -> _typing.Iterable[_torch.Tensor]:
        # Deal with override
        if override is not None:
            self._apply_override(override)

        if self._fmodel is None or self._fmodel.fast_params is None:
            if params is None:
                raise ValueError(
                    "params kwarg must be passed to step if the differentiable "
                    "optimizer doesn't have a view on a patched model with params."
                )
        else:
            params = self._fmodel.fast_params if params is None else params

        params = list(params)

        # Gracefully deal with frozen parameters
        grad_targets = [
            p if p.requires_grad else _torch.tensor([], requires_grad=True)
            for p in params
        ]
        all_grads = _torch.autograd.grad(
            input, grad_targets,
            create_graph=self._track_higher_grads,
            allow_unused=True,
        )
        if grad_callback is not None:
            all_grads = grad_callback(all_grads)
        elif self._grad_callback is not None:
            all_grads = self._grad_callback(all_grads)

        grouped_grads = []
        for group, mapping in zip(self.param_groups, self._group_to_param_list):
            grads = []
            for i, index in enumerate(mapping):
                group["params"][i] = params[index]
                grads.append(all_grads[index])
            grouped_grads.append(grads)

        self._update(grouped_grads)

        new_params = params[:]
        for group, mapping in zip(self.param_groups, self._group_to_param_list):
            for p, index in zip(group["params"], mapping):
                if not first_order:
                    new_params[index] = p
                else:
                    new_params[index] = p.detach().requires_grad_()

        if self._fmodel is not None:
            self._fmodel.update_params(new_params)

        return new_params


setattr(
    sys.modules["higher.optim"].__dict__["DifferentiableOptimizer"],
    "step",
    DifferentiableOptimizer.step,
)
