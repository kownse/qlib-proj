"""
Training managers for DoubleAdapt + FiLM.

IncrementalManager: naive incremental learning baseline
DoubleAdaptFiLMManager: meta-learning with MAML, dual adapters + FiLM macro conditioning

Based on DoubleAdapt (KDD'23) with FiLM extension.
"""

import copy
import typing
from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd
import torch
from torch import optim, nn
import torch.nn.functional as F

from .net import (
    ForecastModelFiLM, DoubleAdaptFiLM,
    has_rnn,
)

from tqdm import tqdm


class IncrementalManager:
    """Naive incremental learning framework with macro support.

    Each rolling task: train on support set, predict on query set.
    No meta-learning - just standard gradient updates.
    """

    def __init__(
        self,
        model,
        lr_model=0.001,
        online_lr=None,
        x_dim=None,
        need_permute=False,
        over_patience=8,
        begin_valid_epoch=0,
        **kwargs,
    ):
        self.fitted = False
        self.lr_model = lr_model
        self.online_lr = online_lr
        self.over_patience = over_patience
        self.begin_valid_epoch = begin_valid_epoch
        self.framework = self._init_framework(
            model, x_dim, lr_model, need_permute=need_permute, **kwargs
        )
        self.opt = self._init_meta_optimizer(**kwargs)

    def _init_framework(self, model, x_dim=None, lr_model=0.001,
                        weight_decay=0.0, need_permute=False, **kwargs):
        return ForecastModelFiLM(
            model, x_dim=x_dim, lr=lr_model,
            need_permute=need_permute, weight_decay=weight_decay,
        )

    def _init_meta_optimizer(self, **kwargs):
        return self.framework.opt

    def state_dict(self):
        destination = OrderedDict()
        destination['framework'] = self.framework.state_dict()
        destination['framework_opt'] = self.framework.opt.state_dict()
        destination['opt'] = self.opt.state_dict()
        return destination

    def load_state_dict(self, state_dict):
        self.framework.load_state_dict(state_dict['framework'])
        self.framework.opt.load_state_dict(state_dict['framework_opt'])
        self.opt.load_state_dict(state_dict['opt'])

    def override_online_lr_(self):
        if self.online_lr is not None:
            if 'lr_model' in self.online_lr:
                self.lr_model = self.online_lr['lr_model']
                self.opt.param_groups[0]['lr'] = self.online_lr['lr_model']

    def fit(self, meta_tasks_train, meta_tasks_val, checkpoint_path=""):
        """Train on rolling tasks with early stopping on validation IC."""
        self.framework.train()
        torch.set_grad_enabled(True)

        best_ic, patience = -1e3, self.over_patience
        best_checkpoint = copy.deepcopy(self.framework.state_dict())

        for epoch in tqdm(range(300), desc="epoch"):
            for phase, task_list in zip(['train', 'val'], [meta_tasks_train, meta_tasks_val]):
                if phase == "val" and epoch < self.begin_valid_epoch:
                    continue
                pred_y, ic = self._run_epoch(phase, task_list)
                if phase == "val":
                    if ic < best_ic:
                        patience -= 1
                    else:
                        best_ic = ic
                        print(f"best ic: {best_ic:.4f}")
                        patience = self.over_patience
                        best_checkpoint = copy.deepcopy(self.framework.state_dict())
            if patience <= 0:
                break

        self.framework.load_state_dict(best_checkpoint)
        self._run_epoch('train', meta_tasks_val)
        self.fitted = True

        if checkpoint_path:
            print(f'Save checkpoint: {checkpoint_path}')
            torch.save(self.state_dict(), checkpoint_path)

    def _run_epoch(self, phase, task_list, tqdm_show=False):
        pred_y_all, mse_all = [], 0
        indices = np.arange(len(task_list))

        if phase == 'train':
            np.random.shuffle(indices)
        else:
            if phase == "val":
                checkpoint = copy.deepcopy(self.state_dict())
            lr_model = self.lr_model
            self.override_online_lr_()

        self.phase = phase
        for i in (tqdm(indices, desc=phase) if tqdm_show else indices):
            torch.cuda.empty_cache()
            meta_input = task_list[i]
            if not isinstance(meta_input['X_train'], torch.Tensor):
                meta_input = {
                    k: torch.tensor(v, device=self.framework.device, dtype=torch.float32)
                    if 'idx' not in k else v
                    for k, v in meta_input.items()
                }
            pred = self._run_task(meta_input, phase)
            if phase != "train":
                test_idx = meta_input["test_idx"]
                pred_y_all.append(
                    pd.DataFrame({
                        "pred": pd.Series(pred, index=test_idx),
                        "label": pd.Series(meta_input["y_test"], index=test_idx),
                    })
                )

        if phase != "train":
            pred_y_all = pd.concat(pred_y_all)
        if phase == "val":
            self.lr_model = lr_model
            self.load_state_dict(checkpoint)
            ic = pred_y_all.groupby("datetime").apply(
                lambda df: df["pred"].corr(df["label"], method="pearson")
            ).mean()
            print(f"  {phase} IC: {ic:.4f}")
            return pred_y_all, ic
        return pred_y_all, None

    def _run_task(self, meta_input, phase):
        """Single naive incremental learning task with macro support."""
        self.framework.opt.zero_grad()
        macro_train = meta_input.get("macro_train")
        if macro_train is not None:
            macro_train = macro_train.to(self.framework.device)

        y_hat = self.framework(
            meta_input["X_train"].to(self.framework.device),
            macro=macro_train,
        )
        loss = self.framework.criterion(
            y_hat, meta_input["y_train"].to(self.framework.device)
        )
        loss.backward()
        self.framework.opt.step()
        self.framework.opt.zero_grad()

        with torch.no_grad():
            macro_test = meta_input.get("macro_test")
            if macro_test is not None:
                macro_test = macro_test.to(self.framework.device)
            pred = self.framework(
                meta_input["X_test"].to(self.framework.device),
                macro=macro_test,
            )
        return pred.detach().cpu().numpy()

    def inference(self, meta_tasks_test, date_slice=slice(None, None)):
        """Perform incremental learning on the test set."""
        self.framework.train()
        self.framework.to(self.framework.device)
        pred_y_all, ic = self._run_epoch("online", meta_tasks_test, tqdm_show=True)
        pred_y_all = pred_y_all.loc[date_slice]
        return pred_y_all


class DoubleAdaptFiLMManager(IncrementalManager):
    """Meta-learning incremental learning with FeatureAdapter + LabelAdapter + FiLM.

    Uses MAML-style inner loop (via higher library) to adapt the GRU+FiLM backbone
    per rolling task, while meta-learning the adapter parameters in the outer loop.
    """

    def __init__(
        self,
        model,
        lr_model=0.001,
        lr_da=0.01,
        lr_ma=0.001,
        lr_x=None,
        lr_y=None,
        online_lr=None,
        weight_decay=0,
        reg=0.5,
        adapt_x=True,
        adapt_y=True,
        first_order=True,
        factor_num=5,
        x_dim=300,
        need_permute=False,
        num_head=8,
        temperature=10,
        begin_valid_epoch=0,
    ):
        super().__init__(
            model, x_dim=x_dim, lr_model=lr_model,
            lr_ma=lr_ma, lr_da=lr_da, lr_x=lr_x, lr_y=lr_y,
            online_lr=online_lr, weight_decay=weight_decay,
            need_permute=need_permute,
            factor_num=factor_num, temperature=temperature, num_head=num_head,
            begin_valid_epoch=begin_valid_epoch,
        )
        self.adapt_x = adapt_x
        self.adapt_y = adapt_y
        self.reg = reg
        self.sigma = 1 ** 2 * 2
        self.factor_num = factor_num
        self.num_head = num_head
        self.temperature = temperature
        self.first_order = first_order
        self.has_rnn = has_rnn(self.framework)

    def _init_framework(self, model, x_dim=None, lr_model=0.001, need_permute=False,
                        num_head=8, temperature=10, factor_num=5, lr_ma=None,
                        weight_decay=0, **kwargs):
        return DoubleAdaptFiLM(
            model, x_dim=x_dim,
            lr=lr_model if lr_ma is None else lr_ma,
            need_permute=need_permute,
            factor_num=factor_num, num_head=num_head, temperature=temperature,
            weight_decay=weight_decay,
        )

    def _init_meta_optimizer(self, lr_da=0.01, lr_x=None, lr_y=None, **kwargs):
        if lr_x is None or lr_y is None:
            return optim.Adam(self.framework.meta_params, lr=lr_da)
        else:
            return optim.Adam([
                {'params': self.framework.teacher_x.parameters(), 'lr': lr_x},
                {'params': self.framework.teacher_y.parameters(), 'lr': lr_y},
            ])

    def override_online_lr_(self):
        if self.online_lr is not None:
            if 'lr_model' in self.online_lr:
                self.lr_model = self.online_lr['lr_model']
            if 'lr_ma' in self.online_lr:
                self.framework.opt.param_groups[0]['lr'] = self.online_lr['lr_ma']
            if 'lr_da' in self.online_lr:
                self.opt.param_groups[0]['lr'] = self.online_lr['lr_da']
            else:
                if 'lr_x' in self.online_lr:
                    self.opt.param_groups[0]['lr'] = self.online_lr['lr_x']
                if 'lr_y' in self.online_lr and len(self.opt.param_groups) > 1:
                    self.opt.param_groups[1]['lr'] = self.online_lr['lr_y']

    def _run_task(self, meta_input, phase):
        """Single DoubleAdapt + FiLM meta-learning task.

        Uses manual first-order MAML (no higher library) for memory efficiency:
        1. Save original backbone weights
        2. Inner loop: one gradient step on support set
        3. Outer loop: compute loss on query set with adapted weights
        4. Update meta-params, restore + update original weights
        """
        self.framework.opt.zero_grad()
        self.opt.zero_grad()

        X = meta_input["X_train"].to(self.framework.device)
        macro_train = meta_input.get("macro_train")
        if macro_train is not None:
            macro_train = macro_train.to(self.framework.device)

        # Save original backbone weights
        orig_state = {k: v.clone() for k, v in self.framework.model.state_dict().items()}

        # === Inner loop: one gradient step on support set ===
        y_hat, X_adapted = self.framework(
            X, macro=macro_train, model=None, transform=self.adapt_x,
        )
        y = meta_input["y_train"].to(self.framework.device)
        if self.adapt_y:
            raw_y = y
            y = self.framework.teacher_y(X, raw_y, inverse=False)

        inner_loss = self.framework.criterion(y_hat, y)

        # Manual gradient step on backbone only
        backbone_params = list(self.framework.model.parameters())
        grads = torch.autograd.grad(
            inner_loss, backbone_params, create_graph=False, allow_unused=True,
        )
        with torch.no_grad():
            for param, grad in zip(backbone_params, grads):
                if grad is not None:
                    param.data -= self.lr_model * grad

        del y_hat, inner_loss, grads
        macro_train = None

        # === Outer loop: evaluate on query set with adapted backbone ===
        X_test = meta_input["X_test"].to(self.framework.device)
        y_test = meta_input["y_test"].to(self.framework.device)
        macro_test = meta_input.get("macro_test")
        if macro_test is not None:
            macro_test = macro_test.to(self.framework.device)

        pred, X_test_adapted = self.framework(
            X_test, macro=macro_test, model=None, transform=self.adapt_x,
        )
        if self.adapt_y:
            pred = self.framework.teacher_y(X_test, pred, inverse=True)

        if phase != "train":
            meta_end = meta_input["meta_end"]
            output = pred.detach().cpu().numpy()
            X_test = X_test[:meta_end]
            X_test_adapted = X_test_adapted[:meta_end]
            pred = pred[:meta_end]
            y_test = y_test[:meta_end]
            if macro_test is not None:
                macro_test = macro_test[:meta_end]
        else:
            output = pred.detach().cpu().numpy()

        # Meta-learner optimization
        if len(y_test) == 0:
            # No query data, keep adapted weights
            return output

        loss = self.framework.criterion(pred, y_test)

        if self.adapt_y:
            loss_y = F.mse_loss(y, raw_y)
            # First-order approximation: compare adapted vs original model
            with torch.no_grad():
                pred2, _ = self.framework(
                    X_test_adapted, macro=macro_test, model=None, transform=False,
                )
                pred2 = self.framework.teacher_y(X_test, pred2, inverse=True).detach()
                loss_old = self.framework.criterion(pred2.view_as(y_test), y_test)
            loss_y = (loss_old.item() - loss.item()) / self.sigma * loss_y + loss_y * self.reg
            loss_y.backward()

        loss.backward()
        if self.adapt_x or self.adapt_y:
            self.opt.step()

        # Restore original weights and apply outer loop gradient
        # The gradients on backbone params come from outer loss
        # We need to: restore original weights, then apply the accumulated gradients
        backbone_grads = {
            k: p.grad.clone() if p.grad is not None else None
            for k, p in zip(orig_state.keys(), backbone_params)
        }
        self.framework.model.load_state_dict(orig_state)
        # Re-attach saved gradients to restored params
        for param, (k, grad) in zip(backbone_params, backbone_grads.items()):
            param.grad = grad
        self.framework.opt.step()

        del orig_state, backbone_grads
        return output
