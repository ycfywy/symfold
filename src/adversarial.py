# -*- coding: utf-8 -*-
"""
Family-Adversarial Pretraining: Gradient Reversal Layer + family classifier.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torch.autograd import Function


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x, lambda_=1.0):
    return GradReverse.apply(x, lambda_)


class FamilyClassifier(nn.Module):
    def __init__(self, feat_dim, num_families, hidden=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_families),
        )

    def forward(self, x, lambda_):
        return self.net(grad_reverse(x, lambda_))
