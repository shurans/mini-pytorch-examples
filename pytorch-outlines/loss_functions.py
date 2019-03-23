'''
This module contains the loss functions used to train the surface normals estimation models.
'''

import torch
import torch.nn as nn
import numpy as np


def cross_entropy2d(logit, target, ignore_index=255, weight=None, batch_average=True):
    """
    The loss is

    .. math::
        \sum_{i=1}^{\\infty} x_{i}

        `(minibatch, C, d_1, d_2, ..., d_K)`

    Args:
        logit (Tensor): Output of network
        target (Tensor): Ground Truth
        ignore_index (int, optional): Defaults to 255. The pixels with this labels do not contribute to loss
        weight (List, optional): Defaults to None. Weight assigned to each class
        batch_average (bool, optional): Defaults to True. Whether to consider the loss of each element in the batch.

    Returns:
        Float: The value of loss.
    """

    n, c, h, w = logit.shape
    target = target.squeeze(1)

    if weight is None:
        criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='sum')
    else:
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weight, dtype=torch.float32),
                                        ignore_index=ignore_index, reduction='sum')

    loss = criterion(logit, target.long())

    if batch_average:
        loss /= n

    return loss
