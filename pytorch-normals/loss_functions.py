'''
This module contains the loss functions used to train the surface normals estimation models.
'''

import torch
import torch.nn as nn
import numpy as np


def loss_fn_cosine(input_vec, target_vec, reduction='elementwise_mean'):
    '''A cosine loss function for use with surface normals estimation.
    Calculates the cosine loss between 2 vectors. Both should be of the same size.

    Arguments:
        input_vec {tensor} -- The 1st vectors with whom cosine loss is to be calculated
                              The dimensions of the matrices are expected to be (batchSize, 3, height, width).
        target_vec {tensor } -- The 2nd vectors with whom cosine loss is to be calculated
                                The dimensions of the matrices are expected to be (batchSize, 3, height, width).

    Keyword Arguments:
        reduction {str} -- Can have values 'elementwise_mean' and 'none'.
                           If 'elemtwise_mean' is passed, the mean of all elements is returned
                           if 'none' is passed, a matrix of all cosine losses is returned, same size as input.
                           (default: {'elementwise_mean'})

    Raises:
        Exception -- Exception is an invalid reduction is passed

    Returns:
        tensor -- A single mean value of cosine loss or a matrix of elementwise cosine loss.
    '''

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    loss_cos = 1.0 - cos(input_vec, target_vec)
    if reduction == 'elementwise_mean':
        loss_cos = torch.mean(loss_cos)
    elif reduction == 'none':
        pass
    else:
        raise Exception(
            'Invalid value for reduction  parameter passed. Please use \'elementwise_mean\' or \'none\''.format())

    return loss_cos


def loss_fn_radians(input_vec, target_vec, reduction='elementwise_mean'):
    '''Loss func for estimation of surface normals. Calculated the angle between 2 vectors
    by taking the inverse cos of cosine loss.

    Arguments:
        input_vec {tensor} -- First vector with whole loss is to be calculated.
                              Expected size (batchSize, 3, height, width)
        target_vec {tensor} -- Second vector with whom the loss is to be calculated.
                               Expected size (batchSize, 3, height, width)

    Keyword Arguments:
        reduction {str} -- Can have values 'elementwise_mean' and 'none'.
                           If 'elemtwise_mean' is passed, the mean of all elements is returned
                           if 'none' is passed, a matrix of all cosine losses is returned, same size as input.
                           (default: {'elementwise_mean'})

    Raises:
        Exception -- If any unknown value passed as reduction argument.

    Returns:
        tensor -- Loss from 2 input vectors. Size depends on value of reduction arg.
    '''

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    loss_cos = cos(input_vec, target_vec)
    if reduction == 'elementwise_mean':
        loss_rad = torch.acos(torch.mean(loss_cos))
    elif reduction == 'none':
        loss_rad = torch.acos(loss_cos)
    else:
        raise Exception(
            'Invalid value for reduction  parameter passed. Please use \'elementwise_mean\' or \'none\''.format())

    return loss_rad


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
