# Copyright (c) Hikvision Research Institute. All rights reserved.
import numpy as np
import mmcv
import torch
import torch.nn as nn
from mmdet.models.losses.utils import weighted_loss
from opera.models.utils.smpl_utils import batch_rodrigues
from ..builder import LOSSES

def batch_l2_loss_param(real,predict,weight):
    # convert to rot mat, multiple angular maps to the same rotation with Pi as a period.
    batch_size = real.shape[0]
    real = batch_rodrigues(real.reshape(-1,3)).contiguous()#(N*J)*3 -> (N*J)*3*3
    predict = batch_rodrigues(predict.reshape(-1,3)).contiguous()#(N*J)*3 -> (N*J)*3*3
    loss = torch.norm((real-predict).view(-1,9), p=2, dim=-1)#self.sl1loss(real,predict)#
    loss = (loss * weight).sum() / (weight.sum())
    return loss

@LOSSES.register_module()
class PoseLoss(nn.Module):
    """OKSLoss.

    Computing the oks loss between a set of predicted poses and target poses.

    Args:
        linear (bool): If True, use linear scale of loss instead of log scale.
            Default: False.
        eps (float): Eps to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
    """

    def __init__(self, loss_weight=0):
        super(PoseLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            valid (torch.Tensor): The visible flag of the target pose.
            area (torch.Tensor): The area of the target pose.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".
        """
        global_orient_pred = pred[:, :3]
        pose_pred = pred[:, 3:]

        gt_global_orient = target[:, :3]
        pose_gt = target[:, 3:]

        global_orient_weight = weight[:, 0].reshape(-1)
        pose_weight = weight[:, 1:].reshape(-1)
        loss = batch_l2_loss_param(gt_global_orient, global_orient_pred, global_orient_weight) + batch_l2_loss_param(pose_gt, pose_pred, pose_weight)
        loss = self.loss_weight * loss
        return loss
