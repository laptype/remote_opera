import numpy as np
import torch
import torch.nn as nn
from ..builder import LOSSES
from opera.models.utils.smpl_utils import rot6D_to_angular, batch_rodrigues
from opera.models.losses.prior_loss import MaxMixturePrior, angle_prior

def batch_l2_loss_param(real,predict):
    # convert to rot mat, multiple angular maps to the same rotation with Pi as a period.
    batch_size = real.shape[0]
    real = batch_rodrigues(real.reshape(-1,3)).contiguous()#(N*J)*3 -> (N*J)*3*3
    predict = batch_rodrigues(predict.reshape(-1,3)).contiguous()#(N*J)*3 -> (N*J)*3*3
    loss = torch.norm((real-predict).view(-1,9), p=2, dim=-1)#self.sl1loss(real,predict)#
    loss = loss.mean()
    return loss

@LOSSES.register_module()
class ParamLoss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, path, loss_weight=1.0):
        super(ParamLoss, self).__init__()
        self.loss_weight = loss_weight
        self.gmm_prior = MaxMixturePrior(prior_folder=path, num_gaussians=8, dtype=torch.float32).cuda()

    def forward(self,
                pose_pred,
                shape_pred,
                weight=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        inds = (weight.sum(-1) == weight.shape[1])
        # global_orient_pred = pose_pred[:,:6]
        # pose_pred = pose_pred[:,6:-10]
        # shape_pred = shape_pred[:,-10:]

        # global_orient_pred = rot6D_to_angular(global_orient_pred)
        # pose_pred = rot6D_to_angular(pose_pred)

        # pose = torch.cat([global_orient_pred, pose_pred, shape_pred], -1)

        # global_orient_pred= global_orient_pred[inds]
        pose_pred = pose_pred[inds]
        # shape_pred = shape_pred[inds]
        pose_pred = pose_pred[:, :69]
        gmm_prior_loss = self.gmm_prior(pose_pred, shape_pred).mean()/100.
        angle_prior_loss = angle_prior(pose_pred).mean()/5.
        loss_prior = gmm_prior_loss + angle_prior_loss
        return loss_prior*self.loss_weight
