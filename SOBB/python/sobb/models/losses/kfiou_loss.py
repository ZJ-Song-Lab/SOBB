"""Kalman Filter IoU loss for oriented bounding boxes.

Reference: Yang et al., "KFIoU: Learning Oriented Object Detection
with Kalman Filter IoU", CVPR 2023.
"""
import jittor as jt
from jittor import nn
from ..registry import LOSSES


@LOSSES.register_module()
class KFIoULoss(nn.Module):
    """KFIoU regression loss for 5-parameter OBB.

    Models the OBB as a Gaussian and computes a Kalman-filter-style
    IoU surrogate that is smooth and fully differentiable.
    """

    def __init__(self, loss_weight=1.0, mode='linear'):
        super(KFIoULoss, self).__init__()
        self.loss_weight = loss_weight
        self.mode = mode

    def _obb_to_gaussian(self, obb):
        x, y, w, h, a = obb[0], obb[1], obb[2], obb[3], obb[4]
        w = jt.clamp(w, min_v=1e-3)
        h = jt.clamp(h, min_v=1e-3)
        cos_a = jt.cos(a)
        sin_a = jt.sin(a)
        R = jt.stack([cos_a, -sin_a, sin_a, cos_a]).reshape(2, 2)
        D = jt.array([[w * w / 4.0, 0.0],
                       [0.0, h * h / 4.0]])
        Sigma = R.matmul(D).matmul(R.transpose(0, 1))
        mu = jt.array([x, y])
        return mu, Sigma

    def _kf_iou(self, mu1, S1, mu2, S2):
        """Kalman-filter IoU surrogate."""
        diff = mu1 - mu2
        S_sum = S1 + S2
        S_sum_inv = jt.linalg.inv(S_sum + jt.eye(2) * 1e-6)
        mahala = (diff.reshape(1, 2)).matmul(S_sum_inv).matmul(diff.reshape(2, 1))
        det_S = jt.linalg.det(S1 + jt.eye(2) * 1e-6)
        det_T = jt.linalg.det(S2 + jt.eye(2) * 1e-6)
        det_sum = jt.linalg.det(S_sum + jt.eye(2) * 1e-6)
        iou = det_S + det_T - det_sum - mahala.sum()
        iou = iou / (det_S + det_T + 1e-6)
        return iou

    def execute(self, pred, target, weight=None, avg_factor=None):
        pred = pred.reshape(-1, 5)
        target = target.reshape(-1, 5)
        losses = []
        for i in range(pred.shape[0]):
            mu1, S1 = self._obb_to_gaussian(pred[i])
            mu2, S2 = self._obb_to_gaussian(target[i])
            iou = self._kf_iou(mu1, S1, mu2, S2)
            if self.mode == 'linear':
                losses.append(1.0 - iou)
            else:
                losses.append(-jt.log(iou + 1e-6))
        loss = jt.stack(losses)
        if weight is not None:
            loss = loss * weight.reshape(-1)
        if avg_factor is not None:
            loss = loss.sum() / max(avg_factor, 1.0)
        else:
            loss = loss.mean()
        return loss * self.loss_weight
