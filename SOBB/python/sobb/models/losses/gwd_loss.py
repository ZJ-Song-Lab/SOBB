"""Gaussian Wasserstein Distance loss for oriented bounding boxes.

Reference: Yang et al., "Rethinking Rotated Object Detection
with Gaussian Wasserstein Distance Loss", ICML 2021.
"""
import jittor as jt
from jittor import nn
from ..registry import LOSSES


@LOSSES.register_module()
class GWDLoss(nn.Module):
    """GWD regression loss for 5-parameter OBB (x, y, w, h, angle).

    Converts predicted and target OBBs to 2-D Gaussians, then
    computes the Wasserstein distance between them as a smooth
    regression loss.
    """

    def __init__(self, loss_weight=1.0, tau=1.0):
        super(GWDLoss, self).__init__()
        self.loss_weight = loss_weight
        self.tau = tau

    def _obb_to_gaussian(self, obb):
        """Convert (x, y, w, h, angle) to 2-D Gaussian (mu, Sigma)."""
        x, y, w, h, a = obb[0], obb[1], obb[2], obb[3], obb[4]
        w = jt.clamp(w, min_v=1e-3)
        h = jt.clamp(h, min_v=1e-3)
        cos_a = jt.cos(a)
        sin_a = jt.sin(a)
        R = jt.stack([
            cos_a, -sin_a,
            sin_a, cos_a
        ]).reshape(2, 2)
        D = jt.array([[w * w / 12.0, 0.0],
                       [0.0, h * h / 12.0]])
        Sigma = R.matmul(D).matmul(R.transpose(0, 1))
        mu = jt.array([x, y])
        return mu, Sigma

    def _gwd(self, mu1, S1, mu2, S2):
        """Wasserstein-2 distance between two Gaussians."""
        diff = (mu1 - mu2).reshape(2, 1)
        S2_inv = jt.linalg.inv(S2 + jt.eye(2) * 1e-6)
        tr_term = (S2_inv * S1).sum()
        mid = S2_inv.matmul(S1).matmul(S2_inv)
        det_term = jt.log(jt.linalg.det(S2 + jt.eye(2) * 1e-6) /
                          jt.linalg.det(S1 + jt.eye(2) * 1e-6))
        return tr_term + (diff.transpose(0, 1)).matmul(S2_inv).matmul(diff).sum() - 2 + det_term

    def execute(self, pred, target, weight=None, avg_factor=None):
        """pred/target: [N, 5] OBB (x, y, w, h, angle)."""
        pred = pred.reshape(-1, 5)
        target = target.reshape(-1, 5)
        losses = []
        for i in range(pred.shape[0]):
            mu1, S1 = self._obb_to_gaussian(pred[i])
            mu2, S2 = self._obb_to_gaussian(target[i])
            d2 = self._gwd(mu1, S1, mu2, S2)
            losses.append(jt.clamp(d2, min_v=0.0))
        loss = jt.stack(losses)
        loss = jt.sqrt(loss + 1e-6) / self.tau
        if weight is not None:
            loss = loss * weight.reshape(-1)
        if avg_factor is not None:
            loss = loss.sum() / max(avg_factor, 1.0)
        else:
            loss = loss.mean()
        return loss * self.loss_weight
