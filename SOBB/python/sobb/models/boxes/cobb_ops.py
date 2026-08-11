"""COBB (Convex OBB) four-candidate geometry and decode ops.

The convex OBB representation generates four candidate oriented boxes
from a horizontal anchor and a sliding ratio parameter.  Each candidate
corresponds to a different convex-hull vertex ordering of the two
intercept pairs produced by the analytic geometry.

Reference: Xiao et al., "Theoretically Achieving Continuous
Representation of Oriented Bounding Boxes", CVPR 2024.
"""
import jittor as jt
import math


def cobb_candidate_geoms(w, h, S):
    """Generate four candidate OBB geometries from (w, h, S).

    Args:
        w, h: horizontal box dimensions.
        S: occupancy ratio (0 < S < 1).

    Returns:
        list of 4 tuples (obb_w, obb_h, angle).
    """
    W = float(w)
    H = float(h)
    s = float(jt.clamp(S, 1e-6, 1.0 - 1e-6).detach()) if hasattr(S, 'detach') else float(S)
    s = min(max(s, 1e-6), 1.0 - 1e-6)
    C1 = (W * W - H * H) / 4.0
    C2 = W * H * (2.0 * s - 1.0) / 4.0
    if abs(C1) < 1e-6 and abs(C2) < 1e-6:
        return [(W, H, 0.0), (W, H, math.pi / 2),
                (H, W, 0.0), (H, W, math.pi / 2)]
    alpha_sq = (C1 + C2) / (C1 - C2)
    alpha_sq = min(max(alpha_sq, 0.0), 1.0)
    alpha = math.sqrt(alpha_sq)
    beta = 1.0
    ow1 = W / (1.0 + alpha)
    oh1 = H * alpha / (1.0 + alpha)
    ow2 = W / (1.0 + beta)
    oh2 = H * beta / (1.0 + beta)
    angle1 = math.atan2(oh1, ow1)
    angle2 = math.atan2(ow1, oh1)
    angle3 = -angle1
    angle4 = -angle2
    return [
        (max(ow1, 1e-3), max(oh1, 1e-3), angle1),
        (max(ow2, 1e-3), max(oh2, 1e-3), angle2),
        (max(ow1, 1e-3), max(oh1, 1e-3), angle3),
        (max(ow2, 1e-3), max(oh2, 1e-3), angle4),
    ]


def cobb_decode_both(hbb, deltas, means=None, stds=None):
    """Decode all four candidate OBBs from HBB + deltas.

    Args:
        hbb: [N, 4] horizontal box (x1, y1, x2, y2).
        deltas: [N, >=5] (dx, dy, dw, dh, z_s, [s1..s4]).

    Returns:
        (cand_list, scores): list of 4 [N, 5] arrays and [N, 4] scores.
    """
    import jittor.nn as jnn
    if means is None:
        means = jt.array([0.0, 0.0, 0.0, 0.0])
    if stds is None:
        stds = jt.array([1.0, 1.0, 1.0, 1.0])
    dx = deltas[:, 0] * stds[0] + means[0]
    dy = deltas[:, 1] * stds[1] + means[1]
    dw = deltas[:, 2] * stds[2] + means[2]
    dh = deltas[:, 3] * stds[3] + means[3]
    z_s = deltas[:, 4]
    px = (hbb[:, 0] + hbb[:, 2]) / 2.0 + dx * (hbb[:, 2] - hbb[:, 0])
    py = (hbb[:, 1] + hbb[:, 3]) / 2.0 + dy * (hbb[:, 3] - hbb[:, 1])
    gw = jt.clamp((hbb[:, 2] - hbb[:, 0]) * jt.exp(dw), min_v=1e-3)
    gh = jt.clamp((hbb[:, 3] - hbb[:, 1]) * jt.exp(dh), min_v=1e-3)
    t_s = jt.clamp(jt.sigmoid(z_s), 1e-6, 1.0 - 1e-6)
    S = 1.0 - t_s
    cands = []
    for k in range(4):
        c = cobb_candidate_geoms_scalar(gw, gh, S, k)
        obb = jt.stack([
            px, py,
            jt.array(c[0]),
            jt.array(c[1]),
            jt.array(c[2]),
        ], dim=1)
        cands.append(obb)
    scores = jt.softmax(deltas[:, 5:9], dim=1) if deltas.shape[1] >= 9 else \
        jt.ones((deltas.shape[0], 4)) / 4.0
    return cands, scores


def cobb_candidate_geoms_scalar(gw, gh, S, k):
    """Per-element four-candidate geometry for tensor inputs."""
    import math
    C1 = (gw * gw - gh * gh) / 4.0
    C2 = gw * gh * (2.0 * S - 1.0) / 4.0
    alpha_sq = jt.clamp((C1 + C2) / (C1 - C2 + 1e-12), 0.0, 1.0)
    alpha = jt.sqrt(alpha_sq + 1e-12)
    ow1 = gw / (1.0 + alpha)
    oh1 = gh * alpha / (1.0 + alpha)
    ow2 = gw / 2.0
    oh2 = gh / 2.0
    angle1 = jt.atan2(oh1, ow1)
    angles = [angle1, -angle1, angle1 + math.pi / 2, -angle1 - math.pi / 2]
    ows = [ow1, ow1, oh1, oh1]
    ohs = [oh1, oh1, ow1, ow1]
    idx = k % 4
    return (ows[idx], ohs[idx], angles[idx])
