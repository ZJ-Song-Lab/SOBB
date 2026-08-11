import jittor as jt
import math
from sobb.ops.box_iou_rotated import box_iou_rotated

def _candidate_geometry(x, y, W, H):
    """Derive OBB (w, h, angle) from an intercept pair (x, y) on the HBB edges.

    The four inscribed vertices are V1(x,0), V2(0,y), V3(W-x,H), V4(W,H-y).
    Edge vectors: e1 = V2-V1 = (-x, y), e2 = V3-V2 = (W-x, H-y).
    The longer edge defines obb_w and the OBB angle; the shorter edge defines obb_h.
    """
    e1x, e1y = -x, y
    e2x, e2y = W - x, H - y
    L1 = jt.sqrt(e1x ** 2 + e1y ** 2)
    L2 = jt.sqrt(e2x ** 2 + e2y ** 2)
    take_e1 = (L1 >= L2).float()
    obb_w = take_e1 * L1 + (1.0 - take_e1) * L2
    obb_h = take_e1 * L2 + (1.0 - take_e1) * L1
    ang_x = take_e1 * e1x + (1.0 - take_e1) * e2x
    ang_y = take_e1 * e1y + (1.0 - take_e1) * e2y
    angle = jt.atan2(ang_y, ang_x)
    return obb_w, obb_h, angle


def sobb_decode(hbb, sobb_deltas, means=[0., 0., 0., 0.], stds=[1., 1., 1., 1.]):
    """
    Decode SOBB parameter vector to OBB.

    The 7-parameter vector is (dx, dy, dw, dh, z_s, s1, s2).
    The first four follow the standard R-CNN delta protocol (means/stds applied).
    z_s is an unconstrained logit decoded via t_s = clip(sigmoid(z_s), eps, 1-eps),
    giving S = 1 - t_s in [0, 1). s1, s2 are raw candidate-quality logits.

    Parameters
    ----------
    hbb : [N, 4] (x_ctr, y_ctr, w, h) - anchors or proposals HBB
    sobb_deltas : [N, 7] (dx, dy, dw, dh, z_s, s1, s2)
    means, stds : length-4 vectors for HBB delta denormalization only

    Returns
    -------
    obb : [N, 5] (x_ctr, y_ctr, w, h, angle)
    """
    if sobb_deltas.shape[0] == 0:
        return jt.zeros((0, 5))

    means = jt.array(means).view(1, -1)
    stds = jt.array(stds).view(1, -1)

    hbb_deltas = sobb_deltas[:, :4] * stds + means
    z_s = sobb_deltas[:, 4]
    s1 = sobb_deltas[:, 5]
    s2 = sobb_deltas[:, 6]

    dx = hbb_deltas[:, 0]
    dy = hbb_deltas[:, 1]
    dw = hbb_deltas[:, 2]
    dh = hbb_deltas[:, 3]

    # 1. Decode Outer HBB
    px = hbb[:, 0]
    py = hbb[:, 1]
    pw = hbb[:, 2]
    ph = hbb[:, 3]

    gw = jt.exp(dw) * pw
    gh = jt.exp(dh) * ph
    gx = dx * pw + px
    gy = dy * ph + py

    # 2. Decode S (Empty Area Ratio) via sigmoid + clip
    # t_s = clip(sigmoid(z_s), eps, 1-eps); S = 1 - t_s, S in [0, 1)
    eps = 1e-6
    t_s_hat = jt.clamp(jt.sigmoid(z_s), eps, 1.0 - eps)
    S = 1.0 - t_s_hat

    # 3. Analytic candidate solution with sign coupling sigma(C2)
    C1 = (gw**2 - gh**2) / 4.0
    C2 = gw * gh * (2.0 * S - 1.0) / 4.0
    # Explicit degenerate branch: at w=h and S=1/2 (C1=C2=0) the two
    # candidates coalesce exactly; force R=A=B=0 there instead of relying
    # on an epsilon floor (Eq. candidate_coalescence).
    deg_f = ((jt.abs(C1) < 1e-6) & (jt.abs(C2) < 1e-6)).float()
    R = jt.sqrt(jt.maximum(C1**2 + 4.0 * C2**2, 0.0))
    A = jt.sqrt(jt.maximum((R + C1) / 2.0, 0.0))
    B = jt.sqrt(jt.maximum((R - C1) / 2.0, 0.0))
    A = deg_f * 0.0 + (1.0 - deg_f) * A
    B = deg_f * 0.0 + (1.0 - deg_f) * B

    # sigma(C2) = +1 if C2 >= 0, -1 if C2 < 0
    sigma_c2 = (C2 >= 0).float() * 2.0 - 1.0

    # Intercept pairs on HBB edges (Eq. analytic_sol)
    # Candidate 1: (x1, y1) = (w/2 + A, h/2 + sigma(C2)*B)
    # Candidate 2: (x2, y2) = (w/2 - A, h/2 - sigma(C2)*B)
    x1 = gw / 2.0 + A
    y1 = gh / 2.0 + sigma_c2 * B
    x2 = gw / 2.0 - A
    y2 = gh / 2.0 - sigma_c2 * B

    # Derive OBB dimensions and angle from candidate vertex edge vectors.
    # Vertices: V1(x,0), V2(0,y), V3(w-x,h), V4(w,h-y);
    # e1 = V2-V1 = (-x, y), e2 = V3-V2 = (w-x, h-y).
    obb_w1, obb_h1, angle1 = _candidate_geometry(x1, y1, gw, gh)
    obb_w2, obb_h2, angle2 = _candidate_geometry(x2, y2, gw, gh)

    # Selection based on s1, s2 scores (Eq. final_selection)
    mask = (s1 > s2).float()
    obb_w = mask * obb_w1 + (1 - mask) * obb_w2
    obb_h = mask * obb_h1 + (1 - mask) * obb_h2
    angle = mask * angle1 + (1 - mask) * angle2

    return jt.stack([gx, gy, obb_w, obb_h, angle], dim=1)

def sobb_encode(hbb, obb, means=[0., 0., 0., 0.], stds=[1., 1., 1., 1.]):
    """
    Encode GT OBB to SOBB parameter vector relative to HBB.

    The HBB deltas (dx, dy, dw, dh) follow the standard R-CNN protocol and are
    normalized by means/stds (length-4). The shape target is the occupancy ratio
    t_s = 1 - S in (0, 1], NOT a log-space encoding. The candidate scores
    (s1, s2) are the raw IoU values t_sk = IoU(P_k, G) in [0, 1], used directly
    as detached BCE targets (not normalized to sum to one).

    Parameters
    ----------
    hbb : [N, 4] (x_ctr, y_ctr, w, h) - anchors or proposals
    obb : [N, 5] (x_ctr, y_ctr, w, h, angle) - GT OBB
    means, stds : length-4 vectors for HBB delta normalization only

    Returns
    -------
    deltas : [N, 7] (dx, dy, dw, dh, t_s, s1, s2)
    """
    if obb.shape[0] == 0:
        return jt.zeros((0, 7))

    # 1. Get outer HBB of GT OBB
    gx, gy, gw, gh, ga = obb[:, 0], obb[:, 1], obb[:, 2], obb[:, 3], obb[:, 4]

    cos_a = jt.abs(jt.cos(ga))
    sin_a = jt.abs(jt.sin(ga))

    # Outer HBB dimensions of GT
    GW = gw * cos_a + gh * sin_a
    GH = gw * sin_a + gh * cos_a

    # 2. HBB deltas
    px, py, pw, ph = hbb[:, 0], hbb[:, 1], hbb[:, 2], hbb[:, 3]
    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = jt.log(jt.maximum(GW / pw, 1e-6))
    dh = jt.log(jt.maximum(GH / ph, 1e-6))

    # 3. Shape parameter: S = empty area ratio, t_s = 1 - S = occupancy ratio
    S = 1.0 - (gw * gh) / (GW * GH)
    S = jt.clamp(S, 0.0, 1.0 - 1e-6)
    t_s = 1.0 - S

    # 4. IoU-calibrated soft labels for s1, s2
    # Analytic candidates with sign coupling sigma(C2)
    C1 = (GW**2 - GH**2) / 4.0
    C2 = GW * GH * (2.0 * S - 1.0) / 4.0
    # Explicit degenerate branch: at w=h and S=1/2 (C1=C2=0) the two
    # candidates coalesce exactly; force R=x_off=y_off=0 there instead of
    # relying on an epsilon floor (Eq. candidate_coalescence).
    deg_f = ((jt.abs(C1) < 1e-6) & (jt.abs(C2) < 1e-6)).float()
    R = jt.sqrt(jt.maximum(C1**2 + 4.0 * C2**2, 0.0))
    x_off = jt.sqrt(jt.maximum((R + C1) / 2.0, 0.0))
    y_off = jt.sqrt(jt.maximum((R - C1) / 2.0, 0.0))
    x_off = deg_f * 0.0 + (1.0 - deg_f) * x_off
    y_off = deg_f * 0.0 + (1.0 - deg_f) * y_off

    sigma_c2 = (C2 >= 0).float() * 2.0 - 1.0

    # Intercept pairs on HBB edges (Eq. analytic_sol)
    x1_ = GW / 2.0 + x_off
    y1_ = GH / 2.0 + sigma_c2 * y_off
    x2_ = GW / 2.0 - x_off
    y2_ = GH / 2.0 - sigma_c2 * y_off

    # Derive OBB dimensions and angle from candidate vertex edge vectors
    ow1, oh1, a1 = _candidate_geometry(x1_, y1_, GW, GH)
    ow2, oh2, a2 = _candidate_geometry(x2_, y2_, GW, GH)

    # Candidate 1 and 2 (using GT center and candidate-specific dimensions)
    cand1 = jt.stack([gx, gy, ow1, oh1, a1], dim=1)
    cand2 = jt.stack([gx, gy, ow2, oh2, a2], dim=1)

    # Calculate IoU with GT for each candidate.
    # Per Eq. (scoring_target): t_sk = IoU(P_k, G), k in {1,2}.
    # Raw IoU values in [0, 1] are used directly as detached BCE targets;
    # they are NOT normalized to sum to one.
    all_ious1 = box_iou_rotated(cand1, obb)
    all_ious2 = box_iou_rotated(cand2, obb)

    ious1 = jt.diag(all_ious1)
    ious2 = jt.diag(all_ious2)

    s1 = ious1
    s2 = ious2

    # Normalize HBB deltas (first 4 only); t_s, s1, s2 are not normalized
    means = jt.array(means).view(1, -1)
    stds = jt.array(stds).view(1, -1)

    hbb_deltas = jt.stack([dx, dy, dw, dh], dim=1)
    hbb_deltas = (hbb_deltas - means) / stds

    deltas = jt.stack([hbb_deltas[:, 0], hbb_deltas[:, 1], hbb_deltas[:, 2],
                       hbb_deltas[:, 3], t_s, s1, s2], dim=1)
    return deltas



def sobb_candidate_geoms(w, h, S):
    """Generate the two analytic candidate intercept pairs from (w, h, S).

    Implements the analytic solution (Eq. analytic_sol) that maps the HBB
    dimensions and empty-area ratio to two feasible intercept pairs on the
    HBB edges.

    Parameters
    ----------
    w : tensor  HBB width  (any shape)
    h : tensor  HBB height
    S : tensor  empty-area ratio in [0, 1)

    Returns
    -------
    x1, y1, x2, y2 : tensors  intercept pairs (unnormalised, in HBB units)
    """
    C1 = (w ** 2 - h ** 2) / 4.0
    C2 = w * h * (2.0 * S - 1.0) / 4.0
    # Explicit degenerate branch: at w=h and S=1/2 (C1=C2=0) the two
    # candidates coalesce exactly; force R=A=B=0 there instead of relying
    # on an epsilon floor (Eq. candidate_coalescence).
    deg_f = ((jt.abs(C1) < 1e-6) & (jt.abs(C2) < 1e-6)).float()
    R = jt.sqrt(jt.maximum(C1 ** 2 + 4.0 * C2 ** 2, 0.0))
    A = jt.sqrt(jt.maximum((R + C1) / 2.0, 0.0))
    B = jt.sqrt(jt.maximum((R - C1) / 2.0, 0.0))
    A = deg_f * 0.0 + (1.0 - deg_f) * A
    B = deg_f * 0.0 + (1.0 - deg_f) * B
    sigma_c2 = (C2 >= 0).float() * 2.0 - 1.0

    x1 = w / 2.0 + A
    y1 = h / 2.0 + sigma_c2 * B
    x2 = w / 2.0 - A
    y2 = h / 2.0 - sigma_c2 * B
    return x1, y1, x2, y2


def dbbox_decode(anchors, sobb_deltas, means=(0., 0., 0., 0.), stds=(1., 1., 1., 1.)):
    """Decode SOBB deltas from rotated anchors to OBB.

    Convenience wrapper that converts rotated anchors (5-param) to their
    axis-aligned HBB (4-param) and then calls :func:`sobb_decode`.

    Parameters
    ----------
    anchors : [N, 5]  (x_ctr, y_ctr, w, h, angle) rotated anchors / proposals
    sobb_deltas : [N, 7]  (dx, dy, dw, dh, z_s, s1, s2)
    means, stds : length-4 vectors for HBB delta denormalisation

    Returns
    -------
    obb : [N, 5]  (x_ctr, y_ctr, w, h, angle)
    """
    if anchors.shape[0] == 0:
        return jt.zeros((0, 5))

    x = anchors[:, 0]
    y = anchors[:, 1]
    w = anchors[:, 2]
    h = anchors[:, 3]
    a = anchors[:, 4]
    cos_a = jt.abs(jt.cos(a))
    sin_a = jt.abs(jt.sin(a))
    W = w * cos_a + h * sin_a
    H = w * sin_a + h * cos_a
    hbb = jt.stack([x, y, W, H], dim=1)

    return sobb_decode(hbb, sobb_deltas, means=list(means), stds=list(stds))



def sobbb_decode_both(hbb, sobb_deltas, means=[0., 0., 0., 0.], stds=[1., 1., 1., 1.]):
    """Decode BOTH candidate OBBs without selection.

    Like :func:`sobb_decode` but returns both analytic candidates instead of
    selecting one via s1/s2.  Used by the risk-aware pairwise scoring loss.

    Parameters
    ----------
    hbb : [N, 4] (x_ctr, y_ctr, w, h)
    sobb_deltas : [N, >=5] (only first 5 used: dx, dy, dw, dh, z_s)
    means, stds : length-4 vectors for HBB delta denormalization

    Returns
    -------
    cand1, cand2 : each [N, 5] (x_ctr, y_ctr, w, h, angle)
    """
    if sobb_deltas.shape[0] == 0:
        z = jt.zeros((0, 5))
        return z, z

    means = jt.array(means).view(1, -1)
    stds = jt.array(stds).view(1, -1)

    hbb_deltas = sobb_deltas[:, :4] * stds + means
    z_s = sobb_deltas[:, 4]

    dx = hbb_deltas[:, 0]
    dy = hbb_deltas[:, 1]
    dw = hbb_deltas[:, 2]
    dh = hbb_deltas[:, 3]

    px = hbb[:, 0]
    py = hbb[:, 1]
    pw = hbb[:, 2]
    ph = hbb[:, 3]

    gw = jt.exp(dw) * pw
    gh = jt.exp(dh) * ph
    gx = dx * pw + px
    gy = dy * ph + py

    eps = 1e-6
    t_s_hat = jt.clamp(jt.sigmoid(z_s), eps, 1.0 - eps)
    S = 1.0 - t_s_hat

    C1 = (gw**2 - gh**2) / 4.0
    C2 = gw * gh * (2.0 * S - 1.0) / 4.0
    # Explicit degenerate branch: at w=h and S=1/2 (C1=C2=0) the two
    # candidates coalesce exactly; force R=A=B=0 there instead of relying
    # on an epsilon floor (Eq. candidate_coalescence).
    deg_f = ((jt.abs(C1) < 1e-6) & (jt.abs(C2) < 1e-6)).float()
    R = jt.sqrt(jt.maximum(C1**2 + 4.0 * C2**2, 0.0))
    A = jt.sqrt(jt.maximum((R + C1) / 2.0, 0.0))
    B = jt.sqrt(jt.maximum((R - C1) / 2.0, 0.0))
    A = deg_f * 0.0 + (1.0 - deg_f) * A
    B = deg_f * 0.0 + (1.0 - deg_f) * B
    sigma_c2 = (C2 >= 0).float() * 2.0 - 1.0

    x1 = gw / 2.0 + A
    y1 = gh / 2.0 + sigma_c2 * B
    x2 = gw / 2.0 - A
    y2 = gh / 2.0 - sigma_c2 * B

    obb_w1, obb_h1, angle1 = _candidate_geometry(x1, y1, gw, gh)
    obb_w2, obb_h2, angle2 = _candidate_geometry(x2, y2, gw, gh)

    cand1 = jt.stack([gx, gy, obb_w1, obb_h1, angle1], dim=1)
    cand2 = jt.stack([gx, gy, obb_w2, obb_h2, angle2], dim=1)
    return cand1, cand2


def reconstruct_gt_obb(anchor_hbb, bbox_target, means=[0., 0., 0., 0.], stds=[1., 1., 1., 1.]):
    """Reconstruct GT OBB from encoded 7-param target + anchor HBB.

    The target (dx, dy, dw, dh, t_s, s1_t, s2_t) encodes the GT OBB relative
    to the anchor HBB.  Since the GT OBB is one of the two analytic candidates
    by construction, reconstruction is exact: decode the GT outer HBB and S
    from the target, compute both candidates, then select the one with the
    higher target IoU (s1_t vs s2_t).

    Parameters
    ----------
    anchor_hbb : [N, 4] (x_ctr, y_ctr, w, h)
    bbox_target : [N, 7] (dx, dy, dw, dh, t_s, s1_t, s2_t)
    means, stds : length-4 vectors for HBB delta denormalization

    Returns
    -------
    gt_obb : [N, 5] (x_ctr, y_ctr, w, h, angle)
    """
    if bbox_target.shape[0] == 0:
        return jt.zeros((0, 5))

    means = jt.array(means).view(1, -1)
    stds = jt.array(stds).view(1, -1)

    hbb_deltas = bbox_target[:, :4] * stds + means
    t_s = bbox_target[:, 4]
    s1_t = bbox_target[:, 5]
    s2_t = bbox_target[:, 6]

    dx = hbb_deltas[:, 0]
    dy = hbb_deltas[:, 1]
    dw = hbb_deltas[:, 2]
    dh = hbb_deltas[:, 3]

    px = anchor_hbb[:, 0]
    py = anchor_hbb[:, 1]
    pw = anchor_hbb[:, 2]
    ph = anchor_hbb[:, 3]

    gw = jt.exp(dw) * pw
    gh = jt.exp(dh) * ph
    gx = dx * pw + px
    gy = dy * ph + py

    S = 1.0 - t_s

    C1 = (gw**2 - gh**2) / 4.0
    C2 = gw * gh * (2.0 * S - 1.0) / 4.0
    # Explicit degenerate branch: at w=h and S=1/2 (C1=C2=0) the two
    # candidates coalesce exactly; force R=A=B=0 there instead of relying
    # on an epsilon floor (Eq. candidate_coalescence).
    deg_f = ((jt.abs(C1) < 1e-6) & (jt.abs(C2) < 1e-6)).float()
    R = jt.sqrt(jt.maximum(C1**2 + 4.0 * C2**2, 0.0))
    A = jt.sqrt(jt.maximum((R + C1) / 2.0, 0.0))
    B = jt.sqrt(jt.maximum((R - C1) / 2.0, 0.0))
    A = deg_f * 0.0 + (1.0 - deg_f) * A
    B = deg_f * 0.0 + (1.0 - deg_f) * B
    sigma_c2 = (C2 >= 0).float() * 2.0 - 1.0

    x1 = gw / 2.0 + A
    y1 = gh / 2.0 + sigma_c2 * B
    x2 = gw / 2.0 - A
    y2 = gh / 2.0 - sigma_c2 * B

    obb_w1, obb_h1, angle1 = _candidate_geometry(x1, y1, gw, gh)
    obb_w2, obb_h2, angle2 = _candidate_geometry(x2, y2, gw, gh)

    mask = (s1_t > s2_t).float()
    obb_w = mask * obb_w1 + (1 - mask) * obb_w2
    obb_h = mask * obb_h1 + (1 - mask) * obb_h2
    angle = mask * angle1 + (1 - mask) * angle2

    return jt.stack([gx, gy, obb_w, obb_h, angle], dim=1)


def compute_risk_aware_inputs(cand1, cand2, gt_obb):
    """Compute quality targets q and candidate separation D for risk-aware loss.

    q_k = IoU(sg(P_k), G) (detached), D = 1 - IoU(sg(P1), sg(P2)) (detached).

    Parameters
    ----------
    cand1, cand2 : [N, 5] predicted candidate OBBs
    gt_obb : [N, 5] ground-truth OBBs

    Returns
    -------
    q : [N, 2] quality targets (q1, q2), detached
    D : [N] candidate separation, detached
    """
    cand1_d = cand1.detach()
    cand2_d = cand2.detach()
    q1 = jt.diag(box_iou_rotated(cand1_d, gt_obb))
    q2 = jt.diag(box_iou_rotated(cand2_d, gt_obb))
    q = jt.stack([q1, q2], dim=1)
    D = 1.0 - jt.diag(box_iou_rotated(cand1_d, cand2_d))
    return q, D


def _log_softmax(x, dim=1):
    """Numerically stable log-softmax along dim."""
    x_max = jt.max(x, dim=dim, keepdims=True)
    x = x - x_max
    return x - jt.log(jt.sum(jt.exp(x), dim=dim, keepdims=True))


def risk_aware_score_loss(ell, q, D, tau_q=0.10, tau_s=1.0, eta=0.05, mu=1.0,
                          lambda_cal=1.0, lambda_pair=1.0, lambda_margin=1.0):
    """Three-component risk-aware pairwise scoring loss.

    L_score = lambda_cal * L_cal + lambda_pair * L_pair + lambda_margin * L_margin

    * Calibration:   L_cal = sum_k BCE(sigmoid(ell_k), q_k)
    * Pairwise:      L_pair = KL(pi* || pi)  with pi*_k = softmax(q/tau_q), pi_k = softmax(ell/tau_s)
    * Risk margin:   L_margin = 1[g>eta] * [mu*D*g - y*(ell1-ell2)]_+
      where g = |q1-q2|, y = sign(q1-q2), D = 1-IoU(P1,P2)

    Parameters
    ----------
    ell : [N, 2] raw scorer logits (ell_1, ell_2) -- receives gradients
    q   : [N, 2] quality targets (q1, q2) -- detached inside
    D   : [N]   candidate separation -- detached inside

    Returns
    -------
    loss : [N] per-sample total scoring loss
    """
    q = q.detach()
    D = D.detach()

    # 1. Calibration: BCE(sigmoid(ell_k), q_k) summed over k
    max_val = jt.clamp(-ell, min_v=0)
    bce = (1 - q) * ell + max_val + jt.log(
        jt.exp(-max_val) + jt.exp(-ell - max_val))
    l_cal = bce.sum(dim=1)

    # 2. Pairwise distribution: KL(pi* || pi)
    log_pi_star = _log_softmax(q / tau_q, dim=1)
    log_pi = _log_softmax(ell / tau_s, dim=1)
    pi_star = jt.exp(log_pi_star)
    l_pair = (pi_star * (log_pi_star - log_pi)).sum(dim=1)

    # 3. Risk-aware margin: 1[g>eta] * [mu*D*g - y*(ell1-ell2)]_+
    q1, q2 = q[:, 0], q[:, 1]
    ell1, ell2 = ell[:, 0], ell[:, 1]
    g = jt.abs(q1 - q2)
    y = (q1 > q2).float() - (q2 > q1).float()
    margin = mu * D * g - y * (ell1 - ell2)
    l_margin = (g > eta).float() * jt.clamp(margin, min_v=0)

    return lambda_cal * l_cal + lambda_pair * l_pair + lambda_margin * l_margin


def _polygon_distance(obb_a, obb_b):
    """Approximate polygon distance between two OBBs via corner L2.

    obb_a, obb_b : [N, 5] (x, y, w, h, angle)
    Returns : [N] mean corner distance
    """
    def _corners(obb):
        cx, cy, w, h, a = obb[:, 0], obb[:, 1], obb[:, 2], obb[:, 3], obb[:, 4]
        cos_a = jt.cos(a)
        sin_a = jt.sin(a)
        dx = jt.stack([-w / 2, w / 2, w / 2, -w / 2], dim=1)
        dy = jt.stack([-h / 2, -h / 2, h / 2, h / 2], dim=1)
        rx = dx * cos_a.unsqueeze(1) - dy * sin_a.unsqueeze(1) + cx.unsqueeze(1)
        ry = dx * sin_a.unsqueeze(1) + dy * cos_a.unsqueeze(1) + cy.unsqueeze(1)
        return jt.stack([rx, ry], dim=2)

    c_a = _corners(obb_a)
    c_b = _corners(obb_b)
    dist = jt.sqrt(((c_a - c_b) ** 2).sum(dim=2)).mean(dim=1)
    return dist


def consistency_loss(logits_orig, logits_pert, cand_orig_1, cand_orig_2,
                     cand_pert_1, cand_pert_2, tau_s=1.0, lambda_cons=1.0):
    # S0-5: Accept lambda_cons as a scaling parameter for the consistency
    # loss. This is multiplied with the per-sample KL divergence before
    # returning, so callers can control the weight via config.
    """Candidate-set consistency loss (paper Eq 34-35).

    L_cons = KL(sg(pi(x)) || rho^dag pi(T(x)))

    rho^dag minimizes total polygon distance over the two permutations
    of the candidate pair from the perturbed view.

    Parameters
    ----------
    logits_orig : [N, 2] scorer logits from original view (detached)
    logits_pert : [N, 2] scorer logits from perturbed view (receives grad)
    cand_orig_1, cand_orig_2 : [N, 5] OBB candidates from original view
    cand_pert_1, cand_pert_2 : [N, 5] OBB candidates from perturbed view

    Returns
    -------
    loss : [N] per-sample consistency loss
    """
    log_pi_orig = _log_softmax(logits_orig.detach() / tau_s, dim=1)
    log_pi_pert = _log_softmax(logits_pert / tau_s, dim=1)
    pi_orig = jt.exp(log_pi_orig)

    d_id = _polygon_distance(cand_orig_1, cand_pert_1) + \
           _polygon_distance(cand_orig_2, cand_pert_2)
    d_swap = _polygon_distance(cand_orig_1, cand_pert_2) + \
             _polygon_distance(cand_orig_2, cand_pert_1)

    use_swap = (d_swap < d_id).float()

    log_pi_pert_0 = log_pi_pert[:, 0]
    log_pi_pert_1 = log_pi_pert[:, 1]
    log_matched_0 = use_swap * log_pi_pert_1 + (1 - use_swap) * log_pi_pert_0
    log_matched_1 = use_swap * log_pi_pert_0 + (1 - use_swap) * log_pi_pert_1
    log_pi_matched = jt.stack([log_matched_0, log_matched_1], dim=1)

    kl = (pi_orig * (log_pi_orig - log_pi_matched)).sum(dim=1)
    kl = jt.clamp(kl, min_v=0.0)
    return kl * lambda_cons
