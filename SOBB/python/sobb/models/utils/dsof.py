import jittor as jt
from jittor import nn
from sobb.utils.registry import BRICKS

@BRICKS.register_module()
class DSOF(nn.Module):
    """Decoupled Shape-Orientation Feature (DSOF) Module.

    Routes (rather than statistically disentangles) input features into two
    task-specific streams through architectural separation and independent
    optimization objectives. The module does not impose feature orthogonality
    and does not claim statistically independent features.

    1. Global Scale-Aware Projection (channel attention) for shape regression
       (t_w, t_h, t_s), which benefits from global context and is robust to
       the multiplicative speckle noise of SAR imagery.
    2. Local Alignment-Aware Projection (spatial attention) for alignment
       regression (t_x, t_y), which relies on fine-grained boundary
       and structural-edge cues. The alignment stream also supplies features
       to the shared candidate-conditioned scorer (Eq. shared_scorer).
    """
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(DSOF, self).__init__()
        self.in_channels = in_channels

        # --- Global Scale-Aware Projection (Channel Attention) ---
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.channel_excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

        # --- Local Alignment-Aware Projection (Spatial Attention) ---
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.Sigmoid()
        )

    def execute(self, x):
        # x shape: [N, C, H, W]
        N, C, H, W = x.shape

        # 1. Global Scale-Aware Stream (F_scale)
        z = self.gap(x).view(N, C)
        channel_weights = self.channel_excitation(z).view(N, C, 1, 1)
        f_scale = x * channel_weights

        # 2. Local Alignment-Aware Stream (F_align)
        # Average pooling and Max pooling along channel dimension
        avg_out = jt.mean(x, dim=1, keepdims=True)
        max_out = jt.max(x, dim=1, keepdims=True)
        spatial_in = jt.concat([avg_out, max_out], dim=1)
        spatial_weights = self.spatial_attention(spatial_in)
        f_align = x * spatial_weights

        return f_scale, f_align


@BRICKS.register_module()
class CandidateScorer(nn.Module):
    """Shared permutation-equivariant candidate scorer for single-stage heads.

    Implements Eq. (shared_scorer):
        s_hat_k = q_phi(F_align, Emb(P_k)),  k in {1,2}

    The same embedding network Emb(.) and scorer q_phi are used for both
    candidates.  The scorer is therefore evaluated twice with shared weights
    rather than implemented as two index-specific output channels.  Permuting
    the two candidates only permutes their scores, which is the
    permutation-equivariance property required by the paper.

    Parameters
    ----------
    feat_channels : int
        Channel dimension of F_align (spatial attention feature from DSOF).
    emb_dim : int
        Dimensionality of the candidate-geometry embedding.
    """
    def __init__(self, feat_channels, emb_dim=64):
        super(CandidateScorer, self).__init__()
        self.feat_channels = feat_channels
        self.emb_dim = emb_dim

        # Emb: encodes 4-D candidate descriptor (a/w, b/h, log(w/h), S)
        self.emb = nn.Sequential(
            nn.Linear(4, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
        )

        # q_phi: combines F_align and Emb(P_k) -> single score per location
        self.q_phi = nn.Sequential(
            nn.Conv2d(feat_channels + emb_dim, feat_channels, 1),
            nn.ReLU(),
            nn.Conv2d(feat_channels, 1, 1),
        )

    def _score_single(self, f_align, cand_norm):
        """Score a single candidate with shared weights."""
        N, C, H, W = f_align.shape
        flat = cand_norm.reshape(N * H * W, 4)
        emb_k = self.emb(flat)
        emb_k = emb_k.reshape(N, H, W, self.emb_dim)
        emb_k = emb_k.permute(0, 3, 1, 2)
        combined = jt.concat([f_align, emb_k], dim=1)
        return self.q_phi(combined)

    def execute(self, f_align, candidates):
        """Score both candidates with the *same* q_phi and Emb.

        Parameters
        ----------
        f_align : [N, C, H, W]
        candidates : [N, 2, H, W, 4]  4-D descriptors (a/w, b/h, log(w/h), S)

        Returns
        -------
        scores : [N, 2, H, W]
        """
        s1 = self._score_single(f_align, candidates[:, 0])
        s2 = self._score_single(f_align, candidates[:, 1])
        return jt.concat([s1, s2], dim=1)


@BRICKS.register_module()
class CandidateScorerFC(nn.Module):
    """FC-based shared candidate scorer for two-stage RoI heads.

    Same permutation-equivariant design as CandidateScorer but operates on
    pooled feature vectors instead of spatial feature maps.
    """
    def __init__(self, fc_out_channels, emb_dim=64):
        super(CandidateScorerFC, self).__init__()
        self.fc_out_channels = fc_out_channels
        self.emb_dim = emb_dim

        self.emb = nn.Sequential(
            nn.Linear(4, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
        )

        self.q_phi = nn.Sequential(
            nn.Linear(fc_out_channels + emb_dim, fc_out_channels),
            nn.ReLU(),
            nn.Linear(fc_out_channels, 1),
        )

    def _score_single(self, f_align_flat, cand_norm):
        emb_k = self.emb(cand_norm)
        combined = jt.concat([f_align_flat, emb_k], dim=1)
        return self.q_phi(combined)

    def execute(self, f_align_flat, candidates):
        """Score both candidates with the *same* q_phi and Emb.

        Parameters
        ----------
        f_align_flat : [N, fc_out_channels]
        candidates : [N, 2, 4]  4-D descriptors (a/w, b/h, log(w/h), S)

        Returns
        -------
        scores : [N, 2]
        """
        s1 = self._score_single(f_align_flat, candidates[:, 0])
        s2 = self._score_single(f_align_flat, candidates[:, 1])
        return jt.concat([s1, s2], dim=1)
