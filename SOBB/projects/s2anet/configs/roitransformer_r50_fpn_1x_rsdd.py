seed = 42  # Set to {1,2,3,4,5} for five-seed experiments
# model settings
# Two-stage RoI Transformer: RPN (HBB) -> BBoxHeadRbbox (HBB->OBB) -> SOBBHead (final SOBB)
model = dict(
    type='RoITransformer',
    backbone=dict(
        type='Resnet50',
        frozen_stages=1,
        return_stages=["layer1","layer2","layer3","layer4"],
        pretrained=True),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs="on_input",
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[4, 8, 16, 32],
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=(.0, .0, .0, .0),
            target_stds=(1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0),
        loss_bbox=dict(
            type='L1Loss', loss_weight=1.0),
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='roi_align', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='ConvFCBBoxHeadRbbox',
        num_classes=2,
        in_channels=256 * 7 * 7,
        roi_feat_size=7,
        num_shared_convs=0,
        num_shared_fcs=2,
        num_cls_convs=0,
        num_cls_fcs=0,
        num_reg_convs=0,
        num_reg_fcs=0,
        fc_out_channels=1024,
        reg_class_agnostic=False,
        target_means=(0., 0., 0., 0., 0.),
        target_stds=(1., 1., 1., 1., 1.),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    rbbox_roi_extractor=dict(
        type='OrientedSingleRoIExtractor',
        roi_layer=dict(type='roi_align_rotated', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32],
        extend_factor=(1.4, 1.4)),
    rbbox_head=dict(
        type='SOBBHead',
        num_classes=2,
        in_channels=256,
        roi_feat_size=7,
        num_shared_fcs=2,
        fc_out_channels=1024,
        reg_class_agnostic=False,
        target_means=(0., 0., 0., 0.),
        target_stds=(1., 1., 1., 1.),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_sobb_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0),
        scorer_mode='scorer_only',
        lambda_cons=0.0,
        loss_cal_weight=1.0,
        loss_pair_weight=1.0,
        loss_margin_weight=1.0),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='BboxOverlaps2D')),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
            min_bbox_size=0,
            nms_pre=2000),
        rpn_proposal=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_thr=0.7),
            max_per_img=2000),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.4,
                    min_pos_iou=0,
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='BboxOverlaps2D')),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                bbox_coder=dict(
                    type='DeltaXYWHRBBoxCoder',
                    target_means=(0., 0., 0., 0., 0.),
                    target_stds=(1., 1., 1., 1., 1.)),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='SALAAssigner',
                    topk=9,
                    alpha=1.0,
                    beta=1.0,
                    iou_calculator=dict(type='BboxOverlaps2D_rotated')),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                bbox_coder=dict(
                    type='SOBBBBoxCoder',
                    target_means=(0., 0., 0., 0.),
                    target_stds=(1., 1., 1., 1.)),
                allowed_border=-1,
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_thr=0.7),
            max_per_img=2000),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms_rotated', iou_thr=0.1),
            max_per_img=2000)))
dataset = dict(
    train=dict(
        type="RSDDDataset",
        dataset_dir='/path/to/your/processed_RSDD/train_800',
        transforms=[
            dict(
                type="ContinuousResize",
                scale_range=(0.8, 1.2),
                base_size=800,
                max_size=1333
            ),
            dict(type='RotatedRandomFlip', prob=0.5, direction='horizontal'),
            dict(type='RotatedRandomFlip', prob=0.5, direction='vertical'),
            dict(type='RandomRotateRange', max_deg=30.0, prob=0.5),
            dict(
                type = "Pad",
                size_divisor=32),
            dict(
                type = "Normalize",
                mean =  [123.675, 116.28, 103.53],
                std = [58.395, 57.12, 57.375],
                to_bgr=False,)

        ],
        batch_size=4,
        num_workers=4,
        shuffle=True,
        filter_empty_gt=False
    ),
    val=dict(
        type="RSDDDataset",
        dataset_dir='/path/to/your/processed_RSDD/val_800',
        transforms=[
            dict(
                type="RotatedResize",
                min_size=800,
                max_size=800
            ),
            dict(
                type = "Pad",
                size_divisor=32),
            dict(
                type = "Normalize",
                mean =  [123.675, 116.28, 103.53],
                std = [58.395, 57.12, 57.375],
                to_bgr=False),
        ],
        batch_size=4,
        num_workers=4,
        shuffle=False
    )
)

optimizer = dict(
    type='SGD',
    lr=0.0025,
    momentum=0.9,
    weight_decay=0.0001,
    grad_clip=dict(
        max_norm=35,
        norm_type=2))

scheduler = dict(
    type='StepLR',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    milestones=[24, 33])


logger = dict(
    type="RunLogger")

max_epoch = 36
# eval_interval : epoch interval
eval_interval = 1
# ckpt_interval : epoch interval
checkpoint_interval = 1
# log_interval : iter interval
log_interval = 50
