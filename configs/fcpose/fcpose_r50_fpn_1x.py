_base_ = [
    '../_base_/datasets/coco_kpt.py',
    '../_base_/default_runtime.py'
]

model=dict(
    type='FCPose',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCPoseHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=True,
        center_sample_radius=1.5,
        centerness_on_reg=True,
        norm_on_bbox=True,
        conv_bias=True,
        num_keypoints=17,
        p3_hm_feat_stride=8,
        p1_hm_feat_stride=2,
        refine_levels=[0, 1, 2],
        stacked_convs_share=4,
        feat_channels_share=128,
        stacked_convs_kpt_head=3,
        feat_channels_kpt_head=32,
        stacked_convs_hm=2,
        feat_channels_hm=128,
        max_proposal_per_img=70,
        with_hm_loss=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_mse=dict(
            type='MSELoss',
            loss_weight=0.2)
    ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100)
)

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=1.0, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[60000, 80000])
# Runner type
runner = dict(type='IterBasedRunner', max_iters=90000)
checkpoint_config = dict(interval=10000, max_keep_ckpts=5)
evaluation = dict(interval=5000, metric=['keypoints', 'bbox'])