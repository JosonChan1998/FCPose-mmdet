# dataset settings
dataset_type = 'CocoKptDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
flip_map = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_label=True, with_kpt=True),
    dict(type='KptRandomCrop',
         crop_size=(0.4, 0.4),
         crop_type='relative_range'),
    dict(
        type='Resize',
        img_scale=[(320, 1333), (800, 1333)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5, flip_map=flip_map),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='HeatmapGenerator', num_keypoints=17),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img',
                               'gt_bboxes',
                               'gt_labels',
                               'gt_keypoints',
                               'gt_kpt_heatmap',
                               'gt_kpt_ignore',
                               'gt_p3_kpt_heatmap',
                               'gt_p3_kpt_ignore',
                               'gt_inst_heatmaps',
                               'gt_kpt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/person_keypoints_train2017.json',
        img_prefix=data_root + 'train2017',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/person_keypoints_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/person_keypoints_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline
))