_base_ = [
    './fcpose_r50_fpn_1x.py'
]

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[180000, 240000])
# Runner type
runner = dict(type='IterBasedRunner', max_iters=270000)