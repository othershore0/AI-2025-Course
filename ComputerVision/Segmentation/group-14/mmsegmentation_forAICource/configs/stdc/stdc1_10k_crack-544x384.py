_base_ = [
    '../_base_/models/stdc.py', '../_base_/datasets/data.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_8k.py'
]
crop_size = (544, 384)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        num_classes=2,
    )
)
param_scheduler = [
    dict(type='LinearLR', by_epoch=False, start_factor=0.1, begin=0, end=8000),
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=10,
        end=8000,
        by_epoch=False,
    )
]

vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(

    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer',save_dir='work_dirs/vis_results')
