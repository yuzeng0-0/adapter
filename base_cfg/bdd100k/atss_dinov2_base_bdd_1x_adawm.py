# model settings
import platform
model = dict(
    type='ATSS',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='Dinov2VisionTransformer',
        arch='base',
        img_size=1024,
        patch_size=16,
        layer_scale_init_value=1e-5,
        out_type = 'featmap',
        init_cfg=dict(type='Pretrained', checkpoint='data/pretrain_models/changed_vit-base-p14to16_dinov2-pre_3rdparty_20230426-ba246503.pth')
    ),
     neck=dict(
        type='SimpleFeaturePyramid',
        in_channels = 768,
        scale_factors=(2.0, 1.0, 0.5,0.25),   # 小于1
        out_channels=256,
        top_block = dict(type='MaxPool',num_levels=1),
        norm_cfg=dict(type='LN'),
        num_outs=5,
        ),
    bbox_head=dict(
        type='ATSSHead',
        num_classes=9,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

backend_args = None
    
dataset_type = "BDDDataset"  # pylint: disable=invalid-name
classes = ["pedestrian", "rider", "car", "bus", "truck", "bicycle", 
    "motorcycle", "traffic light", "traffic sign"]
data_root = "data/bdd100k/"  # pylint: disable=invalid-name


train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", scale=(1280, 720), keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=(1024, 1024),
        allow_negative_crop=True),
    
    dict(type="RandomFlip", prob=0.5),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1280, 720), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape','attributes',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='annotations/det_train.json',
        data_prefix=dict(img='images/100k/train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=24),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='annotations/det_val.json',
        data_prefix=dict(img='images/100k/val/'),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='BDDMetric',
    ann_file=data_root + 'annotations/det_val.json',
    metric='bbox',
    format_only=False)
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# optimizer
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001 / 4, weight_decay=0.1,betas=(0.9, 0.999)),
    constructor='VitDetLayerDecayOptimizerConstructor', 
    paramwise_cfg=dict(
                        num_layers=12, 
                        layer_decay_rate=0.7,
                         layer_sep=[0, 1, 2],
                        custom_keys={
                            'bias': dict(decay_multi=0.),
                            'pos_embed': dict(decay_mult=0.),
                            'relative_position_bias_table': dict(decay_mult=0.),
                            'norm': dict(decay_mult=0.),
                            "rel_pos_h": dict(decay_mult=0.),
                            "rel_pos_w": dict(decay_mult=0.),
                            }
                        )
    )

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)

default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook',  max_keep_ckpts=1,interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    # visualization=dict(type='DetVisualizationHook')
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# vis_backends = [dict(type='LocalVisBackend')]
# visualizer = dict(
#     type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False
