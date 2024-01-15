
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='setup config file')
    parser.add_argument('--experiment-name', help='mlflow_experiment_name')
    parser.add_argument('--data-bucket', help='s3 bucket for dataset')
    parser.add_argument('--data-root', help='root dir for data')
    parser.add_argument('--config-path', help='root of the folder to put the config in')

    args = parser.parse_args()
    return args

def main():

    """
    This script sets up training on MosaicML Infrastructure
    The train loop will read from s3
    """

    args = parse_args()

    databricks_cfg = f"""

_base_ = './rtmdet_tiny_8xb32-300e_coco.py'

experiment_name = '{args.experiment_name}'
data_root = '{args.data_root}'

backend_args = dict(
    _delete_ = True,
    backend='ceph'
)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='CachedMosaic',
        img_scale=(640, 640),
        pad_val=114.0,
        max_cached_images=20,
        random_pop=False),
    dict(
        type='RandomResize',
        scale=(1280, 1280),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='CachedMixUp',
        img_scale=(640, 640),
        ratio_range=(1.0, 1.0),
        max_cached_images=10,
        random_pop=False,
        pad_val=(114, 114, 114),
        prob=0.5),
    dict(type='PackDetInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=(640, 640),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=280,
        switch_pipeline=train_pipeline_stage2)
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

train_cfg = dict(
    max_epochs=3, 
    type='EpochBasedTrainLoop', 
    val_interval=1)

val_evaluator = dict(
    collect_device='gpu',
    ann_file=data_root + 'annotations/instances_val2017.json',
    backend_args=backend_args
)

test_evaluator = val_evaluator

# We can override details with this
mlflow_backend = dict(type='MLflowVisBackend',
                     tracking_uri='databricks',
                     exp_name='{args.experiment_name}')

default_hooks = dict(
    logger=dict(type='LoggerHook',
                backend_args=mlflow_backend, 
                interval=50))

vis_backends = [dict(type='LocalVisBackend'),
                mlflow_backend]

visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

"""

    config=f'{args.config_path}/configs/rtmdet/rtmdet_databricks_coco.py'
    with open(config, 'w') as f:
        f.write(databricks_cfg)


if __name__ == '__main__':
    main()