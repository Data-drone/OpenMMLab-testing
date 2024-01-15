
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
    backend='petrel',
    path_mapping=dict(
        {{'{args.data_root}': '{args.data_bucket}'}}
    )
)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
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
)

test_evaluator = val_evaluator

# We can override details with this
mlflow_backend = dict(type='MLflowVisBackend',
                     tracking_uri='databricks',
                     exp_name=experiment_name)

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