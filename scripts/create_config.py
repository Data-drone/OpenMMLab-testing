
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='setup config file')
    parser.add_argument('--experiment-name', help='mlflow_experiment_name')
    parser.add_argument('--data-root', help='root dir for data')
    parser.add_argument('--config-path', help='root of the folder to put the config in')

    args = parser.parse_args()
    return args

def main():

    args = parse_args()

    databricks_cfg = f"""

_base_ = './rtmdet_tiny_8xb32-300e_coco.py'

experiment_name = '{args.experiment_name}'
data_root = '{args.data_root}'

train_cfg = dict(
    max_epochs=3, 
    type='EpochBasedTrainLoop', 
    val_interval=1)

# I think I can overrule this from the launch command?
# data_root = 'data/coco'

val_evaluator = dict(
    collect_device='gpu'
)

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