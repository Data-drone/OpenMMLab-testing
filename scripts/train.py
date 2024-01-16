# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction, ConfigDict
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

# hacking backends
from mmengine.fileio.backends import register_backend, PetrelBackend

from mmdet.utils import setup_cache_size_limit_of_dynamo
import logging

logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Format of log messages
    datefmt='%Y-%m-%d %H:%M:%S',  # Date format
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--db-host',
        type=str,
        default='none',
        help='Databricks host for mlflow logging'
    )
    parser.add_argument(
        '--db-token',
        type=str,
        default='none',
        help='Databricks token for mlflow logging'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='none',
        help='Root location for data'
    )
    # we need to make to pytorch for db
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='pytorch',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # hack to stop petrel_backend
    register_backend('petrel', PetrelBackend, prefixes=['petrel'], force=True)

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # mlflow settings
    if args.db_host != 'none':
        os.environ['DATABRICKS_HOST'] = args.db_host

    if args.db_token != 'none':
        os.environ['DATABRICKS_TOKEN'] = args.db_token

    if (args.db_host == 'none' and args.db_token != 'none') or \
    (args.db_token == 'none' and args.db_host != 'none'): 
        logger.warning('Missing either host or token config, mlflow integration may not work')        

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # Setup the data root diectory
    def update_data_root(cfg, new_value):
        for k, v in cfg.items():
            if isinstance(v, ConfigDict):
                update_data_root(cfg[k], new_value)
            if isinstance(v, str) and k == 'data_root':
                #print(f'{k} {v}')
                cfg[k] = v.replace(v, new_value)

    if args.data_root != 'none':
        update_data_root(cfg, args.data_root)
        cfg.val_evaluator.ann_file = f'{args.data_root}/annotations/instances_val2017.json'
        cfg.test_evaluator.ann_file = f'{args.data_root}/annotations/instances_val2017.json'

    # enable automatic-mixed-precision training
    if args.amp is True:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()