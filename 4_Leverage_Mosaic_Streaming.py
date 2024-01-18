# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Build OpenMMNet Trainloop  
# MAGIC We will setup our train loop here

# COMMAND ----------

# MAGIC %pip install mosaicml-streaming

# COMMAND ----------

# MAGIC %md # Setup and Config

# COMMAND ----------

# DBTITLE 1, Setup Dataset Folder
import os
import logging

logger = logging.getLogger(__name__)
username = spark.sql("SELECT current_user()").first()['current_user()']

# Get current workspace host and token
browser_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()
db_host = f"https://{browser_host}"
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# Existing MosaicML Configuration
databricks_config = './configs/rtmdet/rtmdet_databricks_coco.py'

s3_path = 's3://db-brian-mosaic/coco-data-streaming'

# COMMAND ----------

dbutils.fs.ls(s3_path)

# COMMAND ----------

# MAGIC %md ## StreamingCOCO Dataset Definiton

# COMMAND ----------

# we can create a new dataloader

# MosaicML CoCo class
from typing import Any, Callable, Optional

from streaming.base import StreamingDataset
import numpy as np
import torch

__all__ = ['StreamingCOCO']


class StreamingCOCO(StreamingDataset):
    """Implementation of the COCO dataset using StreamingDataset.

    Args:
        remote (str, optional): Remote path or directory to download the dataset from. If ``None``,
            its data must exist locally. StreamingDataset uses either ``streams`` or
            ``remote``/``local``. Defaults to ``None``.
        local (str, optional): Local working directory to download shards to. This is where shards
            are cached while they are being used. Uses a temp directory if not set.
            StreamingDataset uses either ``streams`` or ``remote``/``local``. Defaults to ``None``.
        split (str, optional): Which dataset split to use, if any. If provided, we stream from/to
            the ``split`` subdirs of  ``remote`` and ``local``. Defaults to ``None``.
        download_retry (int): Number of download re-attempts before giving up. Defaults to ``2``.
        download_timeout (float): Number of seconds to wait for a shard to download before raising
            an exception. Defaults to ``60``.
        validate_hash (str, optional): Optional hash or checksum algorithm to use to validate
            shards. Defaults to ``None``.
        keep_zip (bool): Whether to keep or delete the compressed form when decompressing
            downloaded shards. If ``False``, keep iff remote is local or no remote. Defaults to
            ``False``.
        epoch_size (int, optional): Number of samples to draw per epoch balanced across all
            streams. If ``None``, takes its value from the total number of underlying samples.
            Provide this field if you are weighting streams relatively to target a larger or
            smaller epoch size. Defaults to ``None``.
        predownload (int, optional): Target number of samples to download per worker in advance
            of current sample. Workers will attempt to download ahead by this many samples during,
            but not before, training. Recommendation is to provide a value greater than per device
            batch size to ensure at-least per device batch size number of samples cached locally.
            If ``None``, its value gets derived using per device batch size and number of
            canonical nodes ``max(batch_size, 256 * batch_size // num_canonical_nodes)``.
            Defaults to ``None``.
        cache_limit (int, optional): Maximum size in bytes of this StreamingDataset's shard cache.
            Before downloading a shard, the least recently used resident shard(s) may be evicted
            (deleted from the local cache) in order to stay under the limit. Set to ``None`` to
            disable shard eviction. Defaults to ``None``.
        partition_algo (str): Which partitioning algorithm to use. Defaults to ``orig``.
        num_canonical_nodes (int, optional): Canonical number of nodes for shuffling with
            resumption. The sample space is divided evenly according to the number of canonical
            nodes. The higher the value, the more independent non-overlapping paths the
            StreamingDataset replicas take through the shards per model replica (increasing data
            source diversity). Defaults to ``None``, which is interpreted as 64 times the number
            of nodes of the initial run.

            .. note::

                For sequential sample ordering, set ``shuffle`` to ``False`` and
                ``num_canonical_nodes`` to the number of physical nodes of the initial run.
        batch_size (int, optional): Batch size of its DataLoader, which affects how the dataset is
            partitioned over the workers. Defaults to ``None``.
        shuffle (bool): Whether to iterate over the samples in randomized order. Defaults to
            ``False``.
        shuffle_algo (str): Which shuffling algorithm to use. Defaults to ``py1s``.
        shuffle_seed (int): Seed for Deterministic data shuffling. Defaults to ``9176``.
        shuffle_block_size (int): Unit of shuffle. Defaults to ``1 << 18``.
        transform (callable, optional): A function/transform that takes in an image and bboxes and
            returns a transformed version. Defaults to ``None``.
    """

    def __init__(self,
                 *,
                 remote: Optional[str] = None,
                 local: Optional[str] = None,
                 split: Optional[str] = None,
                 download_retry: int = 2,
                 download_timeout: float = 60,
                 validate_hash: Optional[str] = None,
                 keep_zip: bool = False,
                 epoch_size: Optional[int] = None,
                 predownload: Optional[int] = None,
                 partition_algo: str = 'orig',
                 cache_limit: Optional[int] = None,
                 num_canonical_nodes: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 shuffle: bool = False,
                 shuffle_algo: str = 'py1s',
                 shuffle_seed: int = 9176,
                 shuffle_block_size: int = 1 << 18,
                 transform: Optional[Callable] = None,
                 allow_unsafe_types: Optional[bool] = False) -> None:
        super().__init__(remote=remote,
                         local=local,
                         split=split,
                         download_retry=download_retry,
                         download_timeout=download_timeout,
                         validate_hash=validate_hash,
                         keep_zip=keep_zip,
                         epoch_size=epoch_size,
                         predownload=predownload,
                         cache_limit=cache_limit,
                         partition_algo=partition_algo,
                         num_canonical_nodes=num_canonical_nodes,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         shuffle_algo=shuffle_algo,
                         shuffle_seed=shuffle_seed,
                         shuffle_block_size=shuffle_block_size,
                         allow_unsafe_types=allow_unsafe_types)
        self.transform = transform

    def get_item(self, idx: int) -> Any:
        """Get sample by global index, blocking to load its shard if missing.

        Args:
            idx (int): Sample index.

        Returns:
            Any: Sample data.
        """
        x = super().get_item(idx)
        img = x['img'].convert('RGB')
        img_id = x['img_id']
        htot = x['htot']
        wtot = x['wtot']
        bbox_sizes = x['bbox_sizes']
        bbox_labels = x['bbox_labels']

        # we need structure this for LoadAnnotations
        # With an instances section
        # Note bboxes are in x1,x2,y2 format

        print(bbox_sizes)

        results = dict(
            img = img,
            img_id = img_id,
            img_shape = (htot, wtot),
            instances = dict(
                bbox = np.asarray(bbox_sizes),
                bbox_label = np.asarray(bbox_labels)
            )
        )

        if self.transform:
            results = self.transform(results)

        return results
    
# COMMAND ----------

# MAGIC %md ## Streaming Dataset to mmdet formater definition

# COMMAND ----------


from torchvision.transforms import functional as F
from torchvision import transforms
from mmcv.transforms.base import BaseTransform
import numpy as np

# create a custom converter to the mmengine standard
class StreamingDataSetConverter(BaseTransform):
    """
    
    Required Keys:
    - img - image in Channel First RGB format

    """

    def __init__(self):
        pass

    def transform(self, results: dict) -> Optional[dict]:

        # the last bit is to transform to BGR from RGB which is what our Streaming Convert job did
        # we also convert to channel last format
        channel_first = np.asarray(F.pil_to_tensor(results['img']))[:,:,::-1]
        results['img'] = np.moveaxis(channel_first, 0, -1)


        return results

# COMMAND ----------

# MAGIC %md # Transforms Definition

# COMMAND ----------

from mmdet.datasets.transforms import (
    CachedMosaic, RandomCrop, YOLOXHSVRandomAug, RandomFlip, LoadAnnotations,
    Pad, CachedMixUp, PackDetInputs, Resize, PackDetInputs
) 
    
from mmcv.transforms import RandomResize

train_transforms = transforms.Compose([
  StreamingDataSetConverter(),
  LoadAnnotations(),
  CachedMosaic(
    img_scale=(640, 640),
    pad_val=114.0,
    max_cached_images=20,
    random_pop=False
  ),
  RandomResize(
    scale=(1280, 1280),
    ratio_range=(0.5, 2.0),
    keep_ratio=True
  ),
  RandomCrop(
    crop_size=(640, 640)
  ),
  YOLOXHSVRandomAug(),
  RandomFlip(
    prob=0.5
  ),
  Pad(
    size=(640, 640), 
    pad_val=dict(img=(114, 114, 114))
  ),
  CachedMixUp(
    img_scale=(640, 640),
    ratio_range=(1.0, 1.0),
    max_cached_images=10,
    random_pop=False,
    pad_val=(114, 114, 114),
    prob=0.5
  ),
  PackDetInputs()
])

# COMMAND ----------

train_transforms_2 = transforms.Compose([
    StreamingDataSetConverter(),
    LoadAnnotations(),
    RandomResize(
        scale=(1280, 1280),
        ratio_range=(0.5, 2.0),
        keep_ratio=True
    ),
    RandomCrop(
        crop_size=(640, 640)
    ),
    YOLOXHSVRandomAug(),
    RandomFlip(
        prob=0.5
    ),
    Pad(
        size=(640, 640), 
        pad_val=dict(img=(114, 114, 114))
    ),
    PackDetInputs()
])

# COMMAND ----------

val_transforms = transforms.Compose([
    StreamingDataSetConverter(),
    Resize(scale=(640, 640), keep_ratio=True),
    Pad(size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    LoadAnnotations(),
    PackDetInputs(meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
])

# COMMAND ----------

# Resetting the loader:
import streaming

local_dir = '/local_disk0/tmp_stream_cache'
dbutils.fs.rm(local_dir, True)
streaming.base.util.clean_stale_shared_memory()

# COMMAND ----------

from mmengine.dataset import pseudo_collate
from mmengine.dataset import DefaultSampler

train_dataset = StreamingCOCO(
  remote=s3_path,
  split='train',
  local=local_dir,
  allow_unsafe_types=True,
  transform=train_transforms
)

#train_sampler = DefaultSampler(train_dataset, shuffle=True)
train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    #sampler=train_sampler,
    collate_fn=pseudo_collate
)

# COMMAND ----------


val_dataset = StreamingCOCO(
  remote=s3_path,
  split='val',
  local=local_dir,
  allow_unsafe_types=True,
  transform=val_transforms
)

#val_sampler = DefaultSampler(val_dataset, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    #sampler=val_sampler,
    collate_fn=pseudo_collate
)

# COMMAND ----------


# DBTITLE 1, Validating Dataloaders
# Testing
for itr, batch in enumerate(train_dataloader):
    print(f'iteration: {itr} Batch: {batch}')
    break

# COMMAND ----------

# MAGIC %md
# MAGIC # Setup Train Loop
# MAGIC
# MAGIC

# COMMAND ----------

from mmengine.config import Config

# DBTITLE 1, Load Config
config = Config.fromfile(databricks_config)
config.keys()

# COMMAND ----------

print(config['test_pipeline'])

# COMMAND ----------

# we need to edit 
custom_hooks=[
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=280,
        switch_pipeline=train_transforms_2)
]
# COMMAND ----------

# Build Runner with config independently

from mmengine.runner import Runner

rtm_det_runner = Runner(
    model=config.get('model'),
    work_dir='/local_disk0/workdir',
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    #test_dataloader=val_dataloader,
    train_cfg=config.get('train_cfg'),
    val_cfg=config.get('val_cfg'),
    #test_cfg=config.get('test_cfg'),
    auto_scale_lr=config.get('auto_scale_lr'),
    optim_wrapper=config.get('optim_wrapper'),
    param_scheduler=config.get('param_scheduler'),
    val_evaluator=dict(
        type='CocoMetric',
        collect_device='gpu',
        #ann_file=data_root + 'annotations/instances_val2017.json',
        metric='bbox',
        proposal_nums=(100, 1, 10),
        format_only=False),
        #backend_args=backend_args),
    #test_evaluator=config.get('test_evaluator'),
    default_hooks=config.get('default_hooks'),
    custom_hooks=config.get('custom_hooks'),
    #data_preprocessor=config.get('data_preprocessor'),
    #load_from=config.get('load_from'),
    #resume=config.get('resume', False),
    #launcher=None,
    env_cfg=config.get('env_cfg', dict(dist_cfg=dict(backend='nccl'))),
    log_processor=config.get('log_processor'),
    log_level=config.get('log_level', 'INFO'),
    visualizer=config.get('visualizer'),
    default_scope=config.get('default_scope', 'mmengine'),
    randomness=config.get('randomness', dict(seed=None)),
    experiment_name=config.get('experiment_name'),
    cfg=config
)

# COMMAND ----------

rtm_det_runner.train()