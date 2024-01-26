# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Build OpenMMNet Trainloop  
# MAGIC We will setup our train loop here

# COMMAND ----------

# DBTITLE 1, Setup Dataset Folder
import os
import logging

logger = logging.getLogger(__name__)
username = spark.sql("SELECT current_user()").first()['current_user()']
data_path = f'/Users/{username}/openmmlab/data/coco'
dbutils.fs.mkdirs(data_path)

# Get current workspace host and token
browser_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()
db_host = f"https://{browser_host}"
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

config_file_path = './configs/rtmdet/rtmdet_databricks_coco.py'

# COMMAND ----------

# DBTITLE 1, Build Configuration
databricks_cfg = f"""

_base_ = './rtmdet_tiny_8xb32-300e_coco.py'

experiment_name = '/Users/{username}/openmmdet'
data_root = '/dbfs{data_path}/'

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

with open(config_file_path, 'w') as f:
    f.write(databricks_cfg)

# COMMAND ----------

# MAGIC %md
# MAGIC # Exploring the Configs

# COMMAND ----------

from mmengine.config import Config

cfg = Config.fromfile(config_file_path)
print(cfg)

# COMMAND ----------

# The cfg object can be accessed like a dict
cfg.keys()

# COMMAND ----------

# Training in OpenMMLab is typically done via cli
# We can look at the python train script for mmdet here: https://github.com/open-mmlab/mmdetection/blob/main/tools/train.py
# We can also turn the logic within into a python function in notebook for execution via TorchDistributor
def train_loop(config:str='./configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py',
               db_host: str = None,
               db_token: str= None) :
  from mmengine.config import Config, ConfigDict
  from mmengine.runner import Runner
  
  from mmdet.utils import setup_cache_size_limit_of_dynamo

  setup_cache_size_limit_of_dynamo()

  os.environ['DATABRICKS_HOST'] = db_host
  os.environ['DATABRICKS_TOKEN'] = db_token

  working_cfg = Config.fromfile(config)
  #working_cfg = Config.fromfile('./configs/rtmdet/rtmdet_databricks_coco.py')
  working_cfg.launcher = 'pytorch' # we need to set this to train properly on databricks
  
  # config overrides - we can either override like I am doing here are create our own config file
  ## data_root seems a paricularly hard var to update due to the nesting of configs
  ## So there is a mmdet.util for it `update_data_root` but that is broken so we will write our own

  # need find different way to override
  def update_data_root(cfg, new_value):
    for k, v in cfg.items():
      if isinstance(v, ConfigDict):
        update_data_root(cfg[k], new_value)
      if isinstance(v, str) and k == 'data_root':
        #print(f'{k} {v}')
        cfg[k] = v.replace(v, new_value)

  update_data_root(working_cfg, f'/dbfs{data_path}/')
  working_cfg.val_evaluator.ann_file = f'/dbfs{data_path}/annotations/instances_val2017.json'
  working_cfg.test_evaluator.ann_file = f'/dbfs{data_path}/annotations/instances_val2017.json'
  
  # essential configs
  working_cfg.work_dir = '/local_disk0/openmmlab'

  runner = Runner.from_cfg(working_cfg)

  runner.train()



# COMMAND ----------

# DBTITLE 1, Native Training
train_loop(config='./configs/rtmdet/rtmdet_databricks_coco.py',
           db_host=db_host, 
           db_token=db_token)

# COMMAND ----------

# DBTITLE 1, Torch Distributor Training
from pyspark.ml.torch.distributor import TorchDistributor

num_gpus_per_node = 2
num_nodes = 1
num_processes = num_gpus_per_node * num_nodes
local_status = True if num_nodes == 1 else False


distributor = TorchDistributor(num_processes=num_processes, 
                                local_mode=local_status, use_gpu=True)
    
logger.info(f"Launching job with TorchDistributor with {num_gpus_per_node} gpus per node and {num_nodes} nodes")
completed_trainer = distributor.run(train_loop, 
                                    './configs/rtmdet/rtmdet_databricks_coco.py',
                                    db_host, db_token)