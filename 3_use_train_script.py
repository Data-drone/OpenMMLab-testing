# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Build OpenMMNet Trainloop  
# MAGIC We can use the train script from OpenMMLab too
# MAGIC
# MAGIC Here we use the train script copied over from mmdet
# MAGIC **Note** we do do an edit for mlflow to work properly

# COMMAND ----------

# DBTITLE 1, Setup Dataset Folder
import os
username = spark.sql("SELECT current_user()").first()['current_user()']
data_path = f'/Users/{username}/openmmlab/data/coco'
dbutils.fs.mkdirs(data_path)

# Get current workspace host and token
browser_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()
db_host = f"https://{browser_host}"
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

os.environ['DATABRICKS_HOST'] = db_host
os.environ['DATABRICKS_TOKEN'] = db_token
os.environ['DATA_ROOT'] = f'/dbfs{data_path}'

# COMMAND ----------


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

config=f'./configs/rtmdet/rtmdet_databricks_coco.py'
with open(config, 'w') as f:
    f.write(databricks_cfg)

# COMMAND ----------

# DBTITLE 1, Native Training
# MAGIC %sh
# MAGIC python scripts/train.py configs/rtmdet/rtmdet_databricks_coco.py --work-dir '/local_disk0/openmmlab' --db-host $DATABRICKS_HOST --db-token $DATABRICKS_TOKEN --launcher none --data-root $DATA_ROOT

# COMMAND ----------
    
# DBTITLE 1, Torch Distributor
from pyspark.ml.torch.distributor import TorchDistributor

num_gpus_per_node = 2
num_nodes = 1
num_processes = num_gpus_per_node * num_nodes
local_status = True if num_nodes == 1 else False


distributor = TorchDistributor(num_processes=num_processes, 
                                local_mode=local_status, use_gpu=True)
    
logger.info(f"Launching job with TorchDistributor with {num_gpus_per_node} gpus per node and {num_nodes} nodes")
completed_trainer = distributor.run('./scripts/train.py', config, f'--work-dir=/local_disk0/openmmlab',
                                    f'--db-host={db_host}', f'--db-token={db_token}', f'--data-root=/dbfs{data_path}')

# COMMAND ----------