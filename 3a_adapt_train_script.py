# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Adapt Train Script for Mosaic  
# MAGIC
# MAGIC There are a few adaptations that we need to make due to a few factors:
# MAGIC - In mosaic there is no fuse mount
# MAGIC - mmdet seems to require that coco annotation be on local disk

# COMMAND ----------

import os
username = spark.sql("SELECT current_user()").first()['current_user()']

data_path = f'/mnt/mosaic-data/openmmlab/data/coco'
dbutils.fs.mkdirs(data_path)

# Get current workspace host and token
browser_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()
db_host = f"https://{browser_host}"
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

s3_path = 's3://db-brian-mosaic/openmmlab/data/coco/'

os.environ['DB_USERNAME'] = username
os.environ['DATABRICKS_HOST'] = db_host
os.environ['DATABRICKS_TOKEN'] = db_token
os.environ['DATA_ROOT'] = s3_path
os.environ['ANNOTATIONS_LOCAL'] = f'/dbfs{data_path}'

# COMMAND ----------

# DBTITLE 1, Create Config File
# MAGIC %sh
# MAGIC python scripts/create_config.py \
# MAGIC     --experiment-name /Users/$DB_USERNAME/openmmlab \
# MAGIC     --data-root $DATA_ROOT \
# MAGIC     --data-bucket s3://db-brian-mosaic/openmmlab/data/coco/ \
# MAGIC     --config-path .

# COMMAND ----------

# DBTITLE 1, Launch Training
# MAGIC %sh
# MAGIC python scripts/train_s3_backend.py configs/rtmdet/rtmdet_databricks_coco.py \
# MAGIC     --work-dir '/local_disk0/openmmlab' \
# MAGIC     --db-host $DATABRICKS_HOST \
# MAGIC     --db-token $DATABRICKS_TOKEN \
# MAGIC     --launcher none \
# MAGIC     --annotation-local $ANNOTATIONS_LOCAL \
# MAGIC     --data-root $DATA_ROOT