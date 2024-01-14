# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Download testing dataset
# MAGIC In order to run training experiments we need to setup some data first

# COMMAND ----------

# DBTITLE 1, Setup Dataset Folder
import os
username = spark.sql("SELECT current_user()").first()['current_user()']
data_path = f'/Users/{username}/openmmlab/data/coco'
dbutils.fs.mkdirs(data_path)

# COMMAND ----------

# DBTITLE 1, Download Dataset
# MAGIC %sh
# MAGIC
# MAGIC rm -rf /local_disk0/mmdet
# MAGIC git clone https://github.com/open-mmlab/mmdetection.git /local_disk0/mmdet
# MAGIC python /local_disk0/mmdet/tools/misc/download_dataset.py --dataset-name coco2017 --save-dir $MMLAB_DATA_DIR --unzip

# COMMAND ----------

# MAGIC %md
# MAGIC This dataset will be stored as json and image files on s3
# MAGIC For this code sample we will not do further data processing or restructuring