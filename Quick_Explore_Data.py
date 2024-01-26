# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Explore Dataset
# MAGIC We will use spark to explore the dataset
# COMMAND ----------
import json
import pandas as pd

root_dir = '/dbfs/mnt/mosaic-data/openmmlab/data/coco'
annotations = f"{root_dir}/annotations/instances_train2017.json"

# COMMAND ----------

# TODO
# - Check no empty imgs in train or val?
# - check if we have annotations tied to images that don't exist?
# - Check if we have images with no bbox?

# COMMAND ----------

# DBTITLE 1, Load Annotations

with open(annotations, "r") as f:
  annotation_file = json.load(f)

#annotation_df = spark.createDataFrame(annotation_file['annotations'], schema)

# COMMAND ----------

# We want to create a filterable dataframe that we can then query as needed
# columns: segmentation / area / iscrowd / image_id / bbox / category_id / id

keys_to_keep = ['iscrowd', 'image_id', 'bbox', 'category_id', 'id']

filtered_data = [{k: d[k] for k in keys_to_keep} for d in annotation_file['annotations']]

annotation_df = pd.DataFrame(filtered_data)
spark_annotation_df = spark.createDataFrame(annotation_df)
spark_images_df = spark.createDataFrame(annotation_file['images'])

# COMMAND ----------

# MAGIC %sql 
# MAGIC CREATE CATALOG IF NOT EXISTS computer_vision;
# MAGIC CREATE SCHEMA IF NOT EXISTS computer_vision.coco_dataset;

# COMMAND ----------

spark_annotation_df.write.saveAsTable('computer_vision.coco_dataset.bbox_annotations')
spark_images_df.write.saveAsTable('computer_vision.coco_dataset.image_list')

# COMMAND ----------

# MAGIC %sql
# MAGIC -- explore the dataset and see how many nulls etc we have
# MAGIC USE computer_vision.coco_dataset;
# MAGIC 
# MAGIC WITH dataset as (
# MAGIC   SELECT * FROM image_list
# MAGIC   LEFT JOIN bbox_annotations
# MAGIC   ON image_list.id == bbox_annotations.image_id
# MAGIC )
# MAGIC 
# MAGIC SELECT count(*) FROM dataset
# MAGIC WHERE bbox is null

# COMMAND ----------

# MAGIC %md
# MAGIC # Explore Image by ID

# COMMAND ----------

from PIL import Image, ImageDraw

image_id = '458705'
image_file = f"{root_dir}/train2017/000000{image_id}.jpg"

# COMMAND ----------

image_to_review = Image.open(image_file)

display(image_to_review)

# COMMAND ----------

display(
 spark.sql(f"""SELECT * FROM computer_vision.coco_dataset.bbox_annotations 
           WHERE image_id == {image_id}""") 
)
# COMMAND ----------

bboxes = spark.sql(f"""SELECT bbox FROM computer_vision.coco_dataset.bbox_annotations 
           WHERE image_id == {image_id}""").toPandas()
#bboxes

# COMMAND ----------

draw = ImageDraw.Draw(image_to_review)
for bbox in bboxes.bbox:
  
  # bbox format: [x_min, y_min, width, height]
  rounded_bbox = [int(round(x)) for x in bbox]
  
  x_min, y_min, width, height = rounded_bbox
  # # Draw rectangle on the image
  draw.rectangle([(x_min, y_min), (width, height)], outline='red')

display(image_to_review)