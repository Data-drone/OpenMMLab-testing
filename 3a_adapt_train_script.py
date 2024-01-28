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
# MAGIC #python scripts/create_config.py \
# MAGIC python scripts/create_iter_config.py \
# MAGIC     --experiment-name /Users/$DB_USERNAME/openmmlab \
# MAGIC     --data-root $DATA_ROOT \
# MAGIC     --data-bucket s3://db-brian-mosaic/openmmlab/data/coco/ \
# MAGIC     --mlflow-host databricks \
# MAGIC     --config-path .

# COMMAND ----------

# DBTITLE 1, Launch Training
# MAGIC %sh
# MAGIC MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true python scripts/train_s3_backend.py configs/rtmdet/rtmdet_databricks_coco.py \
# MAGIC     --work-dir '/local_disk0/openmmlab' \
# MAGIC     --db-host $DATABRICKS_HOST \
# MAGIC     --db-token $DATABRICKS_TOKEN \
# MAGIC     --launcher none \
# MAGIC     --annotation-local $ANNOTATIONS_LOCAL \
# MAGIC     --data-root $DATA_ROOT

# COMMAND ----------

# MAGIC %md # Using the trained Model
# MAGIC The model save location depends on our checkpoint setting and also the work-dir \
# MAGIC Artifacts include the `config.py` and also the `.pth` file of model weights
# MAGIC
# MAGIC **Note**
# MAGIC Of special note for mmdet is that instantiating the model requires a color palette for the model \
# MAGIC We base this code on: https://github.com/roboflow/notebooks/blob/main/notebooks/train-rtmdet-object-detection-on-custom-data.ipynb \
# MAGIC If there is no palette then the `init_detector` will instantiate the CocoDataset class which requires the coco annotation file.
# MAGIC To stop this, we have cut and pasted in the palette from the CocoDataset class 
# MAGIC
# MAGIC https://github.com/open-mmlab/mmdetection/blob/44ebd17b145c2372c4b700bfb9cb20dbd28ab64a/mmdet/datasets/coco.py
# MAGIC
# MAGIC Another note is that when doing inference, the model will run the `test_pipeline` as defined in the model `config` file \
# MAGIC In the case of this training routine, the custom `awss3` backend was used so we need to set `cfg_options` in order to override the backend 
# MAGIC and set it to use the `local` option. 
# MAGIC
# MAGIC This behaviour does mean that for practical applications, we would might have to build quite custom test_pipelines.
# MAGIC

# COMMAND ----------

from mmdet.apis import init_detector

# 

model = init_detector(
  config = '/dbfs/tmp/openmmlab-workdir/rtmdet_databricks_coco.py', 
  checkpoint = '/dbfs/tmp/openmmlab-test/openmmlab-workdir/iter_10.pth',
  palette = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
         (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
         (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
         (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
         (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
         (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
         (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0),
         (174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174),
         (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54),
         (207, 138, 255), (151, 0, 95), (9, 80, 61), (84, 105, 51),
         (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),
         (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),
         (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161),
         (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120),
         (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133),
         (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62),
         (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45),
         (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),
         (246, 0, 122), (191, 162, 208)],
  cfg_options = {'test_pipeline': [{'backend_args': {'backend': 'local'}}]}
  )

# COMMAND ----------

# Supervision library for postprocessing
# MAGIC %pip install supervision

# COMMAND ----------

import cv2
import supervision as sv

from mmdet.apis import inference_detector

IMAGE_PATH = '/dbfs/mnt/mosaic-data/openmmlab/data/coco/test2017/000000013669.jpg'
image = cv2.imread(IMAGE_PATH)

result = inference_detector(model, image)
detections = sv.Detections.from_mmdetection(result)

# filter out low confidence
detections = detections[detections.confidence > 0.1].with_nms()

box_annotator = sv.BoxAnnotator()
annotated_image = box_annotator.annotate(image.copy(), detections)

# COMMAND ----------

from PIL import Image

# OpenMMLab works on BGR so we need to reindex to RGB
im = Image.fromarray(annotated_image[:, :, ::-1])
#im = Image.fromarray(image[:, :, ::-1])

display(im)

# COMMAND ----------

