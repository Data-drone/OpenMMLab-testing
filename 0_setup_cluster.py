# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Open MM Lab Cluster Launch

# COMMAND ----------

# DBTITLE 1,Update for sdk to allow single node GPU instances
%pip install -U databricks-sdk>=0.9.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Setup Workspace Client
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.compute import InitScriptInfo

# Get current workspace host and token
browser_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()
db_host = f"https://{browser_host}"
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

w = WorkspaceClient(
  host  = db_host,
  token = db_token
)

# COMMAND ----------

# DBTITLE 1,Helper function to check if cluster exists
import fnmatch
import os

def find_clusters_by_name(cluster_list, target_name):
    matching_clusters = []
    for cluster in cluster_list:
        if fnmatch.fnmatchcase(cluster.cluster_name, target_name):
            matching_clusters.append(cluster)
    return matching_clusters

# COMMAND ----------

target_name = "openmmlab - single"

# Azure
node_type = 'Standard_NC12s_v3'

# AWS - TODO
# node_type = 'g4dn.4xlarge'

# get current working directory in order to locate init script
full_dir = os.getcwd()
project_dir = full_dir.replace('/Workspace', '')

matching_clusters = find_clusters_by_name(w.clusters.list(), target_name)

if len(matching_clusters) == 0:

    print("Attempting to create cluster. Please wait...")

    c = w.clusters.create_and_wait(
    cluster_name             = target_name,
    spark_version            = '14.2.x-gpu-ml-scala2.12',
    node_type_id             = node_type,
    autotermination_minutes = 66,
    num_workers              = 0,
    spark_conf               = {
            "spark.master": "local[*, 4]",
            "spark.databricks.cluster.profile": "singleNode"
            },
    custom_tags = {"ResourceClass": "SingleNode"},
    init_scripts = [
         InitScriptInfo().from_dict(
            {
                "workspace": {"destination": f"{project_dir}/init-scripts/setup_openmmlab.sh"}
             }
            ),  
    ]
    )

    print(f"The cluster is now ready at " \
        f"{w.config.host}#setting/clusters/{c.cluster_id}/configuration\n")
    

elif len(matching_clusters) == 1:

    cluster = matching_clusters[0].cluster_id

    print("cluster exists already")

# COMMAND ----------
    
# Distributed setup
target_name = "openmmlab - cluster"

# Azure
driver_type = 'Standard_NC6s_v3'
worker_type = 'Standard_NC12s_v3'

# AWS - TODO
# node_type =

# get current working directory in order to locate init script
full_dir = os.getcwd()
project_dir = full_dir.replace('/Workspace', '')

matching_clusters = find_clusters_by_name(w.clusters.list(), target_name)

if len(matching_clusters) == 0:

    print("Attempting to create cluster. Please wait...")

    c = w.clusters.create_and_wait(
    cluster_name             = target_name,
    spark_version            = '14.2.x-gpu-ml-scala2.12',
    node_type_id             = worker_type,
    driver_node_type_id      = driver_type,
    autotermination_minutes = 66,
    num_workers              = 2,
    init_scripts = [
         InitScriptInfo().from_dict(
            {
                "workspace": {"destination": f"{project_dir}/init-scripts/setup_openmmlab.sh"}
             }
            ), 
    ]
    )

    print(f"The cluster is now ready at " \
        f"{w.config.host}#setting/clusters/{c.cluster_id}/configuration\n")
    

elif len(matching_clusters) == 1:

    cluster = matching_clusters[0].cluster_id

    print("cluster exists already")

