# Using OpenMM Lab in Databricks samples

OpenMMLab has a variety of different modules for various different computer vision tasks

As a framework, it is built around it's own utility functions distinct from other frameworks like PyTorch Lightning or HuggingFace

## Working with OpenMMLab libraries

OpenMMLab libraries require a custom package handler `mim`(https://github.com/open-mmlab/mim)

This means to install across a whole Databricks cluster, init-scripts are best

When developing using these libraries, they intend for you to clone the github repo then edit the definitions within to build your own pipelines.

Typically an OpenMMLab project will have a config folder then various python files within that define different configs rather than the more common `yaml` or `json` configs we see with libraries like deepspeed

These configs will then be used by loader functions in order to setup the training loop.
Configs that reference configs that reference configs are quite common and can make OpenMMLab examples difficult to follow.

## Worked Example of working with OpenMMLab Training

See: https://github.com/open-mmlab/mmdetection/blob/main/demo/MMDet_Tutorial.ipynb as a start

The training example starts in the `Prepare a config` section.
Training is based on the `!python tools/train.py` file which is part of the mmdet github files

It references `configs/rtmdet/rtmdet_tiny_1xb4-20e_balloon.py`
The contents of this file in inside the tutorial notebook.
Note that it references ``./rtmdet_tiny_8xb32-300e_coco.py` at the start:

```
_base_ = './rtmdet_tiny_8xb32-300e_coco.py'
```

this is here: https://github.com/open-mmlab/mmdetection/blob/main/configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py

this then refers to another file: 

```
_base_ = './rtmdet_s_8xb32-300e_coco.py'
```

and so on so we have:

- rtmdet_tiny_1xb4-20e_balloon.py
-- rtmdet_tiny_8xb32-300e_coco.py
--- rtmdet_s_8xb32-300e_coco.py
---- rtmdet_l_8xb32-300e_coco.py
----- default_runtime.py
----- schedule_1x.py
----- coco_detection.py
----- rtmdet_tta.py

So to start our project we can either fork the OpenMMLab repo and start from there, carrying all the baggage along as well.
Or we can following the chain of config dependencies that copy the predefined examples for our dataset / our model / our lr_schedule etc

In this case we will copy the necessary files from the mmdet github to build up our train job.

## Steps


### 1) Clone base configs
We will start with the four configs at the lowest level. 
We will just clone these:

That is:

----- default_runtime.py
----- schedule_1x.py
----- coco_detection.py
----- rtmdet_tta.py

It seems like `rtmdet_tta` isn't being used but the commit it came in looks importantish so we will bring it across too: https://github.com/open-mmlab/mmdetection/commit/99e02c7aca89e42b3ce55a8d0eced08bc68f78f1

### 2) Build train config

We can start reviewing the model definitions and training details next,
this starts in: `rtmdet_l_8xb32-300e_coco.py` we will also review `rtmdet_s_8xb32-300e_coco.py` and `rtmdet_tiny_8xb32-300e_coco.py`
to see what changes have been made and how OpenMMLab intends for you to edit and change configs.

In `rtmdet_l_8xb32-300e_coco.py` we see that there is the `_base_` import
followed by:
- model
- train_pipeline
- train_pipeline_stage2
- test_pipeline
- train_dataloader
- val_dataloader
- some params including epochs / base_lr and an interval param
- train_cfg
- val_evaluator
- test_evaluator
- optim_wrapper
- param_scheduler
- default_hooks
- custom_hooks

All these configurations in a way resemble the standard mess you would find inside of a HF TrainingParams type object or something that you might put in a tool like Hydra (https://hydra.cc/docs/intro/)

To understand these modules, we need to look at each of the `dict()` objects these are referring to classes that are either inside of `mmdet` or `mmcv` (the foundation Computer Vision library).

We can find out what these do by doing searches on the `type` field inside the dict. ie:

```
dict(type='LoadImageFromFile', backend_args=backend_args),
```

is from https://mmcv.readthedocs.io/en/latest/api/generated/mmcv.transforms.LoadImageFromFile.html#mmcv.transforms.LoadImageFromFile

each file in the series of config files can overwrite the files that it uses as a `_base_`
so sometimes it can get confusing as to how something is configured.
This documentation provides some guides: https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#delete-key-in-dict

The rtmdet architecture is explained is explained here: https://arxiv.org/pdf/2212.07784.pdf we will not go into the construction of the core model just look at how the configurations work.

`rtmdet_l_8xb32-300e_coco.py` contains the full definition setup to work with the coco data format. we will replicate this

`rtmdet_s_8xb32-300e_coco.py` edits the:
- model backbone / neck / bbox head
The main variables we should look at are the `deepen_factor` and `widen_factor` which reduces the number of parameters in the model. The other changes are to faciliate this reduction in model parameters and make sure that the different modules still connect.

we see similar changes in `rtmdet_tiny_8xb32-300e_coco.py` as well.

Given the overlapping configurations, seeing what the actual config is like can be hard.
We can thus load and check them out

Loading the configs requires running:

```{python}
from mmengine.config import Config

cfg = Config.fromfile('./configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py')
print(cfg)

```

We can then explore the config like a dictionary.


### 3) Running back train job

Openmmlab libraries each have a train script as part of the library repo.
The one for mmdet is here: https://github.com/open-mmlab/mmdetection/blob/v3.3.0/tools/train.py

Exploring the file, we can see that `Runner` is the key class.
We can manually import this and use this to fine tune our model.

## Working with OpenMMLab on Databricks

Now that we have a basic grasp of the library, we can see how to work with it on databricks.
First part is the setup.

To use the mim package manager and make sure we install on all nodes when in a distributed configuration, we should use init scripts rather than install via `sh` or `%pip`

To avoid click ops, we will setup our cluster with the `databricks-sdk`

See: `0_setup_cluster.py` in this script we will setup the cluster config that we want as well as the init-script that we need. We need to use an init-script because we can't use `mim` to setup libraries as openmmlab packages require.

For this particular example, we will use the `coco` dataset. So we need to get the data loaded to `dbfs` for that. That is what `1_download_data.py` does.

Then we can start working with the library and building out our training loops. In `2_build_train_loop.py` we will trigger the training process with Spark Torch-Distributor.

In `3_use_train_script.py` we will use a slightly modified version of the train script from openmmlab mmdet library and trigger that also with Torch Distributor.







