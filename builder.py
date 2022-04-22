# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import ocnn

import datasets
import models
import losses


def get_model(flags):
  params = [flags.depth, flags.channel, flags.nout,
            flags.full_depth, flags.depth_out]
  if flags.name == 'graph_ounet' or \
     flags.name == 'graph_unet' or \
     flags.name == 'graph_ae':
    params.append(flags.resblock_type)
    params.append(flags.bottleneck)

  # if flags.name == 'rounet':
  #   model = models.rounet.ROUNet(*params)
  # elif flags.name == 'r2ounet':
  #   model = models.r2ounet.R2OUNet(*params)
  # elif flags.name == 'regunet':
  #   model = models.regunet.RegUNet(*params)
  # elif flags.name == 'ounet':
  #   model = ocnn.OUNet(*params[:-1])
  # # regress the SDF on the finest octree levels
  # elif flags.name == 'regounet':
  #   model = models.regounet.RegOUNet(*params)
  # # regress the SDF on all octree levels
  # elif flags.name == 'regounet_all':
  #   model = models.regounet_all.RegOUNetALL(*params)
  # # regress the SDF on all octree levels
  # elif flags.name == 'reg2ounet_all':
  #   model = models.reg2ounet_all.Reg2OUNetALL(*params)

  if flags.name == 'octree_ounet':
    model = models.octree_ounet.OctreeOUNet(*params)
  elif flags.name == 'reg1ounet_grad':
    model = models.reg1ounet_grad.Reg1OUNetGrad(*params)
  elif flags.name == 'graph_slounet':
    model = models.graph_slounet.GraphSLOUNet(*params)
  elif flags.name == 'graph_ounet':
    model = models.graph_ounet.GraphOUNet(*params)
  elif flags.name == 'graph_unet':
    model = models.graph_unet.GraphUNet(*params)
  elif flags.name == 'graph_ae':
    model = models.graph_ae.GraphAE(*params)
  else:
    raise ValueError
  return model


def get_dataset(flags):
  # if flags.name.lower() == 'completion':
  #   return datasets.get_completion_dataset(flags)
  # elif flags.name.lower() == 'noise2clean':
  #   return datasets.get_noise2clean_dataset(flags)
  # elif flags.name.lower() == 'convonet':
  #   return datasets.get_convonet_dataset(flags)
  # elif flags.name.lower() == 'deepmls':
  #   return datasets.get_deepmls_dataset(flags)

  if flags.name.lower() == 'shapenet':
    return datasets.get_shapenet_dataset(flags)
  elif flags.name.lower() == 'pointcloud':
    return datasets.get_pointcloud_dataset(flags)
  elif flags.name.lower() == 'singlepointcloud':
    return datasets.get_singlepointcloud_dataset(flags)
  elif flags.name.lower() == 'pointcloud_eval':
    return datasets.get_pointcloud_eval_dataset(flags)
  elif flags.name.lower() == 'synthetic_room':
    return datasets.get_synthetic_room_dataset(flags)
  else:
    raise ValueError


def get_classification_model(flags):
  if flags.name.lower() == 'lenet':
    model = ocnn.LeNet(flags.depth, flags.channel, flags.nout)
  elif flags.name.lower() == 'resnet':
    model = ocnn.ResNet(flags.depth, flags.channel, flags.nout,
                        flags.resblock_num)
  elif flags.name.lower() == 'graphlenet':
    model = models.graph_lenet.GraphLeNet(
        flags.depth, flags.channel, flags.nout)
  elif flags.name.lower() == 'dualgraphlenet':
    model = models.graph_lenet.DualGraphLeNet(
        flags.depth, flags.channel, flags.nout)
  elif flags.name.lower() == 'graphresnet':
    model = models.graph_resnet.GraphResNet(
        flags.depth, flags.channel, flags.nout, flags.resblock_num)
  else:
    raise ValueError
  return model


def get_loss_function(flags):
  if flags.name.lower() == 'shapenet':
    return losses.shapenet_loss
  elif flags.name.lower() == 'dfaust':
    return losses.dfaust_loss
  elif flags.name.lower() == 'synthetic_room':
    return losses.synthetic_room_loss
  else:
    raise ValueError
