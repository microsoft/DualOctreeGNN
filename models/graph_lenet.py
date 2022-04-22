# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
import ocnn

from .modules import GraphConvBnRelu, GraphDownsample, GraphMaxpool
from .dual_octree import DualOctree


class GraphLeNet(torch.nn.Module):
  '''This is to do comparison with the original O-CNN'''

  def __init__(self, depth, channel_in, nout):
    super().__init__()
    self.depth, self.channel_in = depth, channel_in
    channels = [2 ** max(9 - i, 2) for i in range(depth + 1)]
    channels.append(channel_in)

    self.convs = torch.nn.ModuleList([
        GraphConvBnRelu(channels[d + 1], channels[d], n_edge_type=7,
                        avg_degree=7) for d in range(depth, 2, -1)])
    self.pools = torch.nn.ModuleList([
        ocnn.OctreeMaxPool(d) for d in range(depth, 2, -1)])
    self.octree2voxel = ocnn.FullOctree2Voxel(2)
    self.header = torch.nn.Sequential(
        torch.nn.Dropout(p=0.5),                        # drop1
        ocnn.FcBnRelu(channels[3] * 64, channels[2]),   # fc1
        torch.nn.Dropout(p=0.5),                        # drop2
        torch.nn.Linear(channels[2], nout))             # fc2

  def forward(self, octree):
    # Get the initial feature
    data = ocnn.octree_property(octree, 'feature', self.depth)
    assert data.size(1) == self.channel_in

    # build the dual octree
    doctree = DualOctree(octree)
    doctree.post_processing_for_ocnn()

    # forward the network
    for i, d in enumerate(range(self.depth, 2, -1)):
      # perform graph conv
      data = data.squeeze().t()
      edge_idx = doctree.graph[d]['edge_idx']
      edge_type = doctree.graph[d]['edge_type']
      data = self.convs[i](data, edge_idx, edge_type)

      # downsampleing
      data = data.t().unsqueeze(0).unsqueeze(-1)
      data = self.pools[i](data, octree)

    # classification head
    data = self.octree2voxel(data)
    data = self.header(data)
    return data


class DualGraphLeNet(torch.nn.Module):

  def __init__(self, depth, channel_in, nout):
    super().__init__()
    self.depth, self.channel_in = depth, channel_in
    channels = [2 ** max(9 - i, 2) for i in range(depth + 1)]
    channels.append(channel_in)

    self.convs = torch.nn.ModuleList([
        GraphConvBnRelu(channels[d + 1], channels[d], n_edge_type=7,
                        avg_degree=7, n_node_type=d-1)
        for d in range(depth, 2, -1)])
    # self.downsample = torch.nn.ModuleList([
    #     GraphDownsample(channels[d]) for d in range(depth, 2, -1)])
    self.downsample = torch.nn.ModuleList([
        GraphMaxpool() for d in range(depth, 2, -1)])
    self.octree2voxel = ocnn.FullOctree2Voxel(2)
    self.header = torch.nn.Sequential(
        torch.nn.Dropout(p=0.5),                        # drop1
        ocnn.FcBnRelu(channels[3] * 64, channels[2]),   # fc1
        torch.nn.Dropout(p=0.5),                        # drop2
        torch.nn.Linear(channels[2], nout))             # fc2

  def forward(self, octree):
    # build the dual octree
    doctree = DualOctree(octree)
    doctree.post_processing_for_docnn()

    # Get the initial feature
    data = doctree.get_input_feature()
    assert data.size(1) == self.channel_in

    # forward the network
    for i, d in enumerate(range(self.depth, 2, -1)):
      # perform graph conv
      edge_idx = doctree.graph[d]['edge_idx']
      edge_type = doctree.graph[d]['edge_dir']
      node_type = doctree.graph[d]['node_type']
      data = self.convs[i](data, edge_idx, edge_type, node_type)

      # downsampleing
      nnum = doctree.nnum[d]
      lnum = doctree.lnum[d-1]
      leaf_mask = doctree.node_child(d-1) < 0
      data = self.downsample[i](data, leaf_mask, nnum, lnum)

    # classification head
    data = data.t().unsqueeze(0).unsqueeze(-1)
    data = self.octree2voxel(data)
    data = self.header(data)
    return data
