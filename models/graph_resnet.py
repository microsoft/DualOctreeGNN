# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
import ocnn

from .modules import GraphConvBnRelu, GraphResBlocks
from .dual_octree import DualOctree


class GraphResNet(torch.nn.Module):

  def __init__(self, depth, channel_in, nout, resblk_num):
    super().__init__()
    self.depth, self.channel_in = depth, channel_in
    channels = [2 ** max(11 - i, 2) for i in range(depth + 1)]
    channels.append(channels[depth])
    n_edge_type, avg_degree, bottleneck = 7, 7, 4

    self.conv1 = GraphConvBnRelu(
        channel_in, channels[depth], n_edge_type, avg_degree)
    self.resblocks = torch.nn.ModuleList(
        [GraphResBlocks(channels[d + 1], channels[d], resblk_num, bottleneck,
         n_edge_type, avg_degree) for d in range(depth, 2, -1)])
    self.pools = torch.nn.ModuleList(
        [ocnn.OctreeMaxPool(d) for d in range(depth, 2, -1)])
    self.header = torch.nn.Sequential(
        ocnn.FullOctreeGlobalPool(depth=2),      # global pool
        #  torch.nn.Dropout(p=0.5),              # drop
        torch.nn.Linear(channels[3], nout))      # fc

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
      if d == self.depth:  # the first conv
        data = self.conv1(data, edge_idx, edge_type)
      data = self.resblocks[i](data, edge_idx, edge_type)

      # downsampleing
      data = data.t().unsqueeze(0).unsqueeze(-1)
      data = self.pools[i](data, octree)

    # classification head
    data = self.header(data)
    return data
