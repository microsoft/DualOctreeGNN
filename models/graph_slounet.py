# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
import ocnn
import torch.nn
import torch.nn.functional as F

from . import mpu
from . import modules
from . import graph_ounet
from . import dual_octree


class GraphSLDownsample(modules.GraphDownsample):

  def forward(self, x, leaf_mask, numd, lnumd):
    # downsample nodes at layer depth
    outd = x[-numd:]
    outd = self.downsample(outd)

    # get the nodes at layer (depth-1)
    out = torch.zeros(leaf_mask.shape[0], x.shape[1], device=x.device)
    out[leaf_mask.logical_not()] = outd

    if self.channels_in != self.channels_out:
      out = self.conv1x1(out)
    return out


class GraphSLUpsample(modules.GraphUpsample):

  def forward(self, x, leaf_mask, numd):
    # upsample nodes at layer (depth-1)
    outd = x[-numd:]
    out = outd[leaf_mask.logical_not()]
    out = self.upsample(out)

    if self.channels_in != self.channels_out:
      out = self.conv1x1(out)
    return out


class GraphSLOUNet(graph_ounet.GraphOUNet):

  def __init__(self, depth, channel_in, nout, full_depth=2, depth_out=6,
               resblk_type='bottleneck', bottleneck=4):
    super().__init__(depth, channel_in, nout, full_depth, depth_out,
                     resblk_type, bottleneck)

    self.downsample = torch.nn.ModuleList(
        [GraphSLDownsample(self.channels[d], self.channels[d-1])
         for d in range(depth, full_depth, -1)])

    self.upsample = torch.nn.ModuleList(
        [GraphSLUpsample(self.channels[d-1], self.channels[d])
         for d in range(full_depth+1, depth+1)])

  def _get_input_feature(self, doctree):
    return doctree.get_input_feature(all_leaf_nodes=False)

  def octree_decoder(self, convs, doctree_out, doctree, update_octree=False):
    logits = dict()
    reg_voxs = dict()
    deconvs = dict()
    reg_voxs_list = []

    deconvs[self.full_depth] = convs[self.full_depth]
    for i, d in enumerate(range(self.full_depth, self.depth_out+1)):
      if d > self.full_depth:
        nnum = doctree_out.nnum[d-1]
        leaf_mask = doctree_out.node_child(d-1) < 0
        deconvd = self.upsample[i-1](deconvs[d-1], leaf_mask, nnum)
        skip = modules.doctree_align(
            convs[d], doctree.graph[d]['keyd'], doctree_out.graph[d]['keyd'])
        deconvd = deconvd + skip  # skip connections

        edge_idx = doctree_out.graph[d]['edge_idx']
        edge_type = doctree_out.graph[d]['edge_dir']
        node_type = doctree_out.graph[d]['node_type']
        deconvs[d] = self.decoder[i-1](deconvd, edge_idx, edge_type, node_type)

      # predict the splitting label
      logit = self.predict[i](deconvs[d])
      nnum = doctree_out.nnum[d]
      logits[d] = logit[-nnum:]

      # update the octree according to predicted labels
      if update_octree:
        label = logits[d].argmax(1).to(torch.int32)
        octree_out = doctree_out.octree
        octree_out = ocnn.octree_update(octree_out, label, d, split=1)
        if d < self.depth_out:
          octree_out = ocnn.octree_grow(octree_out, target_depth=d+1)
        doctree_out = dual_octree.DualOctree(octree_out)
        doctree_out.post_processing_for_ocnn()

      # predict the signal
      reg_vox = self.regress[i](deconvs[d])
      reg_voxs_list.append(reg_vox)
      reg_voxs[d] = torch.cat(reg_voxs_list, dim=0)

    return logits, reg_voxs, doctree_out.octree

  def forward(self, octree_in, octree_out=None, pos=None):
    # generate dual octrees
    doctree_in = dual_octree.DualOctree(octree_in)
    doctree_in.post_processing_for_ocnn()

    update_octree = octree_out is None
    if update_octree:
      octree_out = ocnn.create_full_octree(self.full_depth, self.nout)
    doctree_out = dual_octree.DualOctree(octree_out)
    doctree_out.post_processing_for_ocnn()

    # run encoder and decoder
    convs = self.octree_encoder(octree_in, doctree_in)
    out = self.octree_decoder(convs, doctree_out, doctree_in, update_octree)
    output = {'logits': out[0], 'reg_voxs': out[1], 'octree_out': out[2]}

    # mpus
    if pos is not None:
      output['mpus'] = self.neural_mpu(pos, out[1], octree_out)

    return output
