# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import ocnn
import torch
import torch.nn

from . import mpu


class OctreeOUNet(ocnn.OUNet):

  def __init__(self, depth, channel_in, nout, full_depth=2, depth_out=6):
    super().__init__(depth, channel_in, nout, full_depth)

    self.header = None
    self.depth_out = depth_out

    self.neural_mpu = mpu.NeuralMPU(self.full_depth, self.depth_out)
    self.regress = torch.nn.ModuleList(
        [self._make_predict_module(self.channels[d], 4)
         for d in range(full_depth, depth + 1)])

  def ocnn_decoder(self, convs, octree_out, octree, update_octree=False):
    logits = dict()
    reg_voxs = dict()
    deconvs = dict()
    reg_voxs_list = []

    deconvs[self.full_depth] = convs[self.full_depth]
    for i, d in enumerate(range(self.full_depth, self.depth_out + 1)):
      if d > self.full_depth:
        deconvd = self.upsample[i - 1](deconvs[d - 1], octree_out)
        skip, _ = ocnn.octree_align(convs[d], octree, octree_out, d)
        deconvd = deconvd + skip
        deconvs[d] = self.decoder[i - 1](deconvd, octree_out)

      # predict the splitting label
      logit = self.predict[i](deconvs[d])
      logit = logit.squeeze().t()  # (1, C, H, 1) -> (H, C)
      logits[d] = logit

      # update the octree according to predicted labels
      if update_octree:
        label = logits[d].argmax(1).to(torch.int32)
        octree_out = ocnn.octree_update(octree_out, label, d, split=1)
        if d < self.depth_out:
          octree_out = ocnn.octree_grow(octree_out, target_depth=d + 1)

      # predict the signal
      reg_vox = self.regress[i](deconvs[d])
      reg_vox = reg_vox.squeeze().t()    # (1, C, H, 1) -> (H, C)
      reg_voxs_list.append(reg_vox)
      reg_voxs[d] = torch.cat(reg_voxs_list, dim=0)

    return logits, reg_voxs, octree_out

  def forward(self, octree_in, octree_out=None, pos=None):
    update_octree = octree_out is None
    if update_octree:
      octree_out = ocnn.create_full_octree(self.full_depth, self.nout)

    # run encoder and decoder
    convs = self.ocnn_encoder(octree_in)
    out = self.ocnn_decoder(convs, octree_out, octree_in, update_octree)
    output = {'logits': out[0], 'reg_voxs': out[1], 'octree_out': out[2]}

    # compute function value with mpu
    if pos is not None:
      output['mpus'] = self.neural_mpu(pos, out[1], out[2])

    # create the mpu wrapper
    def _neural_mpu(pos):
      pred = self.neural_mpu(pos, out[1], out[2])
      return pred[self.depth_out][0]
    output['neural_mpu'] = _neural_mpu

    return output
