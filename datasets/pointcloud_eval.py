import os
import ocnn
import torch
import numpy as np
from plyfile import PlyData

from solver import Dataset


class Transform:
  r"""Load point clouds from ply files, rescale the points and build octree.
  Used to evaluate the network trained on ShapeNet."""

  def __init__(self, flags):
    self.flags = flags

    self.point_scale = flags.point_scale
    self.points2octree = ocnn.Points2Octree(**flags)

  def __call__(self, points, idx):
    # After normalization, the points are in [-1, 1]
    pts = points[:, :3] / self.point_scale

    # construct the points
    ones = torch.ones(pts.shape[0], dtype=torch.float32)
    points = ocnn.points_new(pts, torch.Tensor(), ones, torch.Tensor())
    points, _ = ocnn.clip_points(points, [-1.0]*3, [1.0]*3)

    # transform points to octree
    octree = self.points2octree(points)

    return {'points_in': points, 'octree_in': octree}


def read_file(filename: str):
  plydata = PlyData.read(filename + '.ply')
  vtx = plydata['vertex']
  points = np.stack([vtx['x'], vtx['y'], vtx['z']], axis=1).astype(np.float32)
  output = torch.from_numpy(points.astype(np.float32))
  return output


def get_pointcloud_eval_dataset(flags):
  transform = Transform(flags)
  dataset = Dataset(flags.location, flags.filelist, transform,
                    read_file=read_file, in_memory=flags.in_memory)
  return dataset, ocnn.collate_octrees
