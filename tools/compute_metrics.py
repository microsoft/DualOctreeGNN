# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import argparse
import trimesh.sample
import numpy as np
from tqdm import tqdm
from scipy.spatial import cKDTree

parser = argparse.ArgumentParser()
parser.add_argument('--mesh_folder', type=str, required=True)
parser.add_argument('--filename_out', type=str, required=True)
parser.add_argument('--num_samples', type=int, default=30000)
parser.add_argument('--ref_folder', type=str,  default='data/dfaust/dfaust/mesh_gt')
parser.add_argument('--filelist', type=str, default='data/dfaust/test_all.txt')
args = parser.parse_args()


with open(args.filelist, 'r') as fid:
  lines = fid.readlines()
filenames = [line.strip() for line in lines]


def compute_metrics(filename_ref, filename_pred, num_samples=30000):
  mesh_ref = trimesh.load(filename_ref)
  points_ref, idx_ref = trimesh.sample.sample_surface(mesh_ref, num_samples)
  normals_ref = mesh_ref.face_normals[idx_ref]
  # points_ref, normals_ref = read_ply(filename_ref)

  mesh_pred = trimesh.load(filename_pred)
  points_pred, idx_pred = trimesh.sample.sample_surface(mesh_pred, num_samples)
  normals_pred = mesh_pred.face_normals[idx_pred]

  kdtree_a = cKDTree(points_ref)
  dist_a, idx_a = kdtree_a.query(points_pred)
  chamfer_a = np.mean(dist_a)
  dot_a = np.sum(normals_pred * normals_ref[idx_a], axis=1)
  angle_a = np.mean(np.arccos(dot_a) * (180.0 / np.pi))
  consist_a = np.mean(np.abs(dot_a))

  kdtree_b = cKDTree(points_pred)
  dist_b, idx_b = kdtree_b.query(points_ref)
  chamfer_b = np.mean(dist_b)
  dot_b = np.sum(normals_ref * normals_pred[idx_b], axis=1)
  angle_b = np.mean(np.arccos(dot_b) * (180 / np.pi))
  consist_b = np.mean(np.abs(dot_b))

  return chamfer_a, chamfer_b, angle_a, angle_b, consist_a, consist_b


counter = 0
fid = open(args.filename_out, 'w')
fid.write(('name, '
           'chamfer_a, chamfer_b, chamfer, '
           'angle_a, angle_b, angle, '
           'consist_a, consist_b, normal consistency\n'))
for filename in tqdm(filenames, ncols=80):
  if filename.endswith('.npy'):
    filename = filename[:-4]
  filename_ref = os.path.join(args.ref_folder, filename + '.obj')
  # filename_ref = os.path.join(args.ref_folder, filename + '.ply')
  filename_pred = os.path.join(args.mesh_folder, filename + '.obj')
  metrics = compute_metrics(filename_ref, filename_pred, args.num_samples)

  chamfer_a, chamfer_b = metrics[0], metrics[1]
  angle_a, angle_b = metrics[2], metrics[3]
  consist_a, consist_b = metrics[4], metrics[5]

  msg = '{}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(
      filename,
      chamfer_a, chamfer_b, 0.5 * (chamfer_a + chamfer_b),
      angle_a, angle_b, 0.5 * (angle_a + angle_b),
      consist_a, consist_b, 0.5 * (consist_a + consist_b))
  fid.write(msg)
  tqdm.write(msg)

fid.close()
