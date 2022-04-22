# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import argparse
import trimesh
import trimesh.sample
import numpy as np
import time
import zipfile
import wget
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--run', type=str, default='prepare_dataset')
parser.add_argument('--filelist', type=str, default='test.txt')
parser.add_argument('--mesh_folder', type=str, default='logs/dfaust/mesh')
parser.add_argument('--output_folder', type=str, default='logs/dfaust/mesh')
args = parser.parse_args()


project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
root_folder = os.path.join(project_folder, 'data/dfaust')
shape_scale = 0.8


def create_flag_file(filename):
  r''' Creates a flag file to indicate whether some time-consuming works
  have been done.
  '''

  folder = os.path.dirname(filename)
  if not os.path.exists(folder):
    os.makedirs(folder)
  with open(filename, 'w') as fid:
    fid.write('succ @ ' + time.ctime())


def check_folder(filenames: list):
  r''' Checks whether the folder contains the filename exists.
  '''

  for filename in filenames:
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
      os.makedirs(folder)


def get_filenames(filelist, root_folder):
  r''' Gets filenames from a filelist.
  '''

  filelist = os.path.join(root_folder, 'filelist', filelist)
  with open(filelist, 'r') as fid:
    lines = fid.readlines()
  filenames = [line.split()[0] for line in lines]
  return filenames


def download_filelist():
  r''' Downloads the filelists used for learning.
  '''

  print('-> Download the filelist.')
  url = 'https://www.dropbox.com/s/vxkpaz3umzjvi66/dfaust.filelist.zip?dl=1'
  filename = os.path.join(root_folder, 'filelist.zip')
  wget.download(url, filename, bar=None)

  folder = os.path.join(root_folder, 'filelist')
  with zipfile.ZipFile(filename, 'r') as zip_ref:
    zip_ref.extractall(path=folder)
  os.remove(filename)


def sample_points():
  r''' Samples points from raw scanns for training.
  '''

  num_samples = 100000
  print('-> Sample points.')
  scans_folder = os.path.join(root_folder, 'scans')
  dataset_folder = os.path.join(root_folder, 'dataset')
  filenames = get_filenames('all.txt', root_folder)
  for filename in tqdm(filenames, ncols=80):
    filename_ply = os.path.join(scans_folder, filename + '.ply')
    filename_pts = os.path.join(dataset_folder, filename + '.npy')
    filename_center = filename_pts[:-3] + 'center.npy'
    check_folder([filename_pts])

    # sample points
    mesh = trimesh.load(filename_ply)
    points, idx = trimesh.sample.sample_surface(mesh, num_samples)
    normals = mesh.face_normals[idx]

    # normalize: Centralize + Scale
    center = np.mean(points, axis=0, keepdims=True)
    points = (points - center) * shape_scale
    point_cloud = np.concatenate((points, normals), axis=-1).astype(np.float32)

    # save
    np.save(filename_pts, point_cloud)
    np.save(filename_center, center)


def rescale_mesh():
  r''' Rescales and translates the generated mesh to align with the raw scans
  to compute evaluation metrics.
  '''

  filenames = get_filenames(args.filelist, root_folder)
  for filename in tqdm(filenames, ncols=80):
    filename = filename[:-4]
    filename_output = os.path.join(args.output_folder, filename + '.obj')
    filename_mesh = os.path.join(args.mesh_folder, filename + '.obj')
    filename_center = os.path.join(
        root_folder, 'dataset', filename + '.center.npy')
    check_folder([filename_output])

    center = np.load(filename_center)
    mesh = trimesh.load(filename_mesh)
    vertices = mesh.vertices / shape_scale + center
    mesh.vertices = vertices
    mesh.export(filename_output)


def generate_dataset():
  download_filelist()
  sample_points()


def download_dataset():
  download_filelist()

  print('-> Download the dataset.')
  flag_file = os.path.join(root_folder, 'flags/download_dataset_succ')
  if not os.path.exists(flag_file):
    url = 'https://www.dropbox.com/s/eb5uk8f2fqswhs3/dfaust.dataset.zip?dl=1'
    filename = os.path.join(root_folder, 'dfaust.dataset.zip')
    wget.download(url, filename, bar=None)

    with zipfile.ZipFile(filename, 'r') as zip_ref:
      zip_ref.extractall(path=root_folder)
    # os.remove(filename)
    create_flag_file(flag_file)


if __name__ == '__main__':
  eval('%s()' % args.run)
