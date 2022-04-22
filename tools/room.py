# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import wget
import time
import zipfile
import argparse
import numpy as np
from tqdm import tqdm
from plyfile import PlyData, PlyElement

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=str, required=True)
args = parser.parse_args()

project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
root_folder = os.path.join(project_folder, 'data/room')


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


def download_and_unzip():
  r''' Dowanload and unzip the data.
  '''

  filename = os.path.join(root_folder, 'synthetic_room_dataset.zip')
  flag_file = os.path.join(root_folder, 'flags/download_room_dataset_succ')
  if not os.path.exists(flag_file):
    check_folder([filename])
    print('-> Download synthetic_room_dataset.zip.')
    url = 'https://s3.eu-central-1.amazonaws.com/avg-projects/convolutional_occupancy_networks/data/synthetic_room_dataset.zip'
    wget.download(url, filename)
    create_flag_file(flag_file)

  flag_file = os.path.join(root_folder, 'flags/unzip_succ')
  if not os.path.exists(flag_file):
    print('-> Unzip synthetic_room_dataset.zip.')
    with zipfile.ZipFile(filename, 'r') as zip_ref:
      zip_ref.extractall(path=root_folder)
    # os.remove(filename)
    create_flag_file(flag_file)


def download_filelist():
  r''' Downloads the filelists used for learning.
  '''

  flag_file = os.path.join(root_folder, 'flags/download_filelist_succ')
  if not os.path.exists(flag_file):
    print('-> Download the filelist.')
    url = 'https://www.dropbox.com/s/30v6pdek6777vkr/room.filelist.zip?dl=1'
    filename = os.path.join(root_folder, 'filelist.zip')
    wget.download(url, filename, bar=None)

    folder = os.path.join(root_folder, 'filelist')
    with zipfile.ZipFile(filename, 'r') as zip_ref:
      zip_ref.extractall(path=folder)
    # os.remove(filename)
    create_flag_file(flag_file)


def download_ground_truth_mesh():
  r''' Downloads the ground-truth meshes
  '''

  flag_file = os.path.join(root_folder, 'flags/download_mesh_succ')
  if not os.path.exists(flag_file):
    print('-> Download the filelist.')
    url = 'https://s3.eu-central-1.amazonaws.com/avg-projects/convolutional_occupancy_networks/data/room_watertight_mesh.zip'
    filename = os.path.join(root_folder, 'filelist.zip')
    wget.download(url, filename, bar=None)
    create_flag_file(flag_file)


def generate_test_points():
  r''' Generates points in `ply` format for testing.
  '''

  noise_std = 0.005
  point_sample_num = 10000
  print('-> Generate testing points.')
  # filenames = get_filenames('all.txt')
  filenames = get_filenames('test.txt', root_folder)
  for filename in tqdm(filenames, ncols=80):
    filename_pts = os.path.join(
        root_folder, 'synthetic_room_dataset', filename, 'pointcloud', 'pointcloud_00.npz')
    filename_ply = os.path.join(
        root_folder, 'test.input', filename + '.ply')
    if not os.path.exists(filename_pts): continue
    check_folder([filename_ply])

    # sample points
    pts = np.load(filename_pts)
    points = pts['points'].astype(np.float32)
    noise = noise_std * np.random.randn(point_sample_num, 3)
    rand_idx = np.random.choice(points.shape[0], size=point_sample_num)
    points_noise = points[rand_idx] + noise

    # save ply
    vertices = []
    py_types = (float, float, float)
    npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    for idx in range(points_noise.shape[0]):
      vertices.append(
          tuple(dtype(d) for dtype, d in zip(py_types, points_noise[idx])))
    structured_array = np.array(vertices, dtype=npy_types)
    el = PlyElement.describe(structured_array, 'vertex')
    PlyData([el]).write(filename_ply)


def download_test_points():
  r''' Downloads the test points used in our paper.
  '''
  print('-> Download testing points.')
  flag_file = os.path.join(root_folder, 'flags/download_test_points_succ')
  if not os.path.exists(flag_file):
    url = 'https://www.dropbox.com/s/q3h47042xh6sua7/scene.test.input.zip?dl=1'
    filename = os.path.join(root_folder, 'test.input.zip')
    wget.download(url, filename, bar=None)

    folder = os.path.join(root_folder, 'test.input')
    with zipfile.ZipFile(filename, 'r') as zip_ref:
      zip_ref.extractall(path=folder)
    # os.remove(filename)
    create_flag_file(flag_file)


def generate_dataset():
  download_and_unzip()
  download_filelist()
  # generate_test_points()
  download_test_points()


if __name__ == '__main__':
  eval('%s()' % args.run)
