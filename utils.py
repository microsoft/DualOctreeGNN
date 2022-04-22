# autopep8: off
import ocnn
import torch
import torch.autograd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skimage.measure
import trimesh
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
# autopep8: on


def get_mgrid(size, dim=3):
  r'''
  Example: 
  >>> get_mgrid(3, dim=2)  
      array([[0.0,  0.0],
             [0.0,  1.0],
             [0.0,  2.0],
             [1.0,  0.0],
             [1.0,  1.0],
             [1.0,  2.0],
             [2.0,  0.0],
             [2.0,  1.0],
             [2.0,  2.0]], dtype=float32)
  '''
  coord = np.arange(0, size, dtype=np.float32)
  coords = [coord] * dim
  output = np.meshgrid(*coords, indexing='ij')
  output = np.stack(output, -1)
  output = output.reshape(size**dim, dim)
  return output


def lin2img(tensor):
  channels = 1
  num_samples = tensor.shape
  size = int(np.sqrt(num_samples))
  return tensor.view(channels, size, size)


def make_contour_plot(array_2d, mode='log'):
  fig, ax = plt.subplots(figsize=(2.75, 2.75), dpi=300)

  if(mode == 'log'):
    nlevels = 6
    levels_pos = np.logspace(-2, 0, num=nlevels)  # logspace
    levels_neg = -1. * levels_pos[::-1]
    levels = np.concatenate((levels_neg, np.zeros((0)), levels_pos), axis=0)
    colors = plt.get_cmap("Spectral")(np.linspace(0., 1., num=nlevels * 2 + 1))
  elif(mode == 'lin'):
    nlevels = 10
    levels = np.linspace(-.5, .5, num=nlevels)
    colors = plt.get_cmap("Spectral")(np.linspace(0., 1., num=nlevels))
  else:
    raise NotImplementedError

  sample = np.flipud(array_2d)
  CS = ax.contourf(sample, levels=levels, colors=colors)
  cbar = fig.colorbar(CS)

  ax.contour(sample, levels=levels, colors='k', linewidths=0.1)
  ax.contour(sample, levels=[0], colors='k', linewidths=0.3)
  ax.axis('off')
  return fig


def write_sdf_summary(model, writer, global_step, alias=''):
  size = 128
  coords_2d = get_mgrid(size, dim=2)
  coords_2d = coords_2d / size - 1.0   # [0, size] -> [-1, 1]
  coords_2d = torch.from_numpy(coords_2d)
  with torch.no_grad():
    zeros = torch.zeros_like(coords_2d[:, :1])
    ones = torch.ones_like(coords_2d[:, :1])
    names = ['train_yz_sdf_slice', 'train_xz_sdf_slice', 'train_xy_sdf_slice']
    coords = [torch.cat((zeros, coords_2d), dim=-1),
              torch.cat((coords_2d[:, :1], zeros, coords_2d[:, -1:]), dim=-1),
              torch.cat((coords_2d, -0.75 * ones), dim=-1)]
    for name, coord in zip(names, coords):
      ids = torch.zeros(coord.shape[0], 1)
      coord = torch.cat([coord, ids], dim=1).cuda()
      sdf_values = model(coord)
      sdf_values = lin2img(sdf_values).squeeze().cpu().numpy()
      fig = make_contour_plot(sdf_values)
      writer.add_figure(alias + name, fig, global_step=global_step)


def calc_sdf(model, size=256, max_batch=64**3, bbmin=-1.0, bbmax=1.0):
  # generate samples
  num_samples = size ** 3
  samples = get_mgrid(size, dim=3)
  samples = samples * ((bbmax - bbmin) / size) + bbmin  # [0,sz]->[bbmin,bbmax]
  samples = torch.from_numpy(samples)
  sdfs = torch.zeros(num_samples)

  # forward
  head = 0
  while head < num_samples:
    tail = min(head + max_batch, num_samples)
    sample_subset = samples[head:tail, :]
    idx = torch.zeros(sample_subset.shape[0], 1)
    pts = torch.cat([sample_subset, idx], dim=1).cuda()
    pred = model(pts).squeeze().detach().cpu()
    sdfs[head:tail] = pred
    head += max_batch
  sdfs = sdfs.reshape(size, size, size).numpy()
  return sdfs


def create_mesh(model, filename, size=256, max_batch=64**3, level=0,
                bbmin=-0.9, bbmax=0.9, mesh_scale=1.0, save_sdf=False, **kwargs):
  # marching cubes
  sdf_values = calc_sdf(model, size, max_batch, bbmin, bbmax)
  vtx, faces = np.zeros((0, 3)), np.zeros((0, 3))
  try:
    vtx, faces, _, _ = skimage.measure.marching_cubes(sdf_values, level)
  except:
    pass
  if vtx.size == 0 or faces.size == 0:
    print('Warning from marching cubes: Empty mesh!')
    return

  # normalize vtx
  vtx = vtx * ((bbmax - bbmin) / size) + bbmin   # [0,sz]->[bbmin,bbmax]
  vtx = vtx * mesh_scale                         # rescale

  # save to ply and npy
  mesh = trimesh.Trimesh(vtx, faces)
  mesh.export(filename)
  if save_sdf:
    np.save(filename[:-4] + ".sdf.npy", sdf_values)


def calc_sdf_err(filename_gt, filename_pred):
  scale = 1.0e2  # scale the result for better display
  sdf_gt = np.load(filename_gt)
  sdf = np.load(filename_pred)
  err = np.abs(sdf - sdf_gt).mean() * scale
  return err


def calc_chamfer(filename_gt, filename_pred, point_num):
  scale = 1.0e5  # scale the result for better display
  np.random.seed(101)

  mesh_a = trimesh.load(filename_gt)
  points_a, _ = trimesh.sample.sample_surface(mesh_a, point_num)
  mesh_b = trimesh.load(filename_pred)
  points_b, _ = trimesh.sample.sample_surface(mesh_b, point_num)

  kdtree_a = cKDTree(points_a)
  dist_a, _ = kdtree_a.query(points_b)
  chamfer_a = np.mean(np.square(dist_a)) * scale

  kdtree_b = cKDTree(points_b)
  dist_b, _ = kdtree_b.query(points_a)
  chamfer_b = np.mean(np.square(dist_b)) * scale
  return chamfer_a, chamfer_b


def points2ply(filename, points, scale=1.0):
  xyz = ocnn.points_property(points, 'xyz')
  normal = ocnn.points_property(points, 'normal')
  has_normal = normal is not None
  xyz = xyz.numpy() * scale
  if has_normal: normal = normal.numpy()

  # data types
  data = xyz
  py_types = (float, float, float)
  npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
  if has_normal:
    py_types = py_types + (float, float, float)
    npy_types = npy_types + [('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')]
    data = np.concatenate((data, normal), axis=1)

  # format into NumPy structured array
  vertices = []
  for idx in range(data.shape[0]):
    vertices.append(tuple(dtype(d) for dtype, d in zip(py_types, data[idx])))
  structured_array = np.array(vertices, dtype=npy_types)
  el = PlyElement.describe(structured_array, 'vertex')

  # write ply
  PlyData([el]).write(filename)
