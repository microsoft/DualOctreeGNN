# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import ocnn
import torch
import torch.nn.functional as F


def compute_gradient(y, x):
  grad_outputs = torch.ones_like(y)
  grad = torch.autograd.grad(y, [x], grad_outputs, create_graph=True)[0]
  return grad


def sdf_reg_loss(sdf, grad, sdf_gt, grad_gt, name_suffix=''):
  wg, ws = 1.0, 200.0
  grad_loss = (grad - grad_gt).pow(2).mean() * wg
  sdf_loss = (sdf - sdf_gt).pow(2).mean() * ws
  loss_dict = {'grad_loss' + name_suffix: grad_loss,
               'sdf_loss' + name_suffix: sdf_loss}
  return loss_dict


def sdf_grad_loss(sdf, grad, sdf_gt, grad_gt, name_suffix=''):
  on_surf = sdf_gt != -1
  off_surf = on_surf.logical_not()

  sdf_loss = sdf[on_surf].pow(2).mean() * 200.0
  norm_loss = (grad[on_surf] - grad_gt[on_surf]).pow(2).mean() * 1.0
  intr_loss = torch.exp(-40 * torch.abs(sdf[off_surf])).mean() * 0.1
  grad_loss = (grad[off_surf].norm(2, dim=-1) - 1).abs().mean() * 0.1

  losses = [sdf_loss, intr_loss, norm_loss, grad_loss]
  names = ['sdf_loss', 'inter_loss', 'norm_loss', 'grad_loss']
  names = [name + name_suffix for name in names]
  loss_dict = dict(zip(names, losses))
  return loss_dict


def sdf_grad_regularized_loss(sdf, grad, sdf_gt, grad_gt, name_suffix=''):
  on_surf = sdf_gt != -1
  off_surf = on_surf.logical_not()

  sdf_loss = sdf[on_surf].pow(2).mean() * 200.0
  norm_loss = (grad[on_surf] - grad_gt[on_surf]).pow(2).mean() * 1.0
  intr_loss = torch.exp(-40 * torch.abs(sdf[off_surf])).mean() * 0.1
  grad_loss = (grad[off_surf].norm(2, dim=-1) - 1).abs().mean() * 0.1
  grad_reg_loss = (grad[off_surf] - grad_gt[off_surf]).pow(2).mean() * 0.1

  losses = [sdf_loss, intr_loss, norm_loss, grad_loss, grad_reg_loss]
  names = ['sdf_loss', 'inter_loss', 'norm_loss', 'grad_loss', 'grad_reg_loss']
  names = [name + name_suffix for name in names]
  loss_dict = dict(zip(names, losses))
  return loss_dict


def possion_grad_loss(sdf, grad, sdf_gt, grad_gt, name_suffix=''):
  on_surf = sdf_gt == 0
  out_of_bbox = sdf_gt == 1.0
  off_surf = on_surf.logical_not()

  sdf_loss = sdf[on_surf].pow(2).mean() * 200.0
  norm_loss = (grad[on_surf] - grad_gt[on_surf]).pow(2).mean() * 1.0
  intr_loss = torch.exp(-40 * torch.abs(sdf[off_surf])).mean() * 0.1
  grad_loss = grad[off_surf].pow(2).mean() * 0.1  # poisson loss
  bbox_loss = torch.mean(torch.relu(-sdf[out_of_bbox])) * 100.0

  losses = [sdf_loss, intr_loss, norm_loss, grad_loss, bbox_loss]
  names = ['sdf_loss', 'inter_loss', 'norm_loss', 'grad_loss', 'bbox_loss']
  names = [name + name_suffix for name in names]
  loss_dict = dict(zip(names, losses))
  return loss_dict


def compute_mpu_gradients(mpus, pos, fval_transform=None):
  grads = dict()
  for d in mpus.keys():
    fval, flags = mpus[d]
    if fval_transform is not None:
      fval = fval_transform(fval)
    grads[d] = compute_gradient(fval, pos)[:, :3]
  return grads


def compute_octree_loss(logits, octree_out):
  weights = [1.0] * 16
  # weights = [1.0] * 4 + [0.8, 0.6, 0.4] + [0.2] * 16

  output = dict()
  for d in logits.keys():
    logitd = logits[d]
    label_gt = ocnn.octree_property(octree_out, 'split', d).long()
    output['loss_%d' % d] = F.cross_entropy(logitd, label_gt) * weights[d]
    output['accu_%d' % d] = logitd.argmax(1).eq(label_gt).float().mean()
  return output


def compute_sdf_loss(mpus, grads, sdf_gt, grad_gt, reg_loss_func):
  output = dict()
  for d in mpus.keys():
    sdf, flgs = mpus[d]  # TODO: tune the loss weights and `flgs`
    reg_loss = reg_loss_func(sdf, grads[d], sdf_gt, grad_gt, '_%d' % d)
    # if d < 3:  # ignore depth 2
    #   for key in reg_loss.keys():
    #     reg_loss[key] = reg_loss[key] * 0.0
    output.update(reg_loss)
  return output


def compute_occu_loss_v0(mpus, grads, occu_gt, grad_gt, weight):
  output = dict()
  for d in mpus.keys():
    occu, flgs, grad = mpus[d]

    # pos_weight = torch.ones_like(occu_gt) * 10.0
    loss_o = F.binary_cross_entropy_with_logits(occu, occu_gt, weight=weight)
    # loss_g = torch.mean((grad - grad_gt) ** 2)

    occu = torch.sigmoid(occu)
    non_surface_points = occu_gt != 0.5
    accu = (occu > 0.5).eq(occu_gt).float()[non_surface_points].mean()

    output['occu_loss_%d' % d] = loss_o
    # output['grad_loss_%d' % d] = loss_g
    output['occu_accu_%d' % d] = accu
  return output


def compute_occu_loss_1214(mpus, occu):
  # tried on 2021.12.13
  weights = [0.2] * 4 + [0.4, 0.6, 0.8] + [1.0] * 16  # TODO: tune the weights

  inside = occu == 0
  outside = occu == 1
  output = dict()
  for d in mpus.keys():
    sdf, flgs = mpus[d]

    inside_loss = torch.mean(torch.relu(sdf[inside])) * (1000 * weights[d])
    outside_loss = torch.mean(torch.relu(-sdf[outside])) * (1000 * weights[d])

    output['inside_loss_%d' % d] = inside_loss
    output['outside_loss_%d' % d] = outside_loss
    output['inside_accu_%d' % d] = (sdf[inside] < 0).float().mean()
    output['outside_accu_%d' % d] = (sdf[outside] > 0).float().mean()
  return output


def compute_occu_loss(mpus, grads, occu, grad_gt):
  weights = [1.0] * 16
  # weights = [0.2] * 4 + [0.4, 0.6, 0.8] + [1.0] * 16
  # weights = [0.0] * 7 + [1.0] * 16  # Single level loss

  inside = occu == 0
  outside = occu == 1
  on_surf = occu == 0.5
  off_surf = on_surf.logical_not()

  output = dict()
  for d in mpus.keys():
    sdf, flgs = mpus[d]
    grad = grads[d]
    grad_diff = grad[on_surf] - grad_gt[on_surf]

    sdf_loss = sdf[on_surf].pow(2).mean() * (200 * weights[d])
    norm_loss = grad_diff.pow(2).mean() * (1.0 * weights[d])
    intr_loss = torch.exp(-40 * sdf[off_surf].abs()).mean() * (0.1 * weights[d])
    grad_loss = grad[off_surf].pow(2).mean() * (0.1 * weights[d])

    inside_loss = torch.mean(torch.relu(sdf[inside])) * (500 * weights[d])
    outside_loss = torch.mean(torch.relu(-sdf[outside])) * (2000 * weights[d])
    inside_accu = (sdf[inside] < 0).float().mean()
    outside_accu = (sdf[outside] > 0).float().mean()

    losses = [sdf_loss, norm_loss, grad_loss, intr_loss,
              inside_loss, inside_accu, outside_loss, outside_accu]
    names = ['sdf_loss', 'norm_loss', 'grad_loss', 'inter_loss',
             'inside_loss', 'inside_accu', 'outside_loss', 'outside_accu']
    names = [name + ('_%d' % d) for name in names]
    loss_dict = dict(zip(names, losses))
    output.update(loss_dict)

  return output


def compute_occu_loss_cls(mpus, grads, occu_gt, grad_gt):
  weights = [1.0] * 16
  # weights = [0.2] * 4 + [0.4, 0.6, 0.8] + [1.0] * 16
  # weights = [0.0] * 7 + [1.0] * 16  # Single level loss

  inside = occu_gt == 0
  outside = occu_gt == 1
  on_surf = occu_gt == 0.5
  off_surf = on_surf.logical_not()

  # Use soft-version occupancies
  occu_soft = torch.ones_like(occu_gt) * 0.5
  occu_soft[inside] = 0.3
  occu_soft[outside] = 0.7

  output = dict()
  for d in mpus.keys():
    sdf, flgs = mpus[d]
    grad = grads[d]
    grad_diff = grad[on_surf] - grad_gt[on_surf]

    sdf_loss = sdf[on_surf].pow(2).mean() * (200 * weights[d])
    norm_loss = grad_diff.pow(2).mean() * (1.0 * weights[d])
    intr_loss = torch.exp(-40 * sdf[off_surf].abs()).mean() * (0.1 * weights[d])
    grad_loss = grad[off_surf].pow(2).mean() * (0.1 * weights[d])

    loss_o = F.binary_cross_entropy_with_logits(sdf, occu_soft)
    accu = (sdf.sigmoid() > 0.5).eq(occu_gt).float()
    inside_accu = accu[inside].mean()
    outside_accu = accu[outside].mean()

    losses = [sdf_loss, norm_loss, grad_loss, intr_loss,
              loss_o, inside_accu, outside_accu]
    names = ['sdf_loss', 'norm_loss', 'grad_loss', 'inter_loss',
             'occu_loss', 'inside_accu', 'outside_accu']
    names = [name + ('_%d' % d) for name in names]
    loss_dict = dict(zip(names, losses))
    output.update(loss_dict)

  return output


def get_sdf_loss_function(loss_type=''):
  if loss_type == 'sdf_reg_loss':
    return sdf_reg_loss
  elif loss_type == 'sdf_grad_loss':
    return sdf_grad_loss
  elif loss_type == 'possion_grad_loss':
    return possion_grad_loss
  elif loss_type == 'sdf_grad_reg_loss':
    return sdf_grad_regularized_loss
  else:
    return None


def shapenet_loss(batch, model_out, reg_loss_type=''):
  # octree loss
  output = compute_octree_loss(model_out['logits'], model_out['octree_out'])

  # regression loss
  grads = compute_mpu_gradients(model_out['mpus'], batch['pos'])
  reg_loss_func = get_sdf_loss_function(reg_loss_type)
  sdf_loss = compute_sdf_loss(
      model_out['mpus'], grads, batch['sdf'], batch['grad'], reg_loss_func)
  output.update(sdf_loss)
  return output


def dfaust_loss(batch, model_out, reg_loss_type=''):
  # there is no octree loss
  grads = compute_mpu_gradients(model_out['mpus'], batch['pos'])
  reg_loss_func = get_sdf_loss_function(reg_loss_type)
  output = compute_sdf_loss(
      model_out['mpus'], grads, batch['sdf'], batch['grad'], reg_loss_func)
  return output


def synthetic_room_loss(batch, model_out, reg_loss_type=''):
  # octree loss
  output = compute_octree_loss(model_out['logits'], model_out['octree_out'])

  # grads
  grads = compute_mpu_gradients(model_out['mpus'], batch['pos'])

  # occu loss
  occu_loss = compute_occu_loss(
      model_out['mpus'], grads, batch['occu'], batch['grad'])
  output.update(occu_loss)

  return output
