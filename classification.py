import ocnn
import torch

from solver import Solver, Dataset
from builder import get_classification_model


class ClsSolver(Solver):

  def get_model(self, flags):
    return get_classification_model(flags)

  def get_dataset(self, flags):
    transform = ocnn.TransformCompose(flags)
    dataset = Dataset(flags.location, flags.filelist, transform, in_memory=True)
    return dataset, ocnn.collate_octrees

  def train_step(self, batch):
    octree, label = batch['octree'].cuda(), batch['label'].cuda()
    logits = self.model(octree)
    log_softmax = torch.nn.functional.log_softmax(logits, dim=1)
    loss = torch.nn.functional.nll_loss(log_softmax, label)
    return {'train/loss': loss}

  def test_step(self, batch):
    octree, label = batch['octree'].cuda(), batch['label'].cuda()
    logits = self.model(octree)
    log_softmax = torch.nn.functional.log_softmax(logits, dim=1)
    loss = torch.nn.functional.nll_loss(log_softmax, label)
    pred = torch.argmax(logits, dim=1)
    accu = pred.eq(label).float().mean()
    return {'test/loss': loss, 'test/accu': accu}


if __name__ == "__main__":
  ClsSolver.main()
