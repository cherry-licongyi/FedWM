import torch
from torchvision.models import vgg16
model = vgg16()

from torch.nn.utils import prune

class ThresholdPruning(prune.BasePruningMethod):
  PRUNING_TYPE = "unstructured"

  def __init__(self, threshold):
    self.threshold = threshold

  def compute_mask(self, tensor, default_mask):
    return torch.absolute(tensor) > self.threshold

# parameters_to_prune = (
#   (model.features[0], "weight"),
#   (model.features[2], "weight"),
#   # ... add more layers as needed
# )

# prune.global_unstructured(
#   parameters_to_prune,
#   pruning_method=ThresholdPruning,
#   threshold=0.1
# )