import torch

from detectron2.structures.boxes import Boxes

def masks_union(masks1, masks2):
  assert len(masks1) == len(masks2)
  masks_union = (masks1 + masks2)/2.0
  return masks_union
