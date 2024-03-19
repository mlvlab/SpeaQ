import torch

from detectron2.structures.boxes import Boxes

def boxes_union(boxes1, boxes2):
    """
    Compute the union region of two set of boxes
    Arguments:
      box1: (Boxes) bounding boxes, sized [N,4].
      box2: (Boxes) bounding boxes, sized [N,4].
    Returns:
      (Boxes) union, sized [N,4].
    """
    assert len(boxes1) == len(boxes2)

    union_box = torch.cat((
        torch.min(boxes1.tensor[:,:2], boxes2.tensor[:,:2]),
        torch.max(boxes1.tensor[:,2:], boxes2.tensor[:,2:])
        ),dim=1)
    return Boxes(union_box)
