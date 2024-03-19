"""
DETR model and criterion classes.
"""
from multiprocessing import Condition
import torch
import torch.nn.functional as F
from torch import nn

from .util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

import copy
import numpy as np
from detectron2.utils.registry import Registry
import math 

DETR_REGISTRY = Registry("DETR_REGISTRY")


@DETR_REGISTRY.register()
class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False, use_gt_box=False, use_gt_label=False, **kwargs):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

@DETR_REGISTRY.register()
class IterativeRelationDETR(DETR):
    def __init__(self, backbone, transformer, num_classes, num_queries,num_relation_queries, aux_loss=False, use_gt_box=False, use_gt_label=False,cfg=None, **kwargs):
        super().__init__(backbone=backbone, transformer=transformer, num_classes=num_classes, num_queries=num_queries, aux_loss=aux_loss, use_gt_box=use_gt_box, use_gt_label=use_gt_label, **kwargs)

        self.relation_query_embed = nn.Embedding(num_relation_queries, transformer.d_model)
        self.object_query_embed = nn.Embedding(num_queries, transformer.d_model)

        del self.class_embed
        del self.bbox_embed

        self.only_predicate_multiply = cfg.MODEL.DETR.ONLY_PREDICATE_MULTIPLY
        self.multiply_query = cfg.MODEL.DETR.MULTIPLY_QUERY

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        features, pos = self.backbone(samples)
        
        src, mask = features[-1].decompose()
        assert mask is not None
        output = self.transformer(self.input_proj(src), mask, self.query_embed.weight, self.object_query_embed.weight, self.relation_query_embed.weight, pos[-1])

        if self.only_predicate_multiply:
            output['relation_subject_logits'] = output['relation_subject_logits'].repeat_interleave(self.multiply_query,2)
            output['relation_object_logits'] = output['relation_object_logits'].repeat_interleave(self.multiply_query,2)
            output['relation_subject_coords'] = output['relation_subject_coords'].repeat_interleave(self.multiply_query,2)
            output['relation_object_coords'] = output['relation_object_coords'].repeat_interleave(self.multiply_query,2)

        out = dict()

        out['relation_boxes'] = output['relation_coords'][-1]
        out['relation_logits'] = output['relation_logits'][-1]
        out['relation_subject_logits'] = output['relation_subject_logits'][-1]
        out['relation_object_logits'] = output['relation_object_logits'][-1]
        out['relation_subject_boxes'] = output['relation_subject_coords'][-1]
        out['relation_object_boxes'] = output['relation_object_coords'][-1]

        if self.aux_loss:
            out['aux_outputs_r'] = self._set_aux_loss(output['relation_logits'], output['relation_coords'])
            out['aux_outputs_r_sub'] = self._set_aux_loss(output['relation_subject_logits'], output['relation_subject_coords'])
            out['aux_outputs_r_obj'] = self._set_aux_loss(output['relation_object_logits'], output['relation_object_coords'])
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if outputs_coord is not None:
            return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
        else:
            return [{'pred_logits': a}
                for a in outputs_class[:-1]]



class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2)
    return pos

def build_detr(name, backbone, transformer, num_classes, num_queries,num_relation_queries, aux_loss=False, use_gt_box=False, use_gt_label=False,cfg=None, **kwargs):
    return DETR_REGISTRY.get(name)(backbone, transformer, num_classes, num_queries,num_relation_queries, aux_loss=aux_loss, use_gt_box=use_gt_box, use_gt_label=use_gt_label,cfg=cfg, **kwargs)