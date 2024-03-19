import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
import numpy as np
from torch import nn, Tensor
from detectron2.utils.registry import Registry
from .detr import MLP
from .detr import gen_sineembed_for_position
from .util.misc import inverse_sigmoid
import math
from detectron2.utils.events import get_event_storage
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, generalized_box_iou
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

TRANSFORMER_REGISTRY = Registry("TRANSFORMER_REGISTRY")

@TRANSFORMER_REGISTRY.register()
class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, **kwargs):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)

@TRANSFORMER_REGISTRY.register()
class IterativeRelationTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, cfg=None, **kwargs):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        layer_norm = nn.LayerNorm(d_model)
        relation_layer_norm = nn.LayerNorm(d_model)

        self.decoder = IterativeRelationDecoder(decoder_layer, num_decoder_layers, layer_norm, relation_layer_norm,
                                                      return_intermediate=return_intermediate_dec, d_model=d_model,
                                                    nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)

        self.d_model = d_model
        self.nhead = nhead
        self.object_embed = nn.Linear(d_model, kwargs['num_classes'] + 1)
        self.object_bbox_coords = MLP(d_model, d_model, 4, 3)

        self.relation_embed = nn.Linear(d_model, kwargs['num_relation_classes'] + 1)


        self.num_relation_classes = kwargs['num_relation_classes']
        self.num_object_classes = kwargs['num_classes']


        self._reset_parameters()
        for layer in range(self.decoder.num_layers - 1):
            nn.init.constant_(self.decoder.subject_graph_query_residual[layer].weight, 0)
            nn.init.constant_(self.decoder.subject_graph_query_residual[layer].bias, 0)

            nn.init.constant_(self.decoder.object_graph_query_residual[layer].weight, 0)
            nn.init.constant_(self.decoder.object_graph_query_residual[layer].bias, 0)

            nn.init.constant_(self.decoder.relation_graph_query_residual[layer].weight, 0)
            nn.init.constant_(self.decoder.relation_graph_query_residual[layer].bias, 0)

        for layer in range(self.decoder.num_layers):
            nn.init.constant_(self.decoder.object_pos_linear[layer].weight, 0)
            nn.init.constant_(self.decoder.object_pos_linear[layer].bias, 0)
            nn.init.constant_(self.decoder.relation_pos_linear[layer].weight, 0)
            nn.init.constant_(self.decoder.relation_pos_linear[layer].bias, 0)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, subject_embed, object_embed, relation_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        subject_query_embed = subject_embed.unsqueeze(1).repeat(1, bs, 1)
        object_query_embed = object_embed.unsqueeze(1).repeat(1, bs, 1)
        relation_query_embed = relation_embed.unsqueeze(1).repeat(1, bs, 1)

        # Condition on subject
        tgt_sub = torch.zeros_like(subject_query_embed)
        tgt_obj = torch.zeros_like(object_query_embed)
        tgt_rel = torch.zeros_like(relation_query_embed)


        hs_subject, hs_object, hs_relation = self.decoder(tgt_sub, tgt_obj, tgt_rel, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, subject_pos=subject_query_embed, object_pos=object_query_embed, relation_pos=relation_query_embed)
        relation_subject_class = self.object_embed(hs_subject)
        relation_subject_coords = self.object_bbox_coords(hs_subject).sigmoid()
        relation_object_class = self.object_embed(hs_object)
        relation_object_coords = self.object_bbox_coords(hs_object).sigmoid()
        relation_class = self.relation_embed(hs_relation)
        relation_coords = self.object_bbox_coords(hs_relation).sigmoid()

        output = {
            'relation_coords': relation_coords.transpose(1, 2),
            'relation_logits': relation_class.transpose(1, 2),
            'relation_subject_logits': relation_subject_class.transpose(1, 2),
            'relation_object_logits': relation_object_class.transpose(1, 2),
            'relation_subject_coords': relation_subject_coords.transpose(1, 2),
            'relation_object_coords': relation_object_coords.transpose(1, 2)
        }

        return output

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

class IterativeRelationDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, relation_norm=None, return_intermediate=False, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.subject_layers = _get_clones(decoder_layer, num_layers)
        self.object_layers = _get_clones(decoder_layer, num_layers)
        self.relation_layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.subject_norm = norm
        self.relation_norm = relation_norm
        self.return_intermediate = return_intermediate

        self.object_pos_attn = nn.ModuleList([nn.MultiheadAttention(d_model, nhead, dropout=dropout) for _ in range(num_layers)])
        self.object_pos_linear = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
        self.object_pos_dropout= nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])

        self.relation_pos_attn = nn.ModuleList([nn.MultiheadAttention(d_model, nhead, dropout=dropout, kdim=2*d_model, vdim=2*d_model) for _ in range(num_layers)])
        self.relation_pos_linear = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
        self.relation_pos_dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])

        self.subject_graph_query_attn = nn.ModuleList([nn.MultiheadAttention(d_model, nhead, dropout=dropout, kdim=3*d_model, vdim=3*d_model) for _ in range(num_layers-1)])
        self.subject_graph_query_residual = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers-1)])
        self.subject_graph_query_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers-1)])
        self.subject_graph_query_dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers-1)])

        self.object_graph_query_attn = nn.ModuleList([nn.MultiheadAttention(d_model, nhead, dropout=dropout, kdim=3*d_model, vdim=3*d_model) for _ in range(num_layers-1)])
        self.object_graph_query_residual = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers-1)])
        self.object_graph_query_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers-1)])
        self.object_graph_query_dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers-1)])

        self.relation_graph_query_attn = nn.ModuleList([nn.MultiheadAttention(d_model, nhead, dropout=dropout, kdim=3*d_model, vdim=3*d_model) for _ in range(num_layers-1)])
        self.relation_graph_query_residual = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers-1)])
        self.relation_graph_query_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers-1)])
        self.relation_graph_query_dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers-1)])

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def object_query_generator(self, object_query_embed, subject_features, subject_pos, layer):
        sub = self.with_pos_embed(subject_features, subject_pos)
        tgt = self.object_pos_attn[layer](object_query_embed, sub, value=subject_features)[0]
        tgt_residual = self.object_pos_linear[layer](tgt)
        object_query_embed = object_query_embed + self.object_pos_dropout[layer](tgt_residual)

        return object_query_embed

    def relation_query_generator(self, relation_query_embed, subject_features, object_features, subject_pos, object_pos, layer):
        sub = self.with_pos_embed(subject_features, subject_pos)
        obj = self.with_pos_embed(object_features, object_pos)

        k = torch.cat([sub, obj], -1)
        v = torch.cat([subject_features, object_features], -1)
        tgt = self.relation_pos_attn[layer](relation_query_embed, k, value=v)[0]
        tgt_residual = self.relation_pos_linear[layer](tgt)
        relation_query_embed = relation_query_embed + self.relation_pos_dropout[layer](tgt_residual)
        return relation_query_embed

    def forward(self, tgt_sub, tgt_obj, tgt_rel, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                subject_pos: Optional[Tensor] = None,
                object_pos: Optional[Tensor] = None,
                relation_pos: Optional[Tensor] = None):
        output_subject = tgt_sub
        output_object = tgt_obj
        output_relation = tgt_rel

        intermediate_relation = []
        intermediate_subject = []
        intermediate_object = []
        for layer_id in range(self.num_layers):
            sub_features = self.subject_layers[layer_id](output_subject, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=subject_pos)

            conditional_object_pos = self.object_query_generator(object_pos, sub_features, subject_pos, layer_id)
            obj_features = self.object_layers[layer_id](output_object, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=conditional_object_pos)

            conditional_relation_pos = self.relation_query_generator(relation_pos, sub_features, obj_features, subject_pos, conditional_object_pos, layer_id)
            rel_features = self.relation_layers[layer_id](output_relation, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=conditional_relation_pos)

            if self.return_intermediate:
                intermediate_subject.append(self.subject_norm(sub_features))
                intermediate_object.append(self.subject_norm(obj_features))
                intermediate_relation.append(self.relation_norm(rel_features))
            # import pdb;pdb.set_trace()
            rel_q = rel_features.shape[0]
            obj_q = obj_features.shape[0]
            multiply_q = rel_q//obj_q
            if layer_id != self.num_layers - 1:
                # Get queries for each decoder
                triplet_features = torch.cat([sub_features.repeat_interleave(multiply_q,0), \
                    obj_features.repeat_interleave(multiply_q,0), rel_features], -1)
                triplet_pos = torch.cat([subject_pos.repeat_interleave(multiply_q,0),\
                     conditional_object_pos.repeat_interleave(multiply_q,0),\
                     conditional_relation_pos], -1)
                triplet_features_pos = self.with_pos_embed(triplet_features, triplet_pos)

                subject_with_pos = self.with_pos_embed(sub_features, subject_pos)
                object_with_pos = self.with_pos_embed(obj_features, conditional_object_pos)
                relation_with_pos = self.with_pos_embed(rel_features, conditional_relation_pos)

                # Relation queries
                subject_graph_residual = self.subject_graph_query_residual[layer_id](self.subject_graph_query_attn[layer_id](subject_with_pos, triplet_features_pos, value=triplet_features)[0])
                output_subject = sub_features + self.subject_graph_query_norm[layer_id](self.subject_graph_query_dropout[layer_id](subject_graph_residual))

                object_graph_residual = self.object_graph_query_residual[layer_id](self.object_graph_query_attn[layer_id](object_with_pos, triplet_features_pos, value=triplet_features)[0])
                output_object = obj_features + self.object_graph_query_norm[layer_id](self.object_graph_query_dropout[layer_id](object_graph_residual))

                relation_graph_residual = self.relation_graph_query_residual[layer_id](self.relation_graph_query_attn[layer_id](relation_with_pos, triplet_features_pos, value=triplet_features)[0])
                output_relation = rel_features + self.relation_graph_query_norm[layer_id](self.relation_graph_query_dropout[layer_id](relation_graph_residual))


        if self.subject_norm is not None:
            sub_features = self.subject_norm(sub_features)
            obj_features = self.subject_norm(obj_features)
            rel_features = self.relation_norm(rel_features)
            if self.return_intermediate:
                intermediate_subject.pop()
                intermediate_subject.append(sub_features)
                intermediate_object.pop()
                intermediate_object.append(obj_features)
                intermediate_relation.pop()
                intermediate_relation.append(rel_features)

        if self.return_intermediate:
            return torch.stack(intermediate_subject), torch.stack(intermediate_object), torch.stack(intermediate_relation)

        return sub_features.unsqueeze(0), obj_features.unsqueeze(0), rel_features.unsqueeze(0)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(name, d_model, dropout, nhead, dim_feedforward, num_encoder_layers, num_decoder_layers, normalize_before, return_intermediate_dec, **kwargs):
    return TRANSFORMER_REGISTRY.get(name)(
        d_model=d_model,
        dropout=dropout,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        normalize_before=normalize_before,
        return_intermediate_dec=return_intermediate_dec,
        **kwargs
    )

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")