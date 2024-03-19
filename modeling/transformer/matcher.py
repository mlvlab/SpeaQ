import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F
import numpy as np

from .util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou

from detectron2.utils.registry import Registry
from torchvision.ops.boxes import box_area

sorted_dict = {'on': 712409, 'has': 277936, 'in': 251756, 'of': 146339, 'wearing': 136099, 'near': 96589, 'with': 66425, 'above': 47341, 'holding': 42722, 'behind': 41356, 'under': 22596, 'sitting on': 18643, 'wears': 15457, 'standing on': 14185, 'in front of': 13715, 'attached to': 10190, 'at': 9903, 'hanging from': 9894, 'over': 9317, 'for': 9145, 'riding': 8856, 'carrying': 5213, 'eating': 4688, 'walking on': 4613, 'playing': 3810, 'covering': 3806, 'laying on': 3739, 'along': 3624, 'watching': 3490, 'and': 3477, 'between': 3411, 'belonging to': 3288, 'painted on': 3095, 'against': 3092, 'looking at': 3083, 'from': 2945, 'parked on': 2721, 'to': 2517, 'made of': 2380, 'covered in': 2312, 'mounted on': 2253, 'says': 2241, 'part of': 2065, 'across': 1996, 'flying in': 1973, 'using': 1925, 'on back of': 1914, 'lying on': 1869, 'growing on': 1853, 'walking in': 1740}
sorted_idxs_with_cnt = {30: 712409, 19: 277936, 21: 251756, 29: 146339, 47: 136099, 28: 96589, 49: 66425, 0: 47341, 20: 42722, 7: 41356, 42: 22596, 39: 18643, 48: 15457, 40: 14185, 22: 13715, 6: 10190, 5: 9903, 18: 9894, 32: 9317, 15: 9145, 37: 8856, 10: 5213, 13: 4688, 45: 4613, 36: 3810, 12: 3806, 23: 3739, 3: 3624, 46: 3490, 4: 3477, 9: 3411, 8: 3288, 33: 3095, 2: 3092, 24: 3083, 16: 2945, 34: 2721, 41: 2517, 26: 2380, 11: 2312, 27: 2253, 38: 2241, 35: 2065, 1: 1996, 14: 1973, 43: 1925, 31: 1914, 25: 1869, 17: 1853, 44: 1740}
alphabet_list = [    'above',    'across',    'against',    'along',    'and',    'at',    'attached to',    'behind',    'belonging to',    'between',    'carrying',    'covered in',    'covering',    'eating',    'flying in',    'for',    'from',    'growing on',    'hanging from',    'has',    'holding',    'in',    'in front of',    'laying on',    'looking at',    'lying on',    'made of',    'mounted on',    'near',    'of',    'on',    'on back of',    'over',    'painted on',    'parked on',    'part of',    'playing',    'riding',    'says',    'sitting on',    'standing on',    'to',    'under',    'using',    'walking in',    'walking on',    'watching',    'wearing',    'wears',    'with']
alphabet_to_frequecy = [7, 43, 33, 27, 29, 16, 15, 9, 31, 30, 21, 39, 25, 22, 44, 19, 35, 48, 17, 1, 8, 2, 14, 26, 34, 47, 38, 40, 5, 3, 0, 46, 18, 32, 36, 42, 24, 20, 41, 11, 13, 37, 10, 45, 49, 23, 28, 4, 12, 6]



import copy

MATCHER_REGISTRY = Registry("MATCHER_REGISTRY")

@MATCHER_REGISTRY.register()
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets, return_cost_matrix=False):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()        

        sizes = [len(v["boxes"]) for v in targets]
        C_split = C.split(sizes, -1)
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C_split)]
        if not return_cost_matrix:
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        else:
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices], C_split

@MATCHER_REGISTRY.register()
class IterativeHungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, **kwargs):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.aux_loss = kwargs['cfg'].MODEL.DETR.DEEP_SUPERVISION
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets, return_cost_matrix=False, mask=None):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()        
        if mask is not None:
            C[:, ~mask] = np.float("inf")

        sizes = [len(v["boxes"]) for v in targets]
        C_split = C.split(sizes, -1)
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C_split)]
        
        if not return_cost_matrix:
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        else:
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices], C_split

    @torch.no_grad()
    def forward_relation(self, outputs, targets, return_cost_matrix=False):
        bs, num_queries = outputs["relation_logits"].shape[:2]
        out_prob = outputs["relation_logits"].flatten(0, 1).softmax(-1)
        out_sub_prob = outputs["relation_subject_logits"].flatten(0, 1).softmax(-1)
        out_obj_prob = outputs["relation_object_logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["relation_boxes"].flatten(0, 1)
        out_sub_bbox = outputs["relation_subject_boxes"].flatten(0, 1)
        out_obj_bbox = outputs["relation_object_boxes"].flatten(0, 1)

        if self.aux_loss:
            aux_out_prob = [output['pred_logits'].flatten(0, 1).softmax(-1) for output in outputs['aux_outputs_r']]
            aux_out_sub_prob = [output['pred_logits'].flatten(0, 1).softmax(-1) for output in outputs['aux_outputs_r_sub']]
            aux_out_obj_prob = [output['pred_logits'].flatten(0, 1).softmax(-1) for output in outputs['aux_outputs_r_obj']]
            aux_out_bbox = [output['pred_boxes'].flatten(0, 1) for output in outputs['aux_outputs_r']]
            aux_out_sub_bbox = [output['pred_boxes'].flatten(0, 1) for output in outputs['aux_outputs_r_sub']]
            aux_out_obj_bbox = [output['pred_boxes'].flatten(0, 1) for output in outputs['aux_outputs_r_obj']]

        device = out_prob.device

        gt_labels = [v['combined_labels'] for v in targets]
        gt_boxes = [v['combined_boxes'] for v in targets]
        relations = [v["image_relations"] for v in targets]
        relation_boxes = [v['relation_boxes'] for v in targets]
        
        if len(relations) > 0:
            tgt_ids = torch.cat(relations)[:, 2]
            tgt_sub_labels = torch.cat([gt_label[relation[:, 0]] for gt_label, relation in zip(gt_labels, relations)])
            tgt_obj_labels = torch.cat([gt_label[relation[:, 1]] for gt_label, relation in zip(gt_labels, relations)])
            tgt_boxes = torch.cat(relation_boxes)
            tgt_sub_boxes = torch.cat([gt_box[relation[:, 0]] for gt_box, relation in zip(gt_boxes, relations)])
            tgt_obj_boxes = torch.cat([gt_box[relation[:, 1]] for gt_box, relation in zip(gt_boxes, relations)])
        else:
            tgt_ids = torch.tensor([]).long().to(device)
            tgt_sub_labels = torch.tensor([]).long().to(device)
            tgt_obj_labels = torch.tensor([]).long().to(device)
            tgt_boxes = torch.zeros((0,4)).to(device)
            tgt_sub_boxes = torch.zeros((0,4)).to(device)
            tgt_obj_boxes = torch.zeros((0,4)).to(device)

        cost_class = -out_prob[:, tgt_ids]
        cost_subject_class = -out_sub_prob[:, tgt_sub_labels]
        cost_object_class = -out_obj_prob[:, tgt_obj_labels]
        
        cost_bbox = torch.cdist(out_bbox, tgt_boxes, p=1)
        cost_subject_bbox = torch.cdist(out_sub_bbox, tgt_sub_boxes, p=1)
        cost_object_bbox = torch.cdist(out_obj_bbox, tgt_obj_boxes, p=1)
        
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_boxes))
        cost_subject_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_sub_bbox), box_cxcywh_to_xyxy(tgt_sub_boxes))
        cost_object_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_obj_bbox), box_cxcywh_to_xyxy(tgt_obj_boxes))

        C = self.cost_bbox * (cost_bbox + cost_subject_bbox + cost_object_bbox) + self.cost_class * (cost_class + cost_subject_class + cost_object_class) + self.cost_giou * (cost_giou + cost_subject_giou + cost_object_giou)
        
        # Add aux loss cost
        if self.aux_loss:
            for aux_idx in range(len(aux_out_prob)):
                aux_cost_class = -aux_out_prob[aux_idx][:, tgt_ids]
                aux_cost_subject_class = -aux_out_sub_prob[aux_idx][:, tgt_sub_labels]
                aux_cost_object_class = -aux_out_obj_prob[aux_idx][:, tgt_obj_labels]

                aux_cost_bbox = torch.cdist(aux_out_bbox[aux_idx], tgt_boxes, p=1)
                aux_cost_subject_bbox = torch.cdist(aux_out_sub_bbox[aux_idx], tgt_sub_boxes, p=1)
                aux_cost_object_bbox = torch.cdist(aux_out_obj_bbox[aux_idx], tgt_obj_boxes, p=1)

                aux_cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(aux_out_bbox[aux_idx]), box_cxcywh_to_xyxy(tgt_boxes))
                aux_cost_subject_giou = -generalized_box_iou(box_cxcywh_to_xyxy(aux_out_sub_bbox[aux_idx]), box_cxcywh_to_xyxy(tgt_sub_boxes))
                aux_cost_object_giou = -generalized_box_iou(box_cxcywh_to_xyxy(aux_out_obj_bbox[aux_idx]), box_cxcywh_to_xyxy(tgt_obj_boxes))
                aux_C = self.cost_bbox * (aux_cost_bbox + aux_cost_subject_bbox + aux_cost_object_bbox) + self.cost_class * (aux_cost_class + aux_cost_subject_class + aux_cost_object_class) + self.cost_giou * (aux_cost_giou + aux_cost_subject_giou + aux_cost_object_giou)

                C = C + aux_C
            
        C = C.view(bs, num_queries, -1).cpu()   
        # C : cost matrix with shape of (batch_size, n_queries, n_gts)
        
        sizes = [len(v["image_relations"]) for v in targets]
        C_split = C.split(sizes, -1)
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C_split)]
        indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        # linear_sum_assignment returns matching result as size-2 tuple of row-indices, col-indices

        # Remaining GT objects matching
        pred_masks = {'subject': [], 'object': []}
        target_masks = {'subject' :[], 'object': []}
        combined_indices = {'subject' :[], 'object': [], 'relation': []}
        for image_idx, target in enumerate(targets):
            all_objects = torch.arange(len(gt_labels[image_idx])).to(device)
            relation = target['image_relations']
            curr_relation_idx = indices[image_idx]
            curr_pred_mask = torch.ones(num_queries, device=device)
            curr_pred_mask[curr_relation_idx[0]] = 0
            curr_pred_mask = (curr_pred_mask == 1)
            
            combined_indices['relation'].append((curr_relation_idx[0], curr_relation_idx[1]))
            for branch_idx, branch_type in enumerate(['subject', 'object']):  
                combined_indices[branch_type].append((curr_relation_idx[0], relation[:, branch_idx][curr_relation_idx[1]].cpu()))
        return combined_indices

        
@MATCHER_REGISTRY.register()
class SpeaQHungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, cfg=None, **kwargs):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.aux_loss = cfg.MODEL.DETR.DEEP_SUPERVISION
        self.match_independent = cfg.MODEL.DETR.MATCH_INDEPENDENT
        # mapping order based on train set
        self.relation_order = torch.tensor([30, 19, 21, 29, 47, 28, 49, 0, 20, 7, 42, 39, 48, 40, 22, 6, 5, 18, 32, 15, 37, 10, 13, 45, 36, 12, 23, 3, 46, 4, 9, 8, 33, 2, 24, 16, 34, 41, 26, 11, 27, 38, 35, 1, 14, 43, 31, 25, 17, 44])
        
        #one to many
        self.o2m_scheme = cfg.MODEL.DETR.ONE2MANY_SCHEME
        if self.o2m_scheme == 'dynamic':            
            self.o2m_dynamic_scheme = cfg.MODEL.DETR.ONE2MANY_DYNAMIC_SCHEME
        self.o2m_k = cfg.MODEL.DETR.ONE2MANY_K

        self.query_multiple = cfg.MODEL.DETR.MULTIPLY_QUERY
        self.use_group_mask = cfg.MODEL.DETR.USE_GROUP_MASK
        self.only_predicate_multiply = cfg.MODEL.DETR.ONLY_PREDICATE_MULTIPLY
        self.sorted_dict = sorted_dict
        self.sorted_idxs_with_cnt = sorted_idxs_with_cnt
        self.relation_freq = torch.tensor(list(sorted_dict.values()))
        self.query_multiply = cfg.MODEL.DETR.MULTIPLY_QUERY
        self.num_rel_queries = cfg.MODEL.DETR.NUM_RELATION_QUERIES # num relation quer
        self.num_mul_so_queries = cfg.MODEL.DETR.NUM_OBJECT_QUERIES if cfg.MODEL.DETR.ONLY_PREDICATE_MULTIPLY else cfg.MODEL.DETR.NUM_OBJECT_QUERIES *cfg.MODEL.DETR.MULTIPLY_QUERY
        self.num_groups = cfg.MODEL.DETR.NUM_GROUPS
        self.fg_rel_count = kwargs['statistics']['fg_rel_count']

        self.o2m_predicate_score = cfg.MODEL.DETR.ONE2MANY_PREDICATE_SCORE
        self.o2m_predicate_weight = cfg.MODEL.DETR.ONE2MANY_PREDICATE_WEIGHT


        self.size_of_groups = self.get_group_list_by_n_groups(self.num_groups)

        self.grouping()
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def fill_list(self, num, n):
        quotient, remainder = divmod(num, n)
        lst = [quotient] * n
        for i in range(remainder):
            lst[-1 * (i + 1)] += 1
        return torch.tensor(lst)

    def grouping(self):
        device_group = 'cuda'
        group_tensor = -torch.ones(50, device=device_group)
        sum_of_each_groups = torch.as_tensor(
            [x.sum().item() for x in torch.split(self.relation_freq, self.size_of_groups)], device=device_group)
        n_queries_per_group = (sum_of_each_groups * self.num_mul_so_queries / sum_of_each_groups.sum()).int()
        n_queries_per_group += self.fill_list((self.num_mul_so_queries - n_queries_per_group.sum()).item(), len(n_queries_per_group)).to(device=device_group)
        self.n_queries_per_group = n_queries_per_group.long()
        assert self.num_mul_so_queries == n_queries_per_group.sum()
        self.rel_order = torch.split(self.relation_order, self.size_of_groups)
        for g, row in enumerate(self.rel_order):
            group_tensor[row] = g
        self.group_tensor = group_tensor
        self.freq_list = n_queries_per_group.cpu().numpy()
        self.n_groups = len(self.freq_list)


    def get_group_list_by_n_groups(self, n_groups):
        total_list = list()
        last_checked_index = 0
        current_idx = 0
        size_of_whole_groups = 0
        for i in range(n_groups - 1):
            sum_of_this_group = 0
            size_of_this_group = 0
            remaining_list = self.relation_freq.numpy()[last_checked_index:]
            remaining_half_cnt = remaining_list.sum() // 2
            while sum_of_this_group + self.relation_freq.numpy()[current_idx] < remaining_half_cnt:
                sum_of_this_group += self.relation_freq.numpy()[current_idx]
                size_of_this_group += 1
                size_of_whole_groups += 1
                current_idx += 1

            total_list.append(size_of_this_group)
            last_checked_index = current_idx
        total_list.append(50 - size_of_whole_groups)
        return total_list

    @torch.no_grad()
    def forward(self, outputs, targets, return_cost_matrix=False, mask=None):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()        
        if mask is not None:
            C[:, ~mask] = np.float("inf")

        sizes = [len(v["boxes"]) for v in targets]
        C_split = C.split(sizes, -1)
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C_split)]
        
        if not return_cost_matrix:
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        else:
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices], C_split

    @torch.no_grad()
    def forward_relation(self, outputs, targets, layer_num = -1, return_cost_matrix=False):
        copy_targets = copy.deepcopy(targets)
        bs, num_queries = outputs["relation_logits"].shape[:2]
        if self.o2m_scheme == 'static':
            for t in copy_targets:
                t['image_relations'] = t['image_relations'].repeat_interleave(self.o2m_k, dim=0)
                t['relation_boxes'] = t['relation_boxes'].repeat_interleave(self.o2m_k, dim=0)
                t['relation_labels'] = t['relation_labels'].repeat_interleave(self.o2m_k, dim=0)
            k_mean_log = torch.tensor(self.o2m_k,device=outputs['relation_logits'].device).float()
        elif self.o2m_scheme == 'dynamic' and self.match_independent:
            if self.use_group_mask:
                relations = [v["image_relations"] for v in copy_targets]
                if len(relations) > 0:
                    tgt_ids = torch.cat(relations)[:, 2] #relation cls
                else:
                    assert False, "No relation"
                group_mask = 1-F.one_hot(self.group_tensor[tgt_ids].long(), num_classes=len(self.n_queries_per_group)).t() #(5, #tgt_ids)
                freq_list = [fl * self.query_multiple for fl in self.n_queries_per_group]
                for idx, freq in enumerate(freq_list):
                    temp = group_mask[idx].reshape(1,1,-1).repeat(bs, freq, 1)*1e+6
                    if idx == 0:
                        new_group_mask = temp
                    else:
                        new_group_mask = torch.cat((new_group_mask, temp), 1)

                group_mask = new_group_mask.reshape(bs*num_queries, -1).to(outputs["relation_logits"].device)

            anno_len = [len(t['image_relations']) for t in copy_targets]
            out_sub_bbox = outputs['relation_subject_boxes'].flatten(0, 1)
            out_obj_bbox = outputs['relation_object_boxes'].flatten(0, 1)

            relations = [v["image_relations"] for v in copy_targets]
            gt_boxes = [v['combined_boxes'] for v in copy_targets]
            if len(relations) > 0:
                tgt_ids = torch.cat(relations)[:, 2]  # relation cls
            else:
                assert False, "No relation"
            if len(relations) > 0:
                tgt_sub_boxes = torch.cat([gt_box[relation[:, 0]] for gt_box, relation in zip(gt_boxes, relations)])
                tgt_obj_boxes = torch.cat([gt_box[relation[:, 1]] for gt_box, relation in zip(gt_boxes, relations)])
            else:
                assert False, "No relation"

            sub_iou = box_iou(box_cxcywh_to_xyxy(out_sub_bbox), box_cxcywh_to_xyxy(tgt_sub_boxes))[0].view(bs, num_queries, -1)
            obj_iou = box_iou(box_cxcywh_to_xyxy(out_obj_bbox), box_cxcywh_to_xyxy(tgt_obj_boxes))[0].view(bs, num_queries, -1)
            if self.use_group_mask:
                mask_iou = (group_mask.view(bs,num_queries,-1)==0).float()
                sub_iou *=mask_iou
                obj_iou *=mask_iou

            if self.o2m_dynamic_scheme =='min':
                cum_iou = torch.min(sub_iou,obj_iou)
            elif self.o2m_dynamic_scheme =='max':
                cum_iou = torch.max(sub_iou,obj_iou)
            elif self.o2m_dynamic_scheme =='gm':
                cum_iou = (sub_iou*obj_iou)**0.5
            elif self.o2m_dynamic_scheme =='am':
                cum_iou = 0.5*(sub_iou+obj_iou)
            else:
                NotImplementedError

            if self.o2m_predicate_score:
                rel_out_prob = outputs['relation_logits'].softmax(dim=-1)
                cum_iou += self.o2m_predicate_weight * rel_out_prob[..., tgt_ids]

            topk_iou = torch.topk(cum_iou, self.o2m_k, dim=1)[0]

            batched_k = [(ci[i].sum(dim=0)).int().clamp_(min=1) for i,ci in enumerate(topk_iou.split(anno_len,dim=-1))]

            for t,k_per_im in zip(copy_targets, batched_k):
                t["image_relations"] = t["image_relations"].repeat_interleave(k_per_im, dim=0)
                t["relation_boxes"] = t["relation_boxes"].repeat_interleave(k_per_im, dim=0)
                t["relation_labels"] = t["relation_labels"].repeat_interleave(k_per_im, dim=0)
            k_mean_log = torch.nanmean(torch.cat(batched_k).float(),dim=0).nan_to_num(0)
        else:
            NotImplementedError

        bs, num_queries = outputs["relation_logits"].shape[:2]
        out_prob = outputs["relation_logits"].flatten(0, 1).softmax(-1)
        out_sub_prob = outputs["relation_subject_logits"].flatten(0, 1).softmax(-1)
        out_obj_prob = outputs["relation_object_logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["relation_boxes"].flatten(0, 1)
        out_sub_bbox =  outputs["relation_subject_boxes"].flatten(0, 1)
        out_obj_bbox = outputs["relation_object_boxes"].flatten(0, 1)

        if self.aux_loss and not self.match_independent:
            aux_out_prob = [ output['pred_logits'].flatten(0, 1).softmax(-1) for output in outputs['aux_outputs_r']]
            aux_out_sub_prob = [ output['pred_logits'].flatten(0, 1).softmax(-1) for output in outputs['aux_outputs_r_sub']]
            aux_out_obj_prob = [ output['pred_logits'].flatten(0, 1).softmax(-1) for output in outputs['aux_outputs_r_obj']]
            aux_out_bbox = [ output['pred_boxes'].flatten(0, 1) for output in outputs['aux_outputs_r']]
            aux_out_sub_bbox = [ output['pred_boxes'].flatten(0, 1) for output in outputs['aux_outputs_r_sub']]
            aux_out_obj_bbox = [ output['pred_boxes'].flatten(0, 1) for output in outputs['aux_outputs_r_obj']]

        device = out_prob.device

        gt_labels = [v['combined_labels'] for v in copy_targets]
        gt_boxes = [v['combined_boxes'] for v in copy_targets]
        relations = [v["image_relations"] for v in copy_targets]
        relation_boxes = [v['relation_boxes'] for v in copy_targets]
        
        if len(relations) > 0:
            tgt_ids = torch.cat(relations)[:, 2] #relation cls
            tgt_sub_labels = torch.cat([gt_label[relation[:, 0]] for gt_label, relation in zip(gt_labels, relations)])
            tgt_obj_labels = torch.cat([gt_label[relation[:, 1]] for gt_label, relation in zip(gt_labels, relations)])
            tgt_boxes = torch.cat(relation_boxes)
            tgt_sub_boxes = torch.cat([gt_box[relation[:, 0]] for gt_box, relation in zip(gt_boxes, relations)])
            tgt_obj_boxes = torch.cat([gt_box[relation[:, 1]] for gt_box, relation in zip(gt_boxes, relations)])
        else:
            tgt_ids = torch.tensor([]).long().to(device)
            tgt_sub_labels = torch.tensor([]).long().to(device)
            tgt_obj_labels = torch.tensor([]).long().to(device)
            tgt_boxes = torch.zeros((0,4)).to(device)
            tgt_sub_boxes = torch.zeros((0,4)).to(device)
            tgt_obj_boxes = torch.zeros((0,4)).to(device)

        if self.use_group_mask:
            relations = [v["image_relations"] for v in copy_targets]
            if len(relations) > 0:
                tgt_ids = torch.cat(relations)[:, 2] #relation cls
            else:
                tgt_ids = torch.tensor([]).long().to(device)

            group_mask = 1-F.one_hot(self.group_tensor[tgt_ids].long(), num_classes=len(self.n_queries_per_group)).t().to(outputs['relation_logits'].device)
            freq_list = [fl * self.query_multiple for fl in self.n_queries_per_group]
            for idx, freq in enumerate(freq_list):
                temp = group_mask[idx].reshape(1,1,-1).repeat(bs, freq, 1)*1e+6
                if idx == 0:
                    new_group_mask = temp
                else:
                    new_group_mask = torch.cat((new_group_mask, temp), 1)

            group_mask = new_group_mask.reshape(bs*num_queries, -1)

        cost_class = -out_prob[:, tgt_ids]
        cost_subject_class = -out_sub_prob[:, tgt_sub_labels]
        
        cost_object_class = -out_obj_prob[:, tgt_obj_labels]
        
        
        cost_bbox = torch.cdist(out_bbox, tgt_boxes, p=1)
        cost_subject_bbox = torch.cdist(out_sub_bbox, tgt_sub_boxes, p=1)
        
        cost_object_bbox = torch.cdist(out_obj_bbox, tgt_obj_boxes, p=1)
        
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_boxes))
        cost_subject_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_sub_bbox), box_cxcywh_to_xyxy(tgt_sub_boxes))
        
        cost_object_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_obj_bbox), box_cxcywh_to_xyxy(tgt_obj_boxes))


        C = self.cost_bbox * (cost_bbox + cost_subject_bbox + cost_object_bbox) + self.cost_class * (cost_class + cost_subject_class + cost_object_class) + self.cost_giou * (cost_giou + cost_subject_giou + cost_object_giou)
        # Add aux loss cost
        if self.aux_loss and not self.match_independent:
            for aux_idx in range(len(aux_out_prob)):
                aux_cost_class = -aux_out_prob[aux_idx][:, tgt_ids]
                aux_cost_subject_class = -aux_out_sub_prob[aux_idx][:, tgt_sub_labels]
                aux_cost_object_class = -aux_out_obj_prob[aux_idx][:, tgt_obj_labels]

                aux_cost_bbox = torch.cdist(aux_out_bbox[aux_idx], tgt_boxes, p=1)
                aux_cost_subject_bbox = torch.cdist(aux_out_sub_bbox[aux_idx], tgt_sub_boxes, p=1)
                aux_cost_object_bbox = torch.cdist(aux_out_obj_bbox[aux_idx], tgt_obj_boxes, p=1)

                aux_cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(aux_out_bbox[aux_idx]), box_cxcywh_to_xyxy(tgt_boxes))
                aux_cost_subject_giou = -generalized_box_iou(box_cxcywh_to_xyxy(aux_out_sub_bbox[aux_idx]), box_cxcywh_to_xyxy(tgt_sub_boxes))
                aux_cost_object_giou = -generalized_box_iou(box_cxcywh_to_xyxy(aux_out_obj_bbox[aux_idx]), box_cxcywh_to_xyxy(tgt_obj_boxes))

                
                aux_C = self.cost_bbox * (aux_cost_bbox + aux_cost_subject_bbox + aux_cost_object_bbox) + self.cost_class * (aux_cost_class + aux_cost_subject_class + aux_cost_object_class) + self.cost_giou * (aux_cost_giou + aux_cost_subject_giou + aux_cost_object_giou)
                C = C + aux_C

        if self.use_group_mask:
            C = C + group_mask
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["image_relations"]) for v in copy_targets]
        C_split = C.split(sizes, -1)
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C_split)]
        indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

        # Remaining GT objects matching
        pred_masks = {'subject': [], 'object': []}
        target_masks = {'subject' :[], 'object': []}
        combined_indices = {'subject' :[], 'object': [], 'relation': []}
        for image_idx, target in enumerate(copy_targets):
            all_objects = torch.arange(len(gt_labels[image_idx])).to(device)
            relation = target['image_relations']
            curr_relation_idx = indices[image_idx]
            curr_pred_mask = torch.ones(num_queries, device=device)
            curr_pred_mask[curr_relation_idx[0]] = 0
            curr_pred_mask = (curr_pred_mask == 1)
            
            combined_indices['relation'].append((curr_relation_idx[0], curr_relation_idx[1]))
            for branch_idx, branch_type in enumerate(['subject', 'object']):  
                combined_indices[branch_type].append((curr_relation_idx[0], relation[:, branch_idx][curr_relation_idx[1]].cpu()))
        return combined_indices, k_mean_log, copy_targets


def build_matcher(name, cost_class, cost_bbox, cost_giou, topk=1, cfg=None, **kwargs):
    if topk == 1:
        return MATCHER_REGISTRY.get(name)(cost_class=cost_class, cost_bbox=cost_bbox, cost_giou=cost_giou, cfg=cfg, **kwargs)
    else:
        return MATCHER_REGISTRY.get(name)(cost_class=cost_class, cost_bbox=cost_bbox, cost_giou=cost_giou, topk=topk, cfg=cfg, **kwargs)
