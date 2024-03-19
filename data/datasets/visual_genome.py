import h5py
import json
import math
from math import floor
from PIL import Image, ImageDraw
import random
import os
import torch
import numpy as np
import pickle
import yaml
from detectron2.config import get_cfg
from detectron2.structures import Instances, Boxes, pairwise_iou, BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
import logging
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import copy

class VisualGenomeTrainData:
    """
    Register data for Visual Genome training
    """
    def __init__(self, cfg, split='train'):
        self.cfg = cfg
        self.split = split
        if split == 'train':
            self.mask_location = cfg.DATASETS.VISUAL_GENOME.TRAIN_MASKS
        elif split == 'val':
            self.mask_location = cfg.DATASETS.VISUAL_GENOME.VAL_MASKS
        else:
            self.mask_location = cfg.DATASETS.VISUAL_GENOME.TEST_MASKS
        self.mask_exists = os.path.isfile(self.mask_location)
        self.clamped = True if "clamped" in self.mask_location else ""
        self.per_class_dataset = cfg.DATASETS.VISUAL_GENOME.PER_CLASS_DATASET if split == 'train' else False
        self.bgnn = cfg.DATASETS.VISUAL_GENOME.BGNN if split == 'train' else False
        self.clipped = cfg.DATASETS.VISUAL_GENOME.CLIPPED
        self.precompute = False if (self.cfg.DATASETS.VISUAL_GENOME.FILTER_EMPTY_RELATIONS or self.cfg.DATASETS.VISUAL_GENOME.FILTER_NON_OVERLAP) else True
        try:
            with open('./data/datasets/images_to_remove.txt', 'r') as f:
                ids = f.readlines()
            self.ids_to_remove = {int(x.strip()) : 1 for x in ids[0].replace('[', '').replace(']','').split(",")}
        except:
            self.ids_to_remove = []
        # self._process_data()
        self.dataset_dicts = self._fetch_data_dict()
        self.register_dataset()
        try:
            statistics = self.get_statistics()
        except:
            pass
        if self.bgnn:
            freq = statistics['fg_rel_count'] / statistics['fg_rel_count'].sum() 
            freq = freq.numpy()
            oversample_param = cfg.DATASETS.VISUAL_GENOME.OVERSAMPLE_PARAM
            undersample_param = cfg.DATASETS.VISUAL_GENOME.UNDERSAMPLE_PARAM
            oversampling_ratio = np.maximum(np.sqrt((oversample_param / (freq + 1e-5))), np.ones_like(freq))[:-1]
            sampled_dataset_dicts = []
            sampled_num = []
            unique_relation_ratios = []
            unique_relations_dict = []
            for record in self.dataset_dicts:
                relations = record['relations']
                if len(relations) > 0:
                    unique_relations = np.unique(relations[:,2])
                    repeat_num = int(np.ceil(np.max(oversampling_ratio[unique_relations])))
                    for rep_idx in range(repeat_num):
                        sampled_num.append(repeat_num)
                        unique_relation_ratios.append(oversampling_ratio[unique_relations])
                        sampled_dataset_dicts.append(record)
                        unique_relations_dict.append({rel:idx for idx, rel in enumerate(unique_relations)})
                else:
                    sampled_dataset_dicts.append(record)
                    sampled_num.append(1)
                    unique_relation_ratios.append([])
                    unique_relations_dict.append({})

            self.dataset_dicts = sampled_dataset_dicts
            self.dataloader = BGNNSampler(self.dataset_dicts, sampled_num, oversampling_ratio, undersample_param, unique_relation_ratios, unique_relations_dict)
            DatasetCatalog.remove('VG_{}'.format(self.split))
            self.register_dataset(dataloader=True)
            MetadataCatalog.get('VG_{}'.format(self.split)).set(statistics=statistics) 
            print (self.idx_to_predicates, statistics['fg_rel_count'].numpy().tolist())

        if self.per_class_dataset:
            freq = statistics['fg_rel_count'] / statistics['fg_rel_count'].sum() 
            freq = freq.numpy()
            oversample_param = cfg.DATASETS.VISUAL_GENOME.OVERSAMPLE_PARAM
            undersample_param = cfg.DATASETS.VISUAL_GENOME.UNDERSAMPLE_PARAM
            oversampling_ratio = np.maximum(np.sqrt((oversample_param / (freq + 1e-5))), np.ones_like(freq))[:-1]
            unique_relation_ratios = defaultdict(list)
            unique_relations_dict = defaultdict(list)     
            per_class_dataset = defaultdict(list)
            sampled_num = defaultdict(list)
            for record in self.dataset_dicts:
                relations = record['relations']
                if len(relations) > 0:
                    unique_relations = np.unique(relations[:,2])
                    repeat_num = int(np.ceil(np.max(oversampling_ratio[unique_relations])))
                    for rel in unique_relations:
                        per_class_dataset[rel].append(record)   
                        sampled_num[rel].append(repeat_num)
                        unique_relation_ratios[rel].append(oversampling_ratio[unique_relations]) 
                        unique_relations_dict[rel].append({rel:idx for idx, rel in enumerate(unique_relations)})
            self.dataloader = ClassBalancedSampler(per_class_dataset, len(self.dataset_dicts), sampled_num, oversampling_ratio, undersample_param, unique_relation_ratios, unique_relations_dict)
            DatasetCatalog.remove('VG_{}'.format(self.split))
            self.register_dataset(dataloader=True)
            MetadataCatalog.get('VG_{}'.format(self.split)).set(statistics=statistics) 
            print (self.idx_to_predicates, statistics['fg_rel_count'].numpy().tolist())

    def register_dataset(self, dataloader=False):
        """
        Register datasets to use with Detectron2
        """
        if not dataloader:
            DatasetCatalog.register('VG_{}'.format(self.split), lambda: self.dataset_dicts)
        else:    
            DatasetCatalog.register('VG_{}'.format(self.split), lambda: self.dataloader)            
            
        #Get labels
        self.mapping_dictionary = json.load(open(self.cfg.DATASETS.VISUAL_GENOME.MAPPING_DICTIONARY, 'r'))
        self.idx_to_classes = sorted(self.mapping_dictionary['label_to_idx'], key=lambda k: self.mapping_dictionary['label_to_idx'][k])
        self.idx_to_predicates = sorted(self.mapping_dictionary['predicate_to_idx'], key=lambda k: self.mapping_dictionary['predicate_to_idx'][k])
        self.idx_to_attributes = sorted(self.mapping_dictionary['attribute_to_idx'], key=lambda k: self.mapping_dictionary['attribute_to_idx'][k])
        MetadataCatalog.get('VG_{}'.format(self.split)).set(thing_classes=self.idx_to_classes, predicate_classes=self.idx_to_predicates, attribute_classes=self.idx_to_attributes)
    
    def _fetch_data_dict(self):
        """
        Load data in detectron format
        """
        fileName = "tmp/visual_genome_{}_data_{}{}{}{}{}{}{}{}.pkl".format(self.split, 'masks' if self.mask_exists else '', '_oi' if 'oi' in self.mask_location else '', "_clamped" if self.clamped else "", "_precomp" if self.precompute else "", "_clipped" if self.clipped else "", '_overlapfalse' if not self.cfg.DATASETS.VISUAL_GENOME.FILTER_NON_OVERLAP else "", '_emptyfalse' if not self.cfg.DATASETS.VISUAL_GENOME.FILTER_EMPTY_RELATIONS else '', "_perclass" if self.per_class_dataset else '')
        print("Loading file: ", fileName)
        if os.path.isfile(fileName):
            #If data has been processed earlier, load that to save time
            with open(fileName, 'rb') as inputFile:
                dataset_dicts = pickle.load(inputFile)
        else:
            #Process data
            os.makedirs('tmp', exist_ok=True)
            dataset_dicts = self._process_data()
            with open(fileName, 'wb') as inputFile:
                pickle.dump(dataset_dicts, inputFile)
        return dataset_dicts
            
    def _process_data(self):
        self.VG_attribute_h5 = h5py.File(self.cfg.DATASETS.VISUAL_GENOME.VG_ATTRIBUTE_H5, 'r')
        
        # Remove corrupted images
        image_data = json.load(open(self.cfg.DATASETS.VISUAL_GENOME.IMAGE_DATA, 'r'))
        self.corrupted_ims = ['1592', '1722', '4616', '4617']
        self.image_data = []
        for i, img in enumerate(image_data):
            if str(img['image_id']) in self.corrupted_ims:
                continue
            self.image_data.append(img)
        assert(len(self.image_data) == 108073)
        self.masks = None
        if self.mask_location != "":
            try:
                with open(self.mask_location, 'rb') as f:
                    self.masks = pickle.load(f)
            except:
                pass
        dataset_dicts = self._load_graphs()
        return dataset_dicts

    def get_statistics(self, eps=1e-3, bbox_overlap=True):
        num_object_classes = len(MetadataCatalog.get('VG_{}'.format(self.split)).thing_classes) + 1
        num_relation_classes = len(MetadataCatalog.get('VG_{}'.format(self.split)).predicate_classes) + 1
        
        fg_matrix = np.zeros((num_object_classes, num_object_classes, num_relation_classes), dtype=np.int64)
        bg_matrix = np.zeros((num_object_classes, num_object_classes), dtype=np.int64)
        fg_rel_count = np.zeros((num_relation_classes), dtype=np.int64)
        for idx, data in enumerate(self.dataset_dicts):
            gt_relations = data['relations']
            gt_classes = np.array([x['category_id'] for x in data['annotations']])
            gt_boxes = np.array([x['bbox'] for x in data['annotations']])
            for (o1, o2), rel in zip(gt_classes[gt_relations[:,:2]], gt_relations[:,2]):
                fg_matrix[o1, o2, rel] += 1
                fg_rel_count[rel] += 1

            for (o1, o2) in gt_classes[np.array(box_filter(gt_boxes, must_overlap=bbox_overlap), dtype=int)]:
                bg_matrix[o1, o2] += 1
        bg_matrix += 1
        fg_matrix[:, :, -1] = bg_matrix
        pred_dist = np.log(fg_matrix / fg_matrix.sum(2)[:, :, None] + eps)

        result = {
            'fg_matrix': torch.from_numpy(fg_matrix),
            'pred_dist': torch.from_numpy(pred_dist).float(),
            'fg_rel_count': torch.from_numpy(fg_rel_count).float(),
            'obj_classes': self.idx_to_classes + ['__background__'],
            'rel_classes': self.idx_to_predicates + ['__background__'],
            'att_classes': self.idx_to_attributes,
        }
        print (torch.from_numpy(fg_rel_count).float())
        MetadataCatalog.get('VG_{}'.format(self.split)).set(statistics=result)
        return result

    def _load_graphs(self):
        """
        Parse examples and create dictionaries
        """
        data_split = self.VG_attribute_h5['split'][:]
        split_flag = 2 if self.split == 'test' else 0
        split_mask = data_split == split_flag
        
        #Filter images without bounding boxes
        split_mask &= self.VG_attribute_h5['img_to_first_box'][:] >= 0
        if self.cfg.DATASETS.VISUAL_GENOME.FILTER_EMPTY_RELATIONS:
            split_mask &= self.VG_attribute_h5['img_to_first_rel'][:] >= 0
        image_index = np.where(split_mask)[0]
        
        if self.split == 'val':
            image_index = image_index[:self.cfg.DATASETS.VISUAL_GENOME.NUMBER_OF_VALIDATION_IMAGES]
        elif self.split == 'train':
            image_index = image_index[self.cfg.DATASETS.VISUAL_GENOME.NUMBER_OF_VALIDATION_IMAGES:]
        
        split_mask = np.zeros_like(data_split).astype(bool)
        split_mask[image_index] = True
        
        # Get box information
        all_labels = self.VG_attribute_h5['labels'][:, 0]
        all_attributes = self.VG_attribute_h5['attributes'][:, :]
        all_boxes = self.VG_attribute_h5['boxes_{}'.format(self.cfg.DATASETS.VISUAL_GENOME.BOX_SCALE)][:]  # cx,cy,w,h
        assert np.all(all_boxes[:, :2] >= 0)  # sanity check
        assert np.all(all_boxes[:, 2:] > 0)  # no empty box
        
        # Convert from xc, yc, w, h to x1, y1, x2, y2
        all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
        all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]
        
        first_box_index = self.VG_attribute_h5['img_to_first_box'][split_mask]
        last_box_index = self.VG_attribute_h5['img_to_last_box'][split_mask]
        first_relation_index = self.VG_attribute_h5['img_to_first_rel'][split_mask]
        last_relation_index = self.VG_attribute_h5['img_to_last_rel'][split_mask]

        #Load relation labels
        all_relations = self.VG_attribute_h5['relationships'][:]
        all_relation_predicates = self.VG_attribute_h5['predicates'][:, 0]
        
        image_indexer = np.arange(len(self.image_data))[split_mask]
        # Iterate over images
        dataset_dicts = []
        num_rels = []
        num_objs = []
        for idx, _ in enumerate(image_index):
            record = {}
            #Get image metadata
            image_data = self.image_data[image_indexer[idx]]
            record['file_name'] = os.path.join(self.cfg.DATASETS.VISUAL_GENOME.IMAGES, '{}.jpg'.format(image_data['image_id']))
            record['image_id'] = image_data['image_id']
            record['height'] = image_data['height']
            record['width'] = image_data['width']
            if self.clipped:
                if image_data['coco_id'] in self.ids_to_remove:
                    continue
            #Get annotations
            boxes = all_boxes[first_box_index[idx]:last_box_index[idx] + 1, :]
            gt_classes = all_labels[first_box_index[idx]:last_box_index[idx] + 1]
            gt_attributes = all_attributes[first_box_index[idx]:last_box_index[idx] + 1, :]

            if first_relation_index[idx] > -1:
                predicates = all_relation_predicates[first_relation_index[idx]:last_relation_index[idx] + 1]
                objects = all_relations[first_relation_index[idx]:last_relation_index[idx] + 1] - first_box_index[idx]
                predicates = predicates - 1
                relations = np.column_stack((objects, predicates))
            else:
                assert not self.cfg.DATASETS.VISUAL_GENOME.FILTER_EMPTY_RELATIONS
                relations = np.zeros((0, 3), dtype=np.int32)
            
            if self.cfg.DATASETS.VISUAL_GENOME.FILTER_NON_OVERLAP and self.split == 'train':
                # Remove boxes that don't overlap
                boxes_list = Boxes(boxes)
                ious = pairwise_iou(boxes_list, boxes_list)
                relation_boxes_ious = ious[relations[:,0], relations[:,1]]
                iou_indexes = np.where(relation_boxes_ious > 0.0)[0]
                if iou_indexes.size > 0:
                    relations = relations[iou_indexes]
                else:
                    #Ignore image
                    continue
            #Get masks if possible
            if self.masks is not None:
                try:
                    gt_masks = self.masks[image_data['image_id']]
                except:
                    print (image_data['image_id'])
            record['relations'] = relations
            objects = []
            # if len(boxes) != len(gt_masks):
            mask_idx = 0
            for obj_idx in range(len(boxes)):
                resized_box = boxes[obj_idx] / self.cfg.DATASETS.VISUAL_GENOME.BOX_SCALE * max(record['height'], record['width'])
                obj = {
                      "bbox": resized_box.tolist(),
                      "bbox_mode": BoxMode.XYXY_ABS,
                      "category_id": gt_classes[obj_idx] - 1,
                      "attribute": gt_attributes[obj_idx],           
                }
                if self.masks is not None:
                    if gt_masks['empty_index'][obj_idx]:
                        refined_poly = []
                        for poly_idx, poly in enumerate(gt_masks['polygons'][mask_idx]):
                            if len(poly) >= 6:
                                refined_poly.append(poly)
                        obj["segmentation"] = refined_poly
                        mask_idx += 1
                    else:
                        obj["segmentation"] = []
                    if len(obj["segmentation"]) > 0:
                        objects.append(obj)
                else:
                    objects.append(obj)
            num_objs.append(len(objects))
            num_rels.append(len(relations))  
            
            record['annotations'] = objects
            dataset_dicts.append(record)
        print ("Max Rels:", np.max(num_rels), "Max Objs:", np.max(num_objs))
        print ("Avg Rels:", np.mean(num_rels), "Avg Objs:", np.mean(num_objs))
        print ("Median Rels:", np.median(num_rels), "Median Objs:", np.median(num_objs))
        return dataset_dicts

class ClassBalancedSampler(Dataset):
    """
    Wrap a list to a torch Dataset. It produces elements of the list as data.
    """

    def __init__(self, lst, lst_len, sampled_num, oversampled_ratio, undersample_param, unique_relation_ratios, unique_relations):
        self._lst = lst
        self._len = lst_len
        self.sampled_num = sampled_num
        self.oversampled_ratio = oversampled_ratio
        self.undersample_param = undersample_param
        self.unique_relation_ratios = unique_relation_ratios
        self.unique_relations = unique_relations
        self._num_classes = len(lst.keys())

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        class_idx = np.random.randint(self._num_classes)
        random_example = np.random.randint(len(self._lst[class_idx]))
        record = self._lst[class_idx][random_example]
        relations = record['relations']
        new_record = copy.deepcopy(record)
        if len(relations) > 0:
            unique_relations = self.unique_relations[class_idx][random_example]
            rc = self.unique_relation_ratios[class_idx][random_example]
            ri = self.sampled_num[class_idx][random_example]
            dropout = np.clip(((ri - rc)/ri) * self.undersample_param, 0.0, 1.0)
            random_arr = np.random.uniform(size=len(relations))
            index_arr = np.array([unique_relations[rel] for rel in relations[:, 2]])
            rel_dropout = dropout[index_arr]
            to_keep = rel_dropout < random_arr
            dropped_relations = relations[to_keep]
            new_record['relations'] = dropped_relations
        return new_record

class BGNNSampler(Dataset):
    """
    Wrap a list to a torch Dataset. It produces elements of the list as data.
    """

    def __init__(self, lst, sampled_num, oversampled_ratio, undersample_param, unique_relation_ratios, unique_relations):
        self._lst = lst
        self.sampled_num = sampled_num
        self.oversampled_ratio = oversampled_ratio
        self.undersample_param = undersample_param
        self.unique_relation_ratios = unique_relation_ratios
        self.unique_relations = unique_relations
        

    def __len__(self):
        return len(self._lst)

    def __getitem__(self, idx):
        record = self._lst[idx]
        relations = record['relations']
        new_record = copy.deepcopy(record)
        if len(relations) > 0:
            unique_relations = self.unique_relations[idx]
            rc = self.unique_relation_ratios[idx]
            ri = self.sampled_num[idx]
            dropout = np.clip(((ri - rc)/ri) * self.undersample_param, 0.0, 1.0)
            random_arr = np.random.uniform(size=len(relations))
            index_arr = np.array([unique_relations[rel] for rel in relations[:, 2]])
            rel_dropout = dropout[index_arr]
            to_keep = rel_dropout < random_arr
            dropped_relations = relations[to_keep]
            new_record['relations'] = dropped_relations

        return new_record
   

def box_filter(boxes, must_overlap=False):
    """ Only include boxes that overlap as possible relations. 
    If no overlapping boxes, use all of them."""
    n_cands = boxes.shape[0]

    overlaps = bbox_overlaps(boxes.astype(np.float), boxes.astype(np.float), to_move=0) > 0
    np.fill_diagonal(overlaps, 0)

    all_possib = np.ones_like(overlaps, dtype=np.bool)
    np.fill_diagonal(all_possib, 0)

    if must_overlap:
        possible_boxes = np.column_stack(np.where(overlaps))

        if possible_boxes.size == 0:
            possible_boxes = np.column_stack(np.where(all_possib))
    else:
        possible_boxes = np.column_stack(np.where(all_possib))
    return possible_boxes

def bbox_overlaps(boxes1, boxes2, to_move=1):
    """
    boxes1 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    boxes2 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    """
    #print('boxes1: ', boxes1.shape)
    #print('boxes2: ', boxes2.shape)
    num_box1 = boxes1.shape[0]
    num_box2 = boxes2.shape[0]
    lt = np.maximum(boxes1.reshape([num_box1, 1, -1])[:,:,:2], boxes2.reshape([1, num_box2, -1])[:,:,:2]) # [N,M,2]
    rb = np.minimum(boxes1.reshape([num_box1, 1, -1])[:,:,2:], boxes2.reshape([1, num_box2, -1])[:,:,2:]) # [N,M,2]

    wh = (rb - lt + to_move).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    return inter