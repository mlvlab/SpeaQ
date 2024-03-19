import os
import copy
import numpy as np
import torch
from fvcore.common.file_io import PathManager
from PIL import Image

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures.instances import Instances
from detectron2.data import DatasetCatalog, MetadataCatalog, MapDataset, DatasetFromList, DatasetMapper
from collections import defaultdict
from imantics import Polygons, Mask
import logging
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    RotatedBoxes,
    polygons_to_bitmask,
)

def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    logger = logging.getLogger(__name__)
    tfm_gens = []
    if is_train:
        tfm_gens.append(T.RandomFlip())
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens

def filter_empty_instances(instances, by_box=True, by_mask=True, box_threshold=1e-5):
    """
    Filter out empty instances in an `Instances` object.
    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks
        box_threshold (float): minimum width and height to be considered non-empty
    Returns:
        Instances: the filtered instances.
    """
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty(threshold=box_threshold))
    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.nonempty())

    # TODO: can also filter visible keypoints

    if not r:
        return instances
    m = r[0]
    # for x in r[1:]:
    #     m = m & x
    return instances[m], r

class DetrDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by DETR.
    The callable currently does the following:
    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = [
                T.ResizeShortestEdge([400, 500, 600], sample_style="choice"),
                T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE),
            ]
            self.recompute_boxes = cfg.MODEL.MASK_ON
        else:
            self.crop_gen = None
            self.recompute_boxes = False

        self.mask_on = cfg.MODEL.MASK_ON
        self.tfm_gens = build_transform_gen(cfg, is_train)
        logging.getLogger(__name__).info(
            "Full TransformGens used in training: {}, crop: {}".format(str(self.tfm_gens), str(self.crop_gen))
        )

        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train
        self.filter_duplicate_relations = cfg.DATASETS.VISUAL_GENOME.FILTER_DUPLICATE_RELATIONS
        self.max_num_rels = cfg.DATASETS.VISUAL_GENOME.MAX_NUM_RELATIONS
        self.max_num_objs = cfg.DATASETS.VISUAL_GENOME.MAX_NUM_OBJECTS
        self.data_type = cfg.DATASETS.TYPE


    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        h, w, _ = image.shape
        if w != dataset_dict['width'] or h != dataset_dict['height']:
            dataset_dict['width'] = w
            dataset_dict['height'] = h
        utils.check_image_size(dataset_dict, image)

        if self.crop_gen is None:
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        else:
            if np.random.rand() > 0.5:
                image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            else:
                image, transforms = T.apply_transform_gens(
                    self.tfm_gens[:-1] + self.crop_gen + self.tfm_gens[-1:], image
                )

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # if not self.is_train:
        #     # USER: Modify this if you want to keep them for some reason.
        #     dataset_dict.pop("annotations", None)
        #     return dataset_dict

        # Filter duplicate relations
        rel_present = False
        if "relations" in dataset_dict:
            if self.filter_duplicate_relations and self.is_train:
                relation_dict = defaultdict(list)
                for object_0, object_1, relation in dataset_dict["relations"]:
                    relation_dict[(object_0,object_1)].append(relation)
                dataset_dict["relations"] = [(k[0], k[1], np.random.choice(v)) for k,v in relation_dict.items()]
                
            dataset_dict["relations"] = torch.as_tensor(np.ascontiguousarray(dataset_dict["relations"]))
            rel_present = True
        
        if self.data_type == "VISUAL GENOME RELATION":
            if self.filter_duplicate_relations and self.is_train:
                relation_dict = defaultdict(list)
                relation_idx = defaultdict(list)
                for idx, (object_0, object_1, relation) in enumerate(dataset_dict['relation_mapper']):
                    relation_dict[(object_0,object_1)].append(relation)
                    relation_idx[(object_0,object_1)].append(idx)
                selected_idxs = []
                for k, v in relation_idx.items():
                    selected_idxs.append(np.random.choice(v))
                selected_idxs = np.sort(selected_idxs)
                dataset_dict['annotations'] = [dataset_dict['annotations'][selected_idx] for selected_idx in selected_idxs]


        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            if self.data_type != "VISUAL GENOME RELATION":                
                annos = [
                    utils.transform_instance_annotations(obj, transforms, image_shape)
                    for obj in dataset_dict.pop("annotations")
                    if obj.get("iscrowd", 0) == 0
                ]
                instances = utils.annotations_to_instances(annos, image_shape)
            else:
                annos = [
                    transform_instance_annotations_relation(obj, transforms, image_shape)
                    for obj in dataset_dict.pop("annotations")
                    if obj.get("iscrowd", 0) == 0
                ]
                instances = annotations_to_instances_relation(annos, image_shape)           
                
            if rel_present:
                # Add object attributes
                instances.gt_attributes = torch.tensor([obj['attribute'] for obj in annos], dtype=torch.int64)
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"], filter_mask = utils.filter_empty_instances(instances, return_mask=True)
            
            # Fix GT relations where boxes are removed due to them being too small.
            if rel_present:
                if not filter_mask.all():
                    object_mapper = {int(old_idx): new_idx for new_idx, old_idx in enumerate(torch.arange(filter_mask.size(0))[filter_mask])}
                    new_relations = []
                    for idx, (object_0, object_1, relation) in enumerate(dataset_dict['relations'].numpy()):
                        if (object_0 in object_mapper) and (object_1 in object_mapper):
                            new_relations.append([object_mapper[object_0], object_mapper[object_1], relation])
                    if len(new_relations) > 0:
                        dataset_dict['relations'] = torch.tensor(new_relations)
                    else:
                        dataset_dict['relations'] = torch.zeros(0, 3).long()
                else:
                    if len(dataset_dict['relations']) == 0:
                        dataset_dict['relations'] = torch.zeros(0, 3).long()
            else:
                dataset_dict['relations'] = torch.zeros(0, 3).long()
            
            if self.data_type == "GQA":
                if len(dataset_dict['instances']) > self.max_num_objs:
                    # Randomly sample max number of objects
                    sample_idxs = np.random.permutation(np.arange(len(dataset_dict['instances'])))[:self.max_num_objs]
                    dataset_dict['instances'] = dataset_dict['instances'][sample_idxs]
                    object_mapper = {sample_idx:new_idx for new_idx, sample_idx in enumerate(sample_idxs)}
                    if len(dataset_dict['relations']) > 0:
                        new_relations = []
                        for idx, (object_0, object_1, relation) in enumerate(dataset_dict['relations'].numpy()):
                            if (object_0 in object_mapper) and (object_1 in object_mapper):
                                new_relations.append([object_mapper[object_0], object_mapper[object_1], relation])
                        if len(new_relations) > 0:
                            dataset_dict['relations'] = torch.tensor(new_relations)
                        else:
                            dataset_dict['relations'] = torch.zeros(0, 3).long()

                if len(dataset_dict['relations']) > self.max_num_rels:
                    sample_idxs = np.random.permutation(np.arange(len(dataset_dict['relations'])))[:self.max_num_rels]
                    dataset_dict['relations'] = dataset_dict['relations'][sample_idxs]
        
        return dataset_dict

def transform_instance_annotations_relation(
    annotation, transforms, image_size, *, keypoint_hflip_indices=None
):
    """
    Apply transforms to box, segmentation and keypoints annotations of a single instance.
    It will use `transforms.apply_box` for the box, and
    `transforms.apply_coords` for segmentation polygons & keypoints.
    If you need anything more specially designed for each data structure,
    you'll need to implement your own version of this function or the transforms.
    Args:
        annotation (dict): dict of instance annotations for a single instance.
            It will be modified in-place.
        transforms (TransformList or list[Transform]):
        image_size (tuple): the height, width of the transformed image
        keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.
    Returns:
        dict:
            the same input dict with fields "bbox", "segmentation", "keypoints"
            transformed according to `transforms`.
            The "bbox_mode" field will be set to XYXY_ABS.
    """
    if isinstance(transforms, (tuple, list)):
        transforms = T.TransformList(transforms)
    # bbox is 1d (per-instance bounding box)
    bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    bbox_union = BoxMode.convert(annotation["union_bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    # clip transformed bbox to image size
    bbox = transforms.apply_box(np.array([bbox]))[0].clip(min=0)
    bbox_union = transforms.apply_box(np.array([bbox_union]))[0].clip(min=0)
    annotation["bbox"] = np.minimum(bbox, list(image_size + image_size)[::-1])
    annotation["bbox_union"] = np.minimum(bbox_union, list(image_size + image_size)[::-1])
    annotation["bbox_mode"] = BoxMode.XYXY_ABS
    return annotation

def annotations_to_instances_relation(annos, image_size, mask_format="polygon"):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.
    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width
    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes",
            "gt_masks", "gt_keypoints", if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    boxes = (
        np.stack(
            [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
        )
        if len(annos)
        else np.zeros((0, 4))
    )
    union_boxes = (
        np.stack(
            [BoxMode.convert(obj["bbox_union"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
        )
        if len(annos)
        else np.zeros((0, 4))
    )
    target = Instances(image_size)
    target.gt_boxes = Boxes(boxes)
    target.union_boxes = Boxes(union_boxes)

    classes = [int(obj["category_id"]) for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    return target

