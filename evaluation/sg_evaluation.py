import logging
import copy 
import os
from typing import OrderedDict
from collections import defaultdict
import torch
import numpy as np
import json
from tqdm import tqdm
from functools import reduce
import itertools

from abc import ABC, abstractmethod

from ..data import DetrDatasetMapper

from fvcore.common.file_io import PathManager
import detectron2.utils.comm as comm
from detectron2.structures import Instances
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.evaluation import COCOEvaluator
from detectron2.structures.boxes import pairwise_iou, Boxes
from detectron2.utils.registry import Registry


from .utils import intersect_2d, argsort_desc

idx2predicate = {0: 'above', 1: 'across', 2: 'against', 3: 'along', 4: 'and', 5: 'at', 6: 'attached to', 7: 'behind', 8: 'belonging to', 9: 'between', 10: 'carrying',
                11: 'covered in', 12: 'covering', 13: 'eating', 14: 'flying in', 15: 'for', 16: 'from', 17: 'growing on', 18: 'hanging from', 19: 'has', 20: 'holding',
                21: 'in', 22: 'in front of', 23: 'laying on', 24: 'looking at', 25: 'lying on', 26: 'made of', 27: 'mounted on', 28: 'near', 29: 'of', 30: 'on',
                31: 'on back of', 32: 'over', 33: 'painted on', 34: 'parked on', 35: 'part of', 36: 'playing', 37: 'riding', 38: 'says', 39: 'sitting on', 40: 'standing on',
                41: 'to', 42: 'under', 43: 'using', 44: 'walking in', 45: 'walking on', 46: 'watching', 47: 'wearing', 48: 'wears', 49: 'with'}

SCENEGRAPH_METRIC_REGISTRY = Registry("SCENEGRAPH_METRIC_REGISTRY")

class SceneGraphEvaluator(DatasetEvaluator):

    def __init__(self, dataset_name, cfg, distributed, output_dir=None, metrics=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            cfg (CfgNode): config instance
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will only evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instance_predictions.pth" a file in torch serialization
                   format that contains all the raw original predictions.
                2. "instances_results.json" a json file in COCO's result
                   format. #TODO: fix the commnent after implementation
            metrics (tuple): The metrics using which the scene graphs performance should be evaluated
                Options: ('SGRecall', 'SGNoGraphConstraintRecall', 'SGZeroShotRecall', 'SGPairAccuracy', 'SGMeanRecall')
        """

        SGMETRICS = ('SGRecall', 'SGNoGraphConstraintRecall', 'SGZeroShotRecall', 'SGPairAccuracy', 'SGMeanRecall','query_count_per_rel_class','rel_class_info_per_query','recall_per_class')

        self._mode = self._mode_from_config(cfg)
        self._distributed = distributed
        self._output_dir = output_dir
        self.cfg = cfg

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger('detectron2')

        if metrics is None:
            self._metrics = SGMETRICS
        else:
            for metric in metrics:
                assert metric in SGMETRICS, "Specified scene graph evaluation metric {} not suppoted. Currently supported metrics : {}".format(metric, SGMETRICS)
            self._metrics = metrics
        
        self._logger.info("Following metrics will be use for evaluation")
        self._logger.info("{}".format(self._metrics))

        self.detection_evaluator = COCOEvaluator(dataset_name, cfg, distributed, output_dir)
        self.detection_evaluator._tasks =  ("bbox",)
        #Register a filed for each of the metric
        self._evaluators = build_scenegraph_evaluators(self._metrics, cfg, {}, dataset_name)
        # self._register_evaluator_containers()

        self._metadata = MetadataCatalog.get(dataset_name)

        self._ground_truths = []
        self._predictions = []
        self._zero_shot_triplets = self._get_zero_shot_triplets() - 1

    def reset(self,total):
        self.detection_evaluator.reset()
        self._register_evaluator_containers(total)

    def _get_zero_shot_triplets(self):
        self._logger.info('Loading zero shot triplets')
        return torch.load(self.cfg.MODEL.ROI_SCENEGRAPH_HEAD.ZERO_SHOT_TRIPLETS , map_location=torch.device("cpu")).long().numpy()

    def _mode_from_config(self, cfg):
        '''
        Estimate mode from configuration
        '''
        if cfg.MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_BOX:
            if cfg.MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_OBJECT_LABEL:
                mode = 'predcls'
            else:
                mode = 'sgcls'
        else:
            mode = 'sgdet'

        return mode
    
    def _register_evaluator_containers(self,total):
        for evaluator in self._evaluators.keys():
            self._evaluators[evaluator].register_container(self._mode,self.cfg,total)

    def process(self, inputs, outputs):
        # import pdb; pdb.set_trace()
        for idx, input in enumerate(inputs):
            height, width = outputs[idx]['instances'].image_size
            input['instances'] = resize_instance(input['instances'], height, width)
        
        self.detection_evaluator.process(inputs, outputs)

        for input, output in zip(inputs, outputs):
            ground_truth = {}
            prediction = {}

            ground_truth['relation_tuple'] = input['relations'].to(self._cpu_device) #Relation tupe (obj_id, sub_id, relation label)
            ground_truth['gt_boxes'] = input['instances'].gt_boxes.to(self._cpu_device) #Ground truth object boxes
            ground_truth['labels'] = input['instances'].gt_classes.to(self._cpu_device) #Ground truth object classes
            ground_truth['rel_pair_idxs'] = input['relations'][:,:2].to(self._cpu_device) #Realtion pair index (shape: (num of relations, 2))

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["image_id"] = input["image_id"]
                prediction["instances"] = instances
                prediction['rel_pair_idxs'] = output["rel_pair_idxs"].to(self._cpu_device)
                prediction['pred_rel_scores'] = output["pred_rel_scores"].to(self._cpu_device)
                prediction['pred_rel_labels'] = output['pred_rel_labels'].to(self._cpu_device)
                prediction['query_index']=output['query_index'].to(self._cpu_device)
#                 prediction['subject_scores'] = output['subject_scores'].to(self._cpu_device)
#                 prediction['object_scores'] = output['object_scores'].to(self._cpu_device)
#                 prediction['predicate_scores'] = output['predicate_scores'].to(self._cpu_device)
#                 prediction['triplet_scores'] = output['triplet_scores'].to(self._cpu_device)
                
                if 'pred_rel_scores_1' in output:
                    instances_1 = output['instances_1'].to(self._cpu_device)
                    prediction['instances_1'] = instances_1
                    prediction['rel_pair_idxs_1'] = output["rel_pair_idxs_1"].to(self._cpu_device)
                    prediction['pred_rel_scores_1'] = output["pred_rel_scores_1"].to(self._cpu_device)

                if 'pred_rel_scores_3' in output:
                    instances_3 = output['instances_3'].to(self._cpu_device)
                    prediction['instances_3'] = instances_3
                    prediction['rel_pair_idxs_3'] = output["rel_pair_idxs_3"].to(self._cpu_device)
                    prediction['pred_rel_scores_3'] = output["pred_rel_scores_3"].to(self._cpu_device)

            
            ground_truth_cp = copy.deepcopy(ground_truth)
            prediction_cp = copy.deepcopy(prediction)
        
            del ground_truth 
            del prediction
            self._ground_truths.append(ground_truth_cp)
            self._predictions.append(prediction_cp)
        del outputs
        del inputs 

    def evaluate(self):
        #First evaluate the detection precisions
        result_detector = self.detection_evaluator.evaluate()

        if self._distributed:
            comm.synchronize()
            self._logger.info("Gathering data")
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            ground_truths = comm.gather(self._ground_truths, dst=0)
            ground_truths = list(itertools.chain(*ground_truths))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions
            groundtruths = self._ground_truths

        self._logger.info("Predictions Gathered")

        if len(predictions) == 0:
            self._logger.warning("[SceneGraphEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "scenegraph_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save({'groundtruths':ground_truths, 'predictions':predictions}, f)
        self._logger.info("Saving output prediction")

        result_detector['SG'] = self._evaluate_scenegraphs(ground_truths, predictions)
        
        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "result_dict.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(self._evaluators['SGRecall'].result_dict, f)
        
        return result_detector 

    def _evaluate_scenegraphs(self, ground_truths, predictions):
        
        # result_detector = None
        
        self._logger.info("Computing Scene Graph Metrics")
        num_rel_category = self.cfg.MODEL.ROI_SCENEGRAPH_HEAD.NUM_CLASSES
        multiple_preds = self.cfg.TEST.RELATION.MULTIPLE_PREDS
        iou_thres = self.cfg.TEST.RELATION.IOU_THRESHOLD

        self._logger.info("Preparing Global Container")
        #Prepare Global container
        global_container = {}
        global_container['zeroshot_triplet'] = self._zero_shot_triplets
        global_container['result_dict'] = {}
        global_container['mode'] = self._mode
        global_container['multiple_preds'] = multiple_preds
        global_container['num_rel_category'] = num_rel_category
        global_container['iou_thres'] = iou_thres
        
        for i , (groundtruth, prediction) in tqdm(enumerate(zip(ground_truths, predictions)),desc='Computing recalls'):
            self.evaluate_relation_of_one_image(groundtruth, prediction, global_container,i)

        self._logger.info("Scene Graph Metric Evaluation Complete. Computing recall statistics...")
        # ('SGRecall', 'SGNoGraphConstraintRecall', 'SGZeroShotRecall', 'SGPairAccuracy', 'SGMeanRecall')
        if 'SGMeanRecall' in self._evaluators:
            # calculate mean recall
            self._evaluators['SGMeanRecall'].calculate_mean_recall(self._mode)

        result_str =''
        # print result
        if 'SGRecall' in self._evaluators:
            result_str += self._evaluators['SGRecall'].generate_print_string(self._mode)
        if 'SGNoGraphConstraintRecall' in self._evaluators:
            result_str += self._evaluators['SGNoGraphConstraintRecall'].generate_print_string(self._mode)
        if 'SGZeroShotRecall' in self._evaluators:
            result_str += self._evaluators['SGZeroShotRecall'].generate_print_string(self._mode)
        if 'SGMeanRecall' in self._evaluators:
            result_str += self._evaluators['SGMeanRecall'].generate_print_string(self._mode)
        
        if self.cfg.MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_BOX and 'SGPairAccuracy' in self._evaluators:
            result_str += self._evaluators['SGPairAccuracy'].generate_print_string(self._mode)
        result_str += self._evaluators['recall_per_class'].generate_print_string(self._mode,self.cfg,len(ground_truths))
        result_str += '=' * 100 + '\n'
        
        torch.save(self._evaluators['SGRecall'].result_dict, 'temp.pth')
        self._logger.info('Scene Graph Results for mode: {}'.format(self._mode))
        print(result_str)
        ret = OrderedDict()
        for k, v in self._evaluators['SGMeanRecall'].result_dict[self._mode + '_mean_recall'].items():
            ret['SGMeanRecall@{}'.format(k)] = float(v)
        for k, v in self._evaluators['SGRecall'].result_dict[self._mode + '_recall'].items():
            ret['SGRecall@{}'.format(k)] = np.mean(v)
        return ret

    def evaluate_relation_of_one_image(self, groundtruth, prediction, global_container, i):
        """
        Returns:
            pred_to_gt: Matching from predicate to GT
            pred_5ples: the predicted (id0, id1, cls0, cls1, rel)
            pred_triplet_scores: [cls_0score, relscore, cls1_score]
        """
        #unpack all inputs
        mode = global_container['mode']

        local_container = {}
        local_container['gt_rels'] = groundtruth['relation_tuple'].long().detach().cpu().numpy()

        # if there is no gt relations for current image, then skip it
        if len(local_container['gt_rels']) == 0:
            return

        local_container['gt_boxes'] = groundtruth['gt_boxes'].tensor.detach().cpu().numpy()                   # (#gt_objs, 4)
        local_container['gt_classes'] = groundtruth['labels'].long().detach().cpu().numpy()           # (#gt_objs, )
        # import ipdb; ipdb.set_trace()
        # about relations
        local_container['pred_rel_inds'] = prediction['rel_pair_idxs'].long().detach().cpu().numpy()  # (#pred_rels, 2)
        local_container['rel_scores'] = prediction['pred_rel_scores'].detach().cpu().numpy()          # (#pred_rels, num_pred_class)
        local_container['rel_labels'] = prediction['pred_rel_labels'].detach().cpu().numpy()
        local_container['query_index']=prediction['query_index'].detach().cpu().numpy()
#         local_container['subject_scores']=prediction['subject_scores'].detach().cpu().numpy()
#         local_container['object_scores']=prediction['object_scores'].detach().cpu().numpy()
#         local_container['predicate_scores']=prediction['predicate_scores'].detach().cpu().numpy()
#         local_container['triplet_scores']=prediction['triplet_scores'].detach().cpu().numpy()

        # about objects
        local_container['pred_boxes'] = prediction['instances'].pred_boxes.tensor.detach().cpu().numpy()                  # (#pred_objs, 4)
        local_container['pred_classes'] = prediction['instances'].pred_classes.long().detach().cpu().numpy()     # (#pred_objs, )
        local_container['obj_scores'] = prediction['instances'].scores.detach().cpu().numpy()              # (#pred_objs, )
        # import pdb; pdb.set_trace()
        # to calculate accuracy, only consider those gt pairs
        # This metric is used by "Graphical Contrastive Losses for Scene Graph Parsing" 
        # for sgcls and predcls
        if mode != 'sgdet' and 'SGPairAccuracy' in self._metrics:
            self._evaluators['SGPairAccuracy'].prepare_gtpair(local_container)

        # to calculate the prior label based on statistics
        if 'SGZeroShotRecall' in self._metrics:
            self._evaluators['SGZeroShotRecall'].prepare_zeroshot(global_container, local_container)

        if mode == 'predcls':
            local_container['pred_boxes'] = local_container['gt_boxes']
            local_container['pred_classes'] = local_container['gt_classes']
            local_container['obj_scores'] = np.ones(local_container['gt_classes'].shape[0])

        elif mode == 'sgcls':
            if local_container['gt_boxes'].shape[0] != local_container['pred_boxes'].shape[0]:
                print('Num of GT boxes is not matching with num of pred boxes in SGCLS')
        elif mode == 'sgdet' or mode == 'phrdet':
            pass
        else:
            raise ValueError('invalid mode')

        if local_container['pred_rel_inds'].shape[0] == 0:
            return

        # Traditional Metric with Graph Constraint
        # NOTE: this is the MAIN evaluation function, it must be run first (several important variables need to be update)
        # ('SGRecall', 'SGNoGraphConstraintRecall', 'SGZeroShotRecall', 'SGPairAccuracy', 'SGMeanRecall')

        local_container = self._evaluators['SGRecall'].calculate_recall(global_container, local_container, mode)

        self._evaluators['query_count_per_rel_class'].calculate(global_container, local_container, mode,i)
        self._evaluators['rel_class_info_per_query'].calculate(global_container, local_container, mode)
        self._evaluators['recall_per_class'].calculate(self.cfg,global_container, local_container, mode,i)        
        if 'SGNoGraphConstraintRecall' in self._metrics:
            # No Graph Constraint
            self._evaluators['SGNoGraphConstraintRecall'].calculate_recall(global_container, local_container, mode)
        if 'SGPairAccuracy' in self._metrics:
            # GT Pair Accuracy
            self._evaluators['SGPairAccuracy'].calculate_recall(global_container, local_container, mode)
        if 'SGMeanRecall' in self._metrics:
            # Mean Recall
            self._evaluators['SGMeanRecall'].collect_mean_recall_items(global_container, local_container, mode)
        if 'SGZeroShotRecall' in self._metrics:
            # Zero shot Recall
            self._evaluators['SGZeroShotRecall'].calculate_recall(global_container, local_container, mode, i)
        return 
        

class SceneGraphEvaluation(ABC):
    def __init__(self, result_dict):
        super().__init__()
        self.result_dict = result_dict
 
    @abstractmethod
    def register_container(self, mode, cfg):
        print("Register Result Container")
        pass
    
    @abstractmethod
    def generate_print_string(self, mode):
        print("Generate Print String")
        pass

@SCENEGRAPH_METRIC_REGISTRY.register()
class query_count_per_rel_class(SceneGraphEvaluation):
    def __init__(self, cfg, result_dict, dataset_name):
        super(query_count_per_rel_class, self).__init__(result_dict)
        

    def register_container(self, mode, cfg,total):
        query_num=cfg.MODEL.DETR.NUM_RELATION_QUERIES*cfg.MODEL.DETR.MULTIPLY_QUERY*cfg.TEST.NUM_REL
        num_rel_category=cfg.MODEL.ROI_SCENEGRAPH_HEAD.NUM_CLASSES
        self.result_dict['query_count_per_rel_class'] = {20: torch.zeros(num_rel_category,query_num), 50: torch.zeros(num_rel_category,query_num), 100: torch.zeros(num_rel_category,query_num)}
        self.result_dict['gt_match_query_per_image']={}
    def generate_print_string(self, mode):
        pass

    def calculate(self, global_container, local_container, mode,i):
        pred_rel_inds = local_container['pred_rel_inds']
        rel_scores = local_container['rel_scores']
        rel_labels = local_container['rel_labels']
        query_index = local_container['query_index']

        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        gt_boxes = local_container['gt_boxes']
        pred_classes = local_container['pred_classes']
        pred_boxes = local_container['pred_boxes']
        obj_scores = local_container['obj_scores']

        iou_thres = global_container['iou_thres']

        pred_rels = np.column_stack((pred_rel_inds, rel_labels)) #Backround index at the end
        pred_scores = rel_scores[:,:-1].max(1) #Backround index at the end

        gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels, gt_classes, gt_boxes)
        local_container['gt_triplets'] = gt_triplets
        local_container['gt_triplet_boxes'] = gt_triplet_boxes
        
        pred_triplets, pred_triplet_boxes, pred_triplet_scores = _triplet(
                pred_rels, pred_classes, pred_boxes, pred_scores, obj_scores)

        # Compute recall. It's most efficient to match once and then do recall after
        pred_to_gt,index_list,recall_per_class_index = _compute_pred_matches(
            gt_triplets,
            pred_triplets,
            gt_triplet_boxes,
            pred_triplet_boxes,
            iou_thres,
            phrdet=mode=='phrdet',
        )
        self.result_dict['gt_match_query_per_image'][i]=index_list
        for k in self.result_dict['query_count_per_rel_class']:
            for prediction_index in np.where(index_list<k)[0]:
                index=index_list[prediction_index]
                self.result_dict['query_count_per_rel_class'][k][rel_labels[index]][query_index[index]]+=1

@SCENEGRAPH_METRIC_REGISTRY.register()
class rel_class_info_per_query(SceneGraphEvaluation):
    def __init__(self, cfg, result_dict, dataset_name):
        super(rel_class_info_per_query, self).__init__(result_dict)
        
    def register_container(self, mode, cfg,total):
        query_num=cfg.MODEL.DETR.NUM_RELATION_QUERIES*cfg.MODEL.DETR.MULTIPLY_QUERY*cfg.TEST.NUM_REL
        num_rel_category=cfg.MODEL.ROI_SCENEGRAPH_HEAD.NUM_CLASSES
        self.result_dict['rel_class_count_per_query'] = {20: torch.zeros(query_num,num_rel_category), 50: torch.zeros(query_num,num_rel_category), 100: torch.zeros(query_num,num_rel_category)}
        self.result_dict['rel_class_logit_per_query'] = {20: torch.zeros(query_num,num_rel_category+1), 50: torch.zeros(query_num,num_rel_category+1), 100: torch.zeros(query_num,num_rel_category+1)}

    def generate_print_string(self, mode):
        pass

    def calculate(self, global_container, local_container, mode):
        pred_rel_inds = local_container['pred_rel_inds']
        rel_scores = local_container['rel_scores']
        rel_labels = local_container['rel_labels']
        query_index = local_container['query_index']

        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        gt_boxes = local_container['gt_boxes']
        pred_classes = local_container['pred_classes']
        pred_boxes = local_container['pred_boxes']
        obj_scores = local_container['obj_scores']
    
        iou_thres = global_container['iou_thres']

        pred_rels = np.column_stack((pred_rel_inds, rel_labels)) #Backround index at the end
        pred_scores = rel_scores[:,:-1].max(1) #Backround index at the end

        gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels, gt_classes, gt_boxes)
        local_container['gt_triplets'] = gt_triplets
        local_container['gt_triplet_boxes'] = gt_triplet_boxes
        
        pred_triplets, pred_triplet_boxes, pred_triplet_scores = _triplet(
                pred_rels, pred_classes, pred_boxes, pred_scores, obj_scores)

        # Compute recall. It's most efficient to match once and then do recall after
        pred_to_gt,index_list,recall_per_class_index = _compute_pred_matches(
            gt_triplets,
            pred_triplets,
            gt_triplet_boxes,
            pred_triplet_boxes,
            iou_thres,
            phrdet=mode=='phrdet',
        )

        for k in self.result_dict['rel_class_count_per_query']:
            for prediction_index in np.where(index_list<k)[0]:
                index=index_list[prediction_index]
                self.result_dict['rel_class_count_per_query'][k][query_index[index]][rel_labels[index]]+=1

        for k in self.result_dict['rel_class_logit_per_query']:
            for prediction_index in np.where(index_list<k)[0]:
                index=index_list[prediction_index]
                self.result_dict['rel_class_logit_per_query'][k][query_index[index]].add_(torch.tensor(rel_scores[index]))

@SCENEGRAPH_METRIC_REGISTRY.register()
class recall_per_class(SceneGraphEvaluation):
    def __init__(self, cfg, result_dict, dataset_name):
        super(recall_per_class, self).__init__(result_dict)
        

    def register_container(self, mode,cfg,total):
        num_rel_category=cfg.MODEL.ROI_SCENEGRAPH_HEAD.NUM_CLASSES
        self.result_dict['count_per_class'] = torch.zeros(num_rel_category,total)
        self.result_dict['hit_per_class'] = {20: torch.zeros(num_rel_category,total), 50: torch.zeros(num_rel_category,total), 100: torch.zeros(num_rel_category,total)}
        self.result_dict['gt_pair']={i:{} for i in range(total)}
        self.result_dict['gt_pair_count']={i:0 for i in range(total)}
        self.result_dict['pred_pair']={k:{i:{} for i in range(total)} for k in [20,50,100]}
        self.result_dict['pred_pair_count']={k:{i:0 for i in range(total)} for k in [20,50,100]}
    
    def generate_print_string(self, mode,cfg,total):
        num_rel_category=cfg.MODEL.ROI_SCENEGRAPH_HEAD.NUM_CLASSES
        result_str = 'recall_per_class\n'
        for k in self.result_dict['hit_per_class']:
            result_str+= 'R @ %d\n' % (k)
            for predicate_num in range(num_rel_category):
                per_class_recall=[]
                for j in range(total):
                    count=self.result_dict['count_per_class'][predicate_num][j]
                    hit=self.result_dict['hit_per_class'][k][predicate_num][j]
                    if count>0:
                        per_class_recall.append(hit/count)
                result_str += '%15s: %6.3f; ' % (idx2predicate[predicate_num], np.mean(per_class_recall))
            result_str += '\n'
        return result_str

    def calculate(self, cfg, global_container, local_container, mode,i):
        num_rel_category=cfg.MODEL.ROI_SCENEGRAPH_HEAD.NUM_CLASSES

        pred_rel_inds = local_container['pred_rel_inds']
        rel_scores = local_container['rel_scores']
        rel_labels = local_container['rel_labels']
        query_index = local_container['query_index']

        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        gt_boxes = local_container['gt_boxes']
        pred_classes = local_container['pred_classes']
        pred_boxes = local_container['pred_boxes']
        obj_scores = local_container['obj_scores']

        iou_thres = global_container['iou_thres']

        pred_rels = np.column_stack((pred_rel_inds, rel_labels)) #Backround index at the end
        pred_scores = rel_scores[:,:-1].max(1) #Backround index at the end

        gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels, gt_classes, gt_boxes)
        local_container['gt_triplets'] = gt_triplets
        local_container['gt_triplet_boxes'] = gt_triplet_boxes
        
        pred_triplets, pred_triplet_boxes, pred_triplet_scores = _triplet(
                pred_rels, pred_classes, pred_boxes, pred_scores, obj_scores)

        # Compute recall. It's most efficient to match once and then do recall after
        pred_to_gt,index_list,recall_per_class_index = _compute_pred_matches(
            gt_triplets,
            pred_triplets,
            gt_triplet_boxes,
            pred_triplet_boxes,
            iou_thres,
            phrdet=mode=='phrdet',
        )

        for predicate in gt_rels[:,2]:
            self.result_dict['count_per_class'][predicate][i]+=1
        
        for gt_rel in gt_rels:
            subj,obj,predicate=gt_rel[0],gt_rel[1],gt_rel[2]
            if f'{subj}_{obj}' not in self.result_dict['gt_pair'][i]:
                self.result_dict['gt_pair'][i][f'{subj}_{obj}']=torch.zeros(num_rel_category)
            self.result_dict['gt_pair'][i][f'{subj}_{obj}'][predicate]+=1
        self.result_dict['gt_pair_count'][i]=len(gt_rels)

        for k in self.result_dict['hit_per_class']:
            # for prediction_index in np.where(recall_per_class_index<k)[0]:
            #     index=recall_per_class_index[prediction_index]
            #     self.result_dict['hit_per_class'][k][rel_labels[index]][i]+=1
            match=reduce(np.union1d,pred_to_gt[:k])
            match_count=reduce(lambda a,b:a+b,pred_to_gt[:k])
            for index in range(len(match)):
                match_subj,match_obj,match_predicate=gt_rels[int(match[index]),0],gt_rels[int(match[index]),1],gt_rels[int(match[index]),2]
                self.result_dict['hit_per_class'][k][match_predicate][i]+=1
                if f'{match_subj}_{match_obj}' not in self.result_dict['pred_pair'][k][i]:
                    self.result_dict['pred_pair'][k][i][f'{match_subj}_{match_obj}']=torch.zeros(num_rel_category)
                self.result_dict['pred_pair'][k][i][f'{match_subj}_{match_obj}'][match_predicate]+=match_count.count(match[index])
            self.result_dict['pred_pair_count'][k][i]=len(match_count)


"""
Traditional Recall, implement based on:
https://github.com/rowanz/neural-motifs
"""
@SCENEGRAPH_METRIC_REGISTRY.register()
class SGRecall(SceneGraphEvaluation):
    def __init__(self, cfg, result_dict, dataset_name):
        super(SGRecall, self).__init__(result_dict)
        

    def register_container(self, mode, cfg,total):
        self.result_dict[mode + '_recall'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_recall'].items():
            result_str += '    R @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=Recall(Main).' % mode
        result_str += '\n'
        return result_str

    def calculate_recall(self, global_container, local_container, mode):
        pred_rel_inds = local_container['pred_rel_inds']
        rel_scores = local_container['rel_scores']
        rel_labels = local_container['rel_labels']

        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        gt_boxes = local_container['gt_boxes']
        pred_classes = local_container['pred_classes']
        pred_boxes = local_container['pred_boxes']
        obj_scores = local_container['obj_scores']

        iou_thres = global_container['iou_thres']

        pred_rels = np.column_stack((pred_rel_inds, rel_labels)) #Backround index at the end
        pred_scores = rel_scores[:,:-1].max(1) #Backround index at the end

        gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels, gt_classes, gt_boxes)
        local_container['gt_triplets'] = gt_triplets
        local_container['gt_triplet_boxes'] = gt_triplet_boxes
        
        pred_triplets, pred_triplet_boxes, pred_triplet_scores = _triplet(
                pred_rels, pred_classes, pred_boxes, pred_scores, obj_scores)

        # Compute recall. It's most efficient to match once and then do recall after
        pred_to_gt,index_list,recall_per_class_index = _compute_pred_matches(
            gt_triplets,
            pred_triplets,
            gt_triplet_boxes,
            pred_triplet_boxes,
            iou_thres,
            phrdet=mode=='phrdet',
        )
        local_container['pred_to_gt'] = pred_to_gt

        for k in self.result_dict[mode + '_recall']:
            # the following code are copied from Neural-MOTIFS
            match = reduce(np.union1d, pred_to_gt[:k])
            rec_i = float(len(match)) / float(gt_rels.shape[0])
            self.result_dict[mode + '_recall'][k].append(rec_i) 
        return local_container

"""
No Graph Constraint Recall, implement based on:
https://github.com/rowanz/neural-motifs
"""
@SCENEGRAPH_METRIC_REGISTRY.register()
class SGNoGraphConstraintRecall(SceneGraphEvaluation):
    def __init__(self, cfg, result_dict, dataset_name):
        super(SGNoGraphConstraintRecall, self).__init__(result_dict)

    def register_container(self, mode, cfg,total):
        self.result_dict[mode + '_recall_nogc'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_recall_nogc'].items():
            result_str += ' ng-R @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=No Graph Constraint Recall(Main).' % mode
        result_str += '\n'
        return result_str

    def calculate_recall(self, global_container, local_container, mode):
        obj_scores = local_container['obj_scores']
        pred_rel_inds = local_container['pred_rel_inds']
        rel_scores = local_container['rel_scores']
        pred_boxes = local_container['pred_boxes']
        pred_classes = local_container['pred_classes']
        gt_rels = local_container['gt_rels']

        obj_scores_per_rel = obj_scores[pred_rel_inds].prod(1)
        nogc_overall_scores = obj_scores_per_rel[:,None] * rel_scores[:,:-1] #Backround index at the end
        nogc_score_inds = argsort_desc(nogc_overall_scores)[:100]
        nogc_pred_rels = np.column_stack((pred_rel_inds[nogc_score_inds[:,0]], nogc_score_inds[:,1]))#Backround index at the end(removed +1)
        nogc_pred_scores = rel_scores[nogc_score_inds[:,0], nogc_score_inds[:,1]]#Backround index at the end(removed +1)

        nogc_pred_triplets, nogc_pred_triplet_boxes, _ = _triplet(
                nogc_pred_rels, pred_classes, pred_boxes, nogc_pred_scores, obj_scores
        )

        # No Graph Constraint
        gt_triplets = local_container['gt_triplets']
        gt_triplet_boxes = local_container['gt_triplet_boxes']
        iou_thres = global_container['iou_thres']

        nogc_pred_to_gt,index_list,recall_per_class_index = _compute_pred_matches(
            gt_triplets,
            nogc_pred_triplets,
            gt_triplet_boxes,
            nogc_pred_triplet_boxes,
            iou_thres,
            phrdet=mode=='phrdet',
        )

        local_container['nogc_pred_to_gt'] = nogc_pred_to_gt

        for k in self.result_dict[mode + '_recall_nogc']:
            match = reduce(np.union1d, nogc_pred_to_gt[:k])
            rec_i = float(len(match)) / float(gt_rels.shape[0])
            self.result_dict[mode + '_recall_nogc'][k].append(rec_i)

        return local_container

"""
Zero Shot Scene Graph
Only calculate the triplet that not occurred in the training set
"""
@SCENEGRAPH_METRIC_REGISTRY.register()
class SGZeroShotRecall(SceneGraphEvaluation):
    def __init__(self, cfg,  result_dict, dataset_name):
        super(SGZeroShotRecall, self).__init__(result_dict)

    def register_container(self, mode, cfg,total):
        self.result_dict[mode + '_zeroshot_recall'] = {20: [], 50: [], 100: []} 
        self.result_dict[mode + '_zs_id'] = {20: [], 50: [], 100: []} 

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_zeroshot_recall'].items():
            result_str += '   zR @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=Zero Shot Recall.' % mode
        result_str += '\n'
        return result_str

    def prepare_zeroshot(self, global_container, local_container):
        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        zeroshot_triplets = global_container['zeroshot_triplet']

        sub_id, ob_id, pred_label = gt_rels[:, 0], gt_rels[:, 1], gt_rels[:, 2]
        gt_triplets = np.column_stack((gt_classes[sub_id], gt_classes[ob_id], pred_label))  # num_rel, 3

        self.zeroshot_idx = np.where( intersect_2d(gt_triplets, zeroshot_triplets).sum(-1) > 0 )[0].tolist()

    def calculate_recall(self, global_container, local_container, mode, i):
        pred_to_gt = local_container['pred_to_gt']

        for k in self.result_dict[mode + '_zeroshot_recall']:
            # Zero Shot Recall
            match = reduce(np.union1d, pred_to_gt[:k])
            if len(self.zeroshot_idx) > 0:
                if not isinstance(match, (list, tuple)):
                    match_list = match.tolist()
                else:
                    match_list = match
                zeroshot_match = len(self.zeroshot_idx) + len(match_list) - len(set(self.zeroshot_idx + match_list))
                zero_rec_i = float(zeroshot_match) / float(len(self.zeroshot_idx))
                self.result_dict[mode + '_zeroshot_recall'][k].append(zero_rec_i)
                self.result_dict[mode + '_zs_id'][k].append(i)


"""
Give Ground Truth Object-Subject Pairs
Calculate Recall for SG-Cls and Pred-Cls
Only used in https://github.com/NVIDIA/ContrastiveLosses4VRD for sgcls and predcls
"""
@SCENEGRAPH_METRIC_REGISTRY.register()
class SGPairAccuracy(SceneGraphEvaluation):
    def __init__(self, cfg, result_dict, dataset_name):
        super(SGPairAccuracy, self).__init__(result_dict)

    def register_container(self, mode, cfg,total):
        self.result_dict[mode + '_accuracy_hit'] = {20: [], 50: [], 100: []}
        self.result_dict[mode + '_accuracy_count'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_accuracy_hit'].items():
            a_hit = np.mean(v)
            a_count = np.mean(self.result_dict[mode + '_accuracy_count'][k])
            result_str += '    A @ %d: %.4f; ' % (k, a_hit/a_count)
        result_str += ' for mode=%s, type=TopK Accuracy.' % mode
        result_str += '\n'
        return result_str

    def prepare_gtpair(self, local_container):
        pred_pair_idx = local_container['pred_rel_inds'][:, 0] * 1024 + local_container['pred_rel_inds'][:, 1]
        gt_pair_idx = local_container['gt_rels'][:, 0] * 1024 + local_container['gt_rels'][:, 1]
        self.pred_pair_in_gt = (pred_pair_idx[:, None] == gt_pair_idx[None, :]).sum(-1) > 0

    def calculate_recall(self, global_container, local_container, mode):
        pred_to_gt = local_container['pred_to_gt']
        gt_rels = local_container['gt_rels']

        for k in self.result_dict[mode + '_accuracy_hit']:
            # to calculate accuracy, only consider those gt pairs
            # This metric is used by "Graphical Contrastive Losses for Scene Graph Parsing" 
            # for sgcls and predcls
            if mode != 'sgdet':
                gt_pair_pred_to_gt = []
                for p, flag in zip(pred_to_gt, self.pred_pair_in_gt):
                    if flag:
                        gt_pair_pred_to_gt.append(p)
                if len(gt_pair_pred_to_gt) > 0:
                    gt_pair_match = reduce(np.union1d, gt_pair_pred_to_gt[:k])
                else:
                    gt_pair_match = []
                self.result_dict[mode + '_accuracy_hit'][k].append(float(len(gt_pair_match)))
                self.result_dict[mode + '_accuracy_count'][k].append(float(gt_rels.shape[0]))


"""
Mean Recall: Proposed in:
https://arxiv.org/pdf/1812.01880.pdf CVPR, 2019
"""
@SCENEGRAPH_METRIC_REGISTRY.register()
class SGMeanRecall(SceneGraphEvaluation):
    def __init__(self, cfg, result_dict, dataset_name, print_detail=True):
        super(SGMeanRecall, self).__init__(result_dict)
        self.num_rel = cfg.MODEL.ROI_SCENEGRAPH_HEAD.NUM_CLASSES
        self.print_detail = print_detail
        self.rel_name_list = MetadataCatalog.get(dataset_name).predicate_classes # remove __background__

    def register_container(self, mode, cfg,total):
        #self.result_dict[mode + '_recall_hit'] = {20: [0]*self.num_rel, 50: [0]*self.num_rel, 100: [0]*self.num_rel}
        #self.result_dict[mode + '_recall_count'] = {20: [0]*self.num_rel, 50: [0]*self.num_rel, 100: [0]*self.num_rel}
        self.result_dict[mode + '_mean_recall'] = {20: 0.0, 50: 0.0, 100: 0.0}
        self.result_dict[mode + '_mean_recall_collect'] = {20: [[] for i in range(self.num_rel)], 50: [[] for i in range(self.num_rel)], 100: [[] for i in range(self.num_rel)]}
        self.result_dict[mode + '_mean_recall_list'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_mean_recall'].items():
            result_str += '   mR @ %d: %.4f; ' % (k, float(v))
        result_str += ' for mode=%s, type=Mean Recall.' % mode
        result_str += '\n'
        if self.print_detail:
            # import ipdb; ipdb.set_trace()
            result_str += '----------------------- Details ------------------------\n'
            for n, r in zip(self.rel_name_list, self.result_dict[mode + '_mean_recall_list'][100]):
                result_str += '({}:{:.4f}) '.format(str(n), r)
            result_str += '\n'
            result_str += '--------------------------------------------------------\n'

        return result_str

    def collect_mean_recall_items(self, global_container, local_container, mode):
        pred_to_gt = local_container['pred_to_gt']
        gt_rels = local_container['gt_rels']

        for k in self.result_dict[mode + '_mean_recall_collect']:
            # the following code are copied from Neural-MOTIFS
            match = reduce(np.union1d, pred_to_gt[:k])
            # NOTE: by kaihua, calculate Mean Recall for each category independently
            # this metric is proposed by: CVPR 2019 oral paper "Learning to Compose Dynamic Tree Structures for Visual Contexts"
            recall_hit = [0] * self.num_rel
            recall_count = [0] * self.num_rel
            for idx in range(gt_rels.shape[0]):
                local_label = gt_rels[idx,2]
                recall_count[int(local_label)] += 1
                # if self.num_rel != 50:
                # recall_count[0] += 1

            for idx in range(len(match)):
                local_label = gt_rels[int(match[idx]),2]
                recall_hit[int(local_label)] += 1
                # if self.num_rel != 50:
                # recall_hit[0] += 1
            
            for n in range(self.num_rel):
                if recall_count[n] > 0:
                    self.result_dict[mode + '_mean_recall_collect'][k][n].append(float(recall_hit[n] / recall_count[n]))
 

    def calculate_mean_recall(self, mode):
        for k, v in self.result_dict[mode + '_mean_recall'].items():
            sum_recall = 0
            num_rel_no_bg = self.num_rel
            for idx in range(num_rel_no_bg):
                if len(self.result_dict[mode + '_mean_recall_collect'][k][idx]) == 0:
                    tmp_recall = 0.0
                else:
                    tmp_recall = np.mean(self.result_dict[mode + '_mean_recall_collect'][k][idx])
                self.result_dict[mode + '_mean_recall_list'][k].append(tmp_recall)
                sum_recall += tmp_recall

            self.result_dict[mode + '_mean_recall'][k] = sum_recall / float(num_rel_no_bg)
        return



"""
Accumulate Recall:
calculate recall on the whole dataset instead of each image
"""
@SCENEGRAPH_METRIC_REGISTRY.register()
class SGAccumulateRecall(SceneGraphEvaluation):
    def __init__(self, result_dict):
        super(SGAccumulateRecall, self).__init__(result_dict)

    def register_container(self, mode, cfg,total):
        self.result_dict[mode + '_accumulate_recall'] = {20: 0.0, 50: 0.0, 100: 0.0}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_accumulate_recall'].items():
            result_str += '   aR @ %d: %.4f; ' % (k, float(v))
        result_str += ' for mode=%s, type=Accumulate Recall.' % mode
        result_str += '\n'
        return result_str

    def calculate_accumulate(self, mode):
        for k, v in self.result_dict[mode + '_accumulate_recall'].items():
            self.result_dict[mode + '_accumulate_recall'][k] = float(self.result_dict[mode + '_recall_hit'][k][0]) / float(self.result_dict[mode + '_recall_count'][k][0] + 1e-10)

        return 


def _triplet(relations, classes, boxes, predicate_scores=None, class_scores=None):
    """
    format relations of (sub_id, ob_id, pred_label) into triplets of (sub_label, pred_label, ob_label)
    Parameters:
        relations (#rel, 3) : (sub_id, ob_id, pred_label)
        classes (#objs, ) : class labels of objects
        boxes (#objs, 4)
        predicate_scores (#rel, ) : scores for each predicate
        class_scores (#objs, ) : scores for each object
    Returns: 
        triplets (#rel, 3) : (sub_label, pred_label, ob_label)
        triplets_boxes (#rel, 8) array of boxes for the parts
        triplets_scores (#rel, 3) : (sub_score, pred_score, ob_score)
    """
    sub_id, ob_id, pred_label = relations[:, 0], relations[:, 1], relations[:, 2]
    triplets = np.column_stack((classes[sub_id], pred_label, classes[ob_id]))
    triplet_boxes = np.column_stack((boxes[sub_id], boxes[ob_id]))

    triplet_scores = None
    if predicate_scores is not None and class_scores is not None:
        triplet_scores = np.column_stack((
            class_scores[sub_id], predicate_scores, class_scores[ob_id],
        ))

    return triplets, triplet_boxes, triplet_scores

def resize_instance(results, output_height, output_width, mask_threshold=0.5):
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.

    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.

    Args:
        results (Instances): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.

    Returns:
        Instances: the resized output from the model, based on the output resolution
    """

    # Converts integer tensors to float temporaries
    #   to ensure true division is performed when
    #   computing scale_x and scale_y.
    if isinstance(output_width, torch.Tensor):
        output_width_tmp = output_width.float()
    else:
        output_width_tmp = output_width

    if isinstance(output_height, torch.Tensor):
        output_height_tmp = output_height.float()
    else:
        output_height_tmp = output_height

    scale_x, scale_y = (
        output_width_tmp / results.image_size[1],
        output_height_tmp / results.image_size[0],
    )
    results = Instances((output_height, output_width), **results.get_fields())

    if results.has("gt_boxes"):
        output_boxes = results.gt_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(results.image_size)

    results = results[output_boxes.nonempty()]

    if results.has("pred_masks"):
        results.pred_masks = retry_if_cuda_oom(paste_masks_in_image)(
            results.pred_masks[:, 0, :, :],  # N, 1, M, M
            results.pred_boxes,
            results.image_size,
            threshold=mask_threshold,
        )

    if results.has("pred_keypoints"):
        results.pred_keypoints[:, :, 0] *= scale_x
        results.pred_keypoints[:, :, 1] *= scale_y

    return results


def _compute_pred_matches(gt_triplets, pred_triplets,
                 gt_boxes, pred_boxes, iou_thres, phrdet=False):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    Return:
        pred_to_gt [List of List]
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    index_list=[]
    recall_per_class_index=[]
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0],
                                         gt_boxes[gt_has_match],
                                         keeps[gt_has_match],
                                         ):
        boxes = pred_boxes[keep_inds]
        if phrdet:
            # Evaluate where the union box > 0.5
            gt_box_union = gt_box.reshape((2, 4))
            gt_box_union = np.concatenate((gt_box_union.min(0)[:2], gt_box_union.max(0)[2:]), 0)

            box_union = boxes.reshape((-1, 2, 4))
            box_union = np.concatenate((box_union.min(1)[:,:2], box_union.max(1)[:,2:]), 1)

            inds = pairwise_iou(gt_box_union[None], box_union)[0] >= iou_thres

        else:
            #FIXME Check for indexing
            sub_iou = pairwise_iou(Boxes(gt_box[None,:4]), Boxes(boxes[:, :4]))[0]
            obj_iou = pairwise_iou(Boxes(gt_box[None,4:]), Boxes(boxes[:, 4:]))[0]

            inds = ((sub_iou >= iou_thres) & (obj_iou >= iou_thres)).numpy()
        for order,i in enumerate(np.where(keep_inds)[0][inds]):
            if order==0:
                recall_per_class_index.append(i)
            pred_to_gt[i].append(int(gt_ind))
            index_list.append(i)
    index_list=np.array(index_list)
    recall_per_class_index=np.array(recall_per_class_index)
    return pred_to_gt,index_list,recall_per_class_index


def build_scenegraph_evaluators(metrics, cfg, result_dict, dataset_name):
    
    evaluators = {}
    for name in metrics:
        evaluators[name] = SCENEGRAPH_METRIC_REGISTRY.get(name)(cfg, result_dict, dataset_name)

    return evaluators
