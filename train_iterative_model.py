import sys
import os
import numpy as np
import torch
sys.path.insert(0, '../../')
sys.path.insert(0, '../')

import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer

from SpeaQ.engine import JointTransformerTrainer
from SpeaQ.data import VisualGenomeTrainData, register_datasets, DatasetCatalog, MetadataCatalog
from SpeaQ.configs.defaults import add_dataset_config, add_scenegraph_config
from SpeaQ.modeling import Detr
from detectron2.data.datasets import register_coco_instances
from glob import glob
import pathlib
from shutil import copyfile

parser = default_argument_parser()

def backup_source_codes(cfg):
    if comm.is_main_process():
        output_dir = cfg.OUTPUT_DIR
        source_files = glob('**/*', recursive=True)

        for file in source_files:
            filedir_split = file.split('/')
            if filedir_split[0] == 'wandb':
                continue
            filename_split = file.split('.')
            if len(filename_split) == 1:
                continue
            extension = filename_split[-1]
            if extension == 'pth' or extension == 'pkl':
                continue
            else:
                target_dir = os.path.join(output_dir, 'code_backup', file)
                os.makedirs(os.path.dirname(target_dir), exist_ok=True)
                copyfile(file, target_dir)

def setup(args):
    cfg = get_cfg()
    add_dataset_config(cfg)
    add_scenegraph_config(cfg)
    assert(cfg.MODEL.ROI_SCENEGRAPH_HEAD.MODE in ['predcls', 'sgls', 'sgdet']), "Mode {} not supported".format(cfg.MODEL.ROI_SCENEGRaGraph.MODE)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    register_datasets(cfg)
    # register_coco_data(cfg)
    default_setup(cfg, args)
    
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="LSDA")
    return cfg

def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = JointTransformerTrainer.build_model(cfg)
        # from thop import profile
        # input = torch.randn(1, 3, 800, 1333)
        # macs, params = profile(model, inputs=(input, ))

        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = JointTransformerTrainer.test(cfg, model)
        # if comm.is_main_process():
        #     verify_results(cfg, res)
        return res
    backup_source_codes(cfg)
    trainer = JointTransformerTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

if __name__ == '__main__':
    args = parser.parse_args()
    try:
        # use the last 4 numbers in the job id as the id
        # default_port = os.environ['SLURM_JOB_ID']
        # default_port = default_port[-4:]
        #
        # # all ports should be in the 10k+ range
        # default_port = int(default_port) + 15000
        default_port = args.dist_url
    except Exception:
        default_port = 30050

    args.dist_url = 'tcp://127.0.0.1:'+str(default_port)
    print(args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
