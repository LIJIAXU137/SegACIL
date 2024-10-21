"""
SSUL
Copyright (c) 2021-present NAVER Corp.
MIT License
"""

from datetime import timedelta
from tqdm import tqdm
import network

# from trainer.trainer2 import Trainer
from trainer.trainer import Trainer
from trainer.traineracil import Traineracil
# from trainer.trainer_s import Trainer
# from trainer.trainer_bg import Trainer # 不推测背景
# from trainer.trainer_fb import Trainer # 冻结backbone
# from trainer.trainer_pro import Trainer # 使用Prototype和MLP
# from trainer.trainer_cor import Trainer # 使用Correlation

import utils
import os
import time
import random
import numpy as np
import cv2


import torch
import torch.nn as nn
from utils.parser import get_argparser
from utils.utils import AverageMeter
from utils.tasks import get_tasks
from utils.memory import memory_sampling_balanced

import torch.distributed as dist

torch.backends.cudnn.benchmark = True


def main(opts, device):
    # os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
        
    # opts.target_cls = get_tasks(opts.dataset, opts.task, opts.curr_step)
    opts.num_classes = [len(get_tasks(opts.dataset, opts.task, step)) for step in range(opts.curr_step+1)]
    print(opts.num_classes)
    opts.target_cls = [get_tasks(opts.dataset, opts.task, step) for step in range(opts.curr_step+1)]
    
    opts.num_classes = [1, opts.num_classes[0]-1] + opts.num_classes[1:]
    print(opts.num_classes)
    print("==============================================")
    print(f"  task : {opts.task}")
    print(f"  step : {opts.curr_step}")
    print("  Device: %s" % device)
    # print( "  opts : ")
    print(opts)
    print("==============================================")

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)
    
   

    if opts.test_only:
        trainer.do_evaluate(mode='test')
        return
    if opts.method == 'acil':
        trainer = Traineracil(opts, device)   
    else:
        trainer = Trainer(opts, device)
    trainer.train()
    # trainer.prototype_for_train()



# def main2(opts, device,total_step):
#     # os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
        
#     # opts.target_cls = get_tasks(opts.dataset, opts.task, opts.curr_step)
#     opts.num_classes = [len(get_tasks(opts.dataset, opts.task, step)) for step in range(opts.curr_step+1)]
#     print(opts.num_classes)
#     opts.target_cls = [get_tasks(opts.dataset, opts.task, step) for step in range(opts.curr_step+1)]
    
#     print("==============================================")
#     print(f"  task : {opts.task}")
#     print(f"  step : {opts.curr_step}")
#     print("  Device: %s" % device)
#     # print( "  opts : ")
#     print(opts)
#     print("==============================================")

#     # Setup random seed
#     torch.manual_seed(opts.random_seed)
#     np.random.seed(opts.random_seed)
#     random.seed(opts.random_seed)
    
   

#     if opts.test_only:
#         trainer.do_evaluate(mode='test')
#         return
#     if opts.method == 'acil':
#         trainer = Traineracil(opts, device)   
#     else:
#         trainer = Trainer(opts, device)
#     trainer.train()
    # trainer.prototype_for_train()

if __name__ == '__main__':
            
    opts = get_argparser()
    # if torch.cuda.device_count() > 1:
    #     dist.init_process_group("nccl", timeout=timedelta(minutes=120))
    #     rank, world_size = dist.get_rank(), dist.get_world_size()
    #     device_id = rank % torch.cuda.device_count()
    #     device = torch.device(device_id)
    #     opts.local_rank = rank
    #     os.environ['NCCL_BLOCKING_WAIT'] = '0'  # not to enforce timeout
        
    # else:
    print(f"Available GPUs: {torch.cuda.device_count()}")
    opts.local_rank = 0
    # torch.cuda.set_device(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      
        
    start_step = opts.curr_step
    total_step = len(get_tasks(opts.dataset, opts.task))


    # device=torch.device('cuda:1')
    # if device is not None and isinstance(device, torch.device) and device.type == 'cuda':
    #     gpu_id = torch.cuda.current_device()
    #     print(f"Current GPU ID: {gpu_id}")
    # else:
    #     print("Device is not a GPU or not specified.")
    if opts.method=='acil':
        if opts.initial:
            opts.curr_step = 0
            main(opts, device)
        else:
            for step in range(start_step, total_step):
                opts.curr_step = step
                main(opts, device)
    else:
        if opts.initial:
            opts.curr_step = 0
            main(opts, device)
        else:
            for step in range(start_step, total_step):
                opts.curr_step = step
                main(opts, device)
    # main(opts)
    
    # if torch.cuda.device_count() > 1:
    #     dist.destroy_process_group()






