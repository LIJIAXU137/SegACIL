from tqdm import tqdm
from datasets import *
import network
import utils
import os
import time
import random
import argparse
import numpy as np
import cv2
from collections import Counter
from torch.utils import data
from utils import ext_transforms as et
from metrics import StreamSegMetrics
from utils.Buffer import RandomBuffer, GaussianKernel
from utils.AnalyticLinear import GeneralizedARM, RecursiveLinear
import torch
import torch.nn as nn
from utils.ckpt import save_ckpt, load_ckpt
from utils.utils import AverageMeter
from utils.tasks import get_tasks
from utils.memory import memory_sampling_balanced
from utils import *
from network import get_modelmap

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import json
from utils.scheduler import build_scheduler
from utils.loss import build_criterion
from utils.logger import Logger
import torch.nn.functional as F

class AIR(nn.Module):
    def __init__(self, backbone_output, backbone, buffer_size, gamma, device=None, dtype=torch.double, linear=RecursiveLinear,learned_classes=None):
        super(AIR, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.backbone = backbone
        self.backbone_output = backbone_output
        self.buffer_size = buffer_size
        self.buffer = RandomBuffer(backbone_output, buffer_size, **factory_kwargs)
        self.analytic_linear = linear(buffer_size, gamma, **factory_kwargs)
        self.H = 0
        self.W = 0
        self.channle = 0
        self.B = 0

        self.eval()
    
    @torch.no_grad()
    def feature_expansion(self, X):
        input=X
        X, _ = self.backbone(X)
  
        X = F.interpolate(X, input.shape[-2:], mode='bilinear', align_corners=False)
        self.B,self.channle,self.H,self.W = X.shape
        X = X.view(self.B,self.channle,-1).permute(0,2,1) # B, H*W, C

        return self.buffer(X)# B, H*W, C-> B, H*W, buffer_size
    @torch.no_grad()
    def forward(self, X):
        return self.analytic_linear(self.feature_expansion(X))
    
    @torch.no_grad()
    def fit(self, X, y, *args, **kwargs):
        X = self.feature_expansion(X)
        
        y = y.unsqueeze(1)
        # y = y.float()
        # 使用最近邻插值代替双线性插值
        # y = F.interpolate(y, size=(self.H, self.W), mode='nearest')
        y = y.long()
        self.analytic_linear.fit(X, y)

    @torch.no_grad()
    def update(self):
        self.analytic_linear.update()




class Traineracil(object):
    def __init__(self, opts, device) -> None:
        super(Traineracil, self).__init__()
        self.opts = opts
        self.device = device
        self.model_name = opts.model
        self.num_classes = opts.num_classes
        self.output_stride = opts.output_stride
        self.bn_freeze = opts.bn_freeze if opts.curr_step > 0 else False
        self.separable_conv = opts.separable_conv
        self.curr_step = opts.curr_step
        self.lr = opts.lr
        self.weight_decay = opts.weight_decay
        self.overlap = opts.overlap
        self.dataset = opts.dataset
        self.task = opts.task
        self.pseudo = opts.pseudo
        self.pseudo_thresh = opts.pseudo_thresh
        self.loss_type = opts.loss_type
        self.amp = opts.amp
        self.batch_size = opts.batch_size
        self.ckpt = opts.ckpt
        self.train_epoch = opts.train_epoch
        self.local_rank = opts.local_rank
        self.buffer= opts.buffer
        self.gamma = opts.gamma
        self.setting=opts.setting
        self.subpath = opts.subpath
        self.curr_step = opts.curr_step
        
        self.opts=opts
        self.curr_idx = [
            sum(len(get_tasks(opts.dataset, opts.task, step)) for step in range(opts.curr_step)), 
            sum(len(get_tasks(opts.dataset, opts.task, step)) for step in range(opts.curr_step+1))
        ]
        
        self.init_models()
        # Set up metrics
        self.metrics = StreamSegMetrics(opts.num_classes, dataset=opts.dataset)
        if self.setting=='sequential':
            scheme="sequential"
        elif self.setting=='disjoint':
            scheme="disjoint"
        elif self.setting=='overlap':
            scheme="overlap"
        self.scheme=scheme
        root_path = f"checkpoints/{opts.subpath}/{self.task}/{scheme}/step{opts.curr_step}/"
        if self.local_rank == 0:
            utils.mkdir(root_path)
        root_path_prev = f"checkpoints/{opts.subpath}/{self.task}/{scheme}/step{opts.curr_step-1}/"
        self.ckpt_str = f"{root_path}%s_%s_%s_step_%d_{scheme}.pth"
        self.ckpt_str_prev = f"{root_path_prev}%s_%s_%s_step_%d_{scheme}.pth"
        self.root_path = root_path
        self.root_path_prev = root_path_prev
        
        self.init_ckpt()
        self.train_loader, self.val_loader, self.test_loader, self.memory_loader = init_dataloader(opts)
        
        self.init_iters(opts)

        self.scheduler = build_scheduler(opts, self.optimizer, self.total_itrs)
        self.criterion = build_criterion(opts)
        self.avg_loss = AverageMeter()
        self.avg_time = AverageMeter()
        self.avg_loss_std = AverageMeter()
        self.aux_loss_1 = AverageMeter()
        self.aux_loss_2 = AverageMeter()
        self.aux_loss_3 = AverageMeter()
        self.aux_loss_4 = AverageMeter()
        self.logger = Logger(root_path)

        # self.kl_loss = nn.KLDivLoss(reduction='none')
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def init_models(self):
        # Set up model
        model_map = get_modelmap()

        if self.local_rank==0:
            print(f"Category components: {self.num_classes}")
        self.model = model_map[self.model_name](num_classes=self.num_classes, output_stride=self.output_stride, bn_freeze=self.bn_freeze)
        if self.separable_conv and 'plus' in self.model_name:
            network.convert_to_separable_conv(self.model.classifier)
        utils.set_bn_momentum(self.model.backbone, momentum=0.01)
        # if torch.cuda.device_count() > 1:
        #     self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        #     if self.bn_freeze:
        #         self.model.freeze_bn()
        #         self.model.freeze_dropout()
            
        if self.curr_step > 0:
            """ load previous model """
            self.model_prev = model_map[self.model_name](num_classes=self.num_classes[:-1], output_stride=self.output_stride, bn_freeze=self.bn_freeze)
            if self.separable_conv and 'plus' in self.model_name:
                network.convert_to_separable_conv(self.model_prev.classifier)
            utils.set_bn_momentum(self.model_prev.backbone, momentum=0.01)
            self.model_prev = nn.SyncBatchNorm.convert_sync_batchnorm(self.model_prev)
            self.model_prev.freeze_bn()
            self.model_prev.freeze_dropout()
        else:
            self.model_prev = None

        self.optimizer = self.init_optimizer()

        self.model = self.model.to(self.device)

        self.model.train()
        
        if self.curr_step > 0:
            self.model_prev = self.model_prev.to(self.device)
            self.model_prev.eval()
            for param in self.model_prev.parameters():
                param.requires_grad = False



    def init_ckpt(self):
        if self.curr_step > 0: # previous step checkpoint
            self.ckpt = self.ckpt_str_prev % (self.model_name, self.dataset, self.task, self.curr_step-1)
        else:
            return
        if self.curr_step > 1:
            self.ckpt = self.root_path_prev + "final.pth"

        print(self.ckpt)
        assert os.path.isfile(self.ckpt)
        # checkpoint = torch.load(self.ckpt, map_location=torch.device('cpu'))["model_state"]
        # self.model_prev.load_state_dict(checkpoint, strict=True)



        # self.model.load_state_dict(checkpoint, strict=False)

        print("Model restored from %s" % self.ckpt)
        # del checkpoint  # free memory


        
    def init_optimizer(self):
        # Set up optimizer & parameters
  
        training_params = [{'params': self.model.backbone.parameters(), 'lr': 0.001},
                        {'params': self.model.classifier.parameters(), 'lr': 0.01}]
        optimizer = torch.optim.SGD(params=training_params, 
                                    lr=self.lr, 
                                    momentum=0.9, 
                                    weight_decay=self.weight_decay,
                                    nesterov=True)
        
        # if self.local_rank == 0:
        #     print("----------- trainable parameters --------------")
        #     for name, param in self.model.named_parameters():
        #         if param.requires_grad:
        #             print(name, param.shape)
        #             pass
        #     print("-----------------------------------------------")

        return optimizer
    

    
    def init_iters(self, opts):
         # Restore
        self.best_score = -1
        
        self.total_itrs = self.train_epoch * len(self.train_loader)
        self.val_interval = max(100, self.total_itrs // 100)
        print(f"... train epoch : {self.train_epoch} , iterations : {self.total_itrs} , val_interval : {self.val_interval}")
                
    def train(self):
        scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        # =====  Train  =====
        if self.curr_step == 0:
            for epoch in range(self.train_epoch):
                self.model.train()
                for seq, (images, labels, _) in enumerate(self.train_loader):
                    images = images.to(self.device, dtype=torch.float32, non_blocking=True)
                    labels = labels.to(self.device, dtype=torch.long, non_blocking=True)
                    
                    outputs, loss = self.train_episode(images, labels, scaler, epoch)
                    
                    if self.local_rank==0 and seq % 10 == 0:
                        print("[%s / step %d] Epoch %d, Itrs %d/%d, Loss=%4f, StdLoss=%.4f, A1=%.4f, A2=%.4f, A3=%.4f, A4=%.4f Time=%.2f , LR=%.8f" %
                            (self.task, self.curr_step, epoch, seq, len(self.train_loader), 
                            self.avg_loss.avg, self.avg_loss_std.avg, self.aux_loss_1.avg, self.aux_loss_2.avg, self.aux_loss_3.avg, self.aux_loss_4.avg, self.avg_time.avg*1000, self.optimizer.param_groups[0]['lr']))
                        self.logger.write_loss(self.avg_loss.avg, epoch * len(self.train_loader) + seq + 1)

                if self.local_rank == 0 and (len(self.train_loader) > 100 or epoch % 5 ==4):
                    print("[Validation]")
                    val_score = self.validate()
                    print(self.metrics.to_str_val(val_score))
                    
                    class_iou = list(val_score['Class IoU'].values())
                    val_score = np.mean( class_iou[self.curr_idx[0]:self.curr_idx[1]] + [class_iou[0]])
                    curr_score = np.mean( class_iou[self.curr_idx[0]:self.curr_idx[1]] )
                    print("curr_val_score : %.4f\n" % (curr_score))
                    self.logger.write_score(curr_score, epoch)
                    
                    if curr_score > self.best_score:  # save best model
                        print("... save best ckpt : ", curr_score)
                        self.best_score = curr_score
                        save_ckpt(self.ckpt_str % (self.model_name, self.dataset, self.task, self.curr_step), self.model, self.optimizer, self.best_score)

        elif self.curr_step == 1:
            self.opts.curr_step=0
            self.train_loader0, self.val_loader0, self.test_loader0, self.memory_loader0 = init_dataloader(self.opts)
            #re-alignment
            self.model=load_ckpt(self.ckpt)[0]
            self.model = self.model.to(self.device)
            self.make_model()
          
            self.incremental_learning(self.train_loader0)
            self.root_path0 = f"checkpoints/{self.subpath}/{self.task}/{self.scheme}/step0/"
            save_ckpt(self.root_path0+"final.pth", self.model, self.optimizer, self.best_score)
            self.do_evaluate3(mode='test')
            ##CIL
            self.incremental_learning(self.train_loader)
   

        elif self.curr_step > 1:
            self.model=load_ckpt(self.ckpt)[0]
            self.incremental_learning(self.train_loader)

        if self.local_rank == 0 :
            save_ckpt(self.root_path+"final.pth", self.model, self.optimizer, self.best_score)
            print("... Training Done")
            if self.curr_step > 0:
                self.do_evaluate2(mode='test')



    def train_episode(self, images, labels, scaler, epoch):
        self.optimizer.zero_grad()
        end_time = time.time()
        """ forwarding and optimization """
        with torch.cuda.amp.autocast(enabled=self.amp):

            outputs, features = self.model(images)

            if self.pseudo and self.curr_step > 0:
                
                pass

                
            else:
                # print("labels : ", labels.shape)
                #打印labels的标签有哪些以及统计具体数量，比如说有多少个0，多少个1
                # 统计每个标签的数量
                # unique_labels, counts = torch.unique(labels, return_counts=True)

                # 打印每个标签及其数量
                # for label, count in zip(unique_labels, counts):
                #     print(f"标签 {label} 的数量: {count}")

          
                outputs = F.interpolate(outputs, labels.shape[-2:], mode='bilinear', align_corners=False)
           
                loss = std_loss = self.criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                self.scheduler.step()
                self.avg_loss.update(loss.item())
                self.avg_time.update(time.time() - end_time)
                self.avg_loss_std.update(std_loss.item())



        return outputs, loss
    
    def do_evaluate(self, mode='val'):
        print("[Testing Best Model]")
        # best_ckpt = self.ckpt_str % (self.model_name, self.dataset, self.task, self.curr_step)
        best_ckpt = self.root_path+"final.pth"
        
        checkpoint = torch.load(best_ckpt, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        self.model.eval()
        
        test_score = self.validate(mode)
        print(self.metrics.to_str(test_score))

        class_iou = list(test_score['Class IoU'].values())
        class_acc = list(test_score['Class Acc'].values())
        first_cls = len(get_tasks(self.dataset, self.task, 0))

        print(f"...from 0 to {first_cls-1} : best/test_before_mIoU : %.6f" % np.mean(class_iou[:first_cls]))
        print(f"...from {first_cls} to {len(class_iou)-1} best/test_after_mIoU : %.6f" % np.mean(class_iou[first_cls:]))
        print(f"...from 0 to {first_cls-1} : best/test_before_acc : %.6f" % np.mean(class_acc[:first_cls]))
        print(f"...from {first_cls} to {len(class_iou)-1} best/test_after_acc : %.6f" % np.mean(class_acc[first_cls:]))

        test_score[f'0 to {first_cls-1} mIoU'] = np.mean(class_iou[:first_cls])
        test_score[f'{first_cls} to {len(class_iou)-1} mIoU'] = np.mean(class_iou[first_cls:])
        test_score[f'0 to {first_cls-1} mAcc'] = np.mean(class_acc[:first_cls])
        test_score[f'{first_cls} to {len(class_iou)-1} mAcc'] = np.mean(class_acc[first_cls:])

        # save results as json
        with open(f"{self.root_path}/test_results.json", 'w') as f:
            f.write(json.dumps(test_score, indent=4))
            f.close()

    def do_evaluate2(self, mode='val'):
        print("[Testing Best Model]")
        # best_ckpt = self.ckpt_str % (self.model_name, self.dataset, self.task, self.curr_step)
        best_ckpt = self.root_path+"final.pth"
        
        self.model=load_ckpt(best_ckpt)[0]
        self.model.eval()
        
        test_score = self.validate2(mode)
        print(self.metrics.to_str(test_score))

        class_iou = list(test_score['Class IoU'].values())
        class_acc = list(test_score['Class Acc'].values())
        first_cls = len(get_tasks(self.dataset, self.task, 0))

        print(f"...from 0 to {first_cls-1} : best/test_before_mIoU : %.6f" % np.mean(class_iou[:first_cls]))
        print(f"...from {first_cls} to {len(class_iou)-1} best/test_after_mIoU : %.6f" % np.mean(class_iou[first_cls:]))
        print(f"...from 0 to {first_cls-1} : best/test_before_acc : %.6f" % np.mean(class_acc[:first_cls]))
        print(f"...from {first_cls} to {len(class_iou)-1} best/test_after_acc : %.6f" % np.mean(class_acc[first_cls:]))

        test_score[f'0 to {first_cls-1} mIoU'] = np.mean(class_iou[:first_cls])
        test_score[f'{first_cls} to {len(class_iou)-1} mIoU'] = np.mean(class_iou[first_cls:])
        test_score[f'0 to {first_cls-1} mAcc'] = np.mean(class_acc[:first_cls])
        test_score[f'{first_cls} to {len(class_iou)-1} mAcc'] = np.mean(class_acc[first_cls:])

        # save results as json
        with open(f"{self.root_path}/test_results.json", 'w') as f:
            f.write(json.dumps(test_score, indent=4))
            f.close()

    def do_evaluate3(self, mode='val'):
        print("[Testing Best Model]")
        # best_ckpt = self.ckpt_str % (self.model_name, self.dataset, self.task, self.curr_step)
        best_ckpt = self.root_path0+"final.pth"
        
        self.model=load_ckpt(best_ckpt)[0]
        self.model.eval()
        
        test_score = self.validate2(mode)
        print(self.metrics.to_str(test_score))

        class_iou = list(test_score['Class IoU'].values())
        class_acc = list(test_score['Class Acc'].values())
        first_cls = len(get_tasks(self.dataset, self.task, 0))

        print(f"...from 0 to {first_cls-1} : best/test_before_mIoU : %.6f" % np.mean(class_iou[:first_cls]))
        print(f"...from {first_cls} to {len(class_iou)-1} best/test_after_mIoU : %.6f" % np.mean(class_iou[first_cls:]))
        print(f"...from 0 to {first_cls-1} : best/test_before_acc : %.6f" % np.mean(class_acc[:first_cls]))
        print(f"...from {first_cls} to {len(class_iou)-1} best/test_after_acc : %.6f" % np.mean(class_acc[first_cls:]))

        test_score[f'0 to {first_cls-1} mIoU'] = np.mean(class_iou[:first_cls])
        test_score[f'{first_cls} to {len(class_iou)-1} mIoU'] = np.mean(class_iou[first_cls:])
        test_score[f'0 to {first_cls-1} mAcc'] = np.mean(class_acc[:first_cls])
        test_score[f'{first_cls} to {len(class_iou)-1} mAcc'] = np.mean(class_acc[first_cls:])

        # save results as json
        with open(f"{self.root_path}/test_results.json", 'w') as f:
            f.write(json.dumps(test_score, indent=4))
            f.close()



    def validate2(self, mode='val'):
        """Do validation and return specified samples"""
        self.metrics.reset()
        ret_samples = []
        self.model.eval()

        with torch.no_grad():
            for i, (images, labels, _) in enumerate(tqdm(self.val_loader if mode=='val' else self.test_loader)):
                
                images = images.to(self.device, dtype=torch.float32, non_blocking=True)
                labels = labels.to(self.device, dtype=torch.long, non_blocking=True)
                
                outputs= self.model(images)
                # print("outputs.shape",outputs.shape)

                if self.loss_type == 'bce_loss':
                    outputs = torch.sigmoid(outputs)
                else:
                    outputs = torch.softmax(outputs, dim=1)
                outputs=outputs.permute(0,3,1,2)
                # outputs = F.interpolate(outputs, labels.shape[-2:], mode='bilinear', align_corners=False)

                preds = outputs.detach().max(dim=1)[1].cpu().numpy()
                targets = labels.cpu().numpy()
                

                

                
                self.metrics.update(targets, preds)
                    
            score = self.metrics.get_results()
        return score
    


    def validate(self, mode='val'):
        """Do validation and return specified samples"""
        self.metrics.reset()
        ret_samples = []
        self.model.eval()

        with torch.no_grad():
            for i, (images, labels, _) in enumerate(tqdm(self.val_loader if mode=='val' else self.test_loader)):
                
                images = images.to(self.device, dtype=torch.float32, non_blocking=True)
                labels = labels.to(self.device, dtype=torch.long, non_blocking=True)
                
                outputs, _ = self.model(images)
                
                if self.loss_type == 'bce_loss':
                    outputs = torch.sigmoid(outputs)
                else:
                    outputs = torch.softmax(outputs, dim=1)

                outputs = F.interpolate(outputs, labels.shape[-2:], mode='bilinear', align_corners=False)
                
                preds = outputs.detach().max(dim=1)[1].cpu().numpy()
                targets = labels.cpu().numpy()
                self.metrics.update(targets, preds)
                    
            score = self.metrics.get_results()
        return score


    def make_model(self):
        # Extract backbone (input_norm and encoder) from previous model
        self.model.classifier.head = nn.Identity()

        backbone = self.model
        self.air_model = AIR(
            backbone_output=256,
            backbone=backbone,
            buffer_size=self.buffer,
            gamma=self.gamma,
            device=self.device,
            dtype=torch.double,
            linear=RecursiveLinear,
           
        )
        self.air_model.to(self.device)
        self.model= self.air_model  # Overwrite self.model with ACIL model
        # print(self.model)


    def incremental_learning(self,train_loader):
        self.model.eval()
        for seq, (X, y, _) in enumerate(train_loader):
            # 获取 y 中每个标签的值以及对应的个数
            unique_labels, label_counts = torch.unique(y, return_counts=True)

            X, y = X.to(self.device), y.to(self.device)
            self.model.fit(X, y)
        self.model.update()