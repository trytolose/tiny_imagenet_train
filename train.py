from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

import torch
import os
import numpy as np

import numpy as np
from PIL import Image

import argparse
from torchvision import models
from torch.nn.modules import loss
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from tqdm import tqdm
from dataset import TinyImagenetDataset
from sklearn.metrics import accuracy_score
from albumentations import Compose, RandomCrop, Normalize, HorizontalFlip, Resize, RandomContrast, RandomGamma, Normalize, Rotate, CenterCrop
from albumentations.pytorch import ToTensorV2
from model import get_model
from utils.losses import LabelSmoothing
from utils.radam import RAdam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from collections import deque
import copy
from utils.cutmixup import cutmix, cutmix_criterion, mixup, mixup_criterion



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
    

def train(args, model, device, train_loader, optimizer, loss_function, epoch, writer):
    model.train()
    model.to(device)
    correct = 0
    
    losses = []
    for images, labels in tqdm(train_loader, ncols=70, leave=False):
        data, target = images.to(device), labels.to(device)

        optimizer.zero_grad()

        if np.random.rand()<args.cutmix:
                data, targets = cutmix(data, target, 1.0)
                output = model(data)
                loss = cutmix_criterion(output, targets, device)
        else: 
            output = model(data)
            loss = loss_function(output, target)

        # if np.random.rand()<0.5:
        #         data, targets = mixup(data, target, 1.0)
        #         output = model(data)
        #         loss = mixup_criterion(output, targets, DEVICE)
        # else: 
        #     output = model(data)
        #     loss = loss_function(output, target)


        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
    train_log = {}
    train_log['loss'] =  np.mean(losses)
    train_log['acc'] = correct / len(train_loader.dataset)
        
    return train_log



def test(model, device, test_loader, loss_function, epoch, writer):

    
    model.eval()
    model.to(device)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        
        for images, labels in tqdm(test_loader, ncols=70, leave=False):
            data, target = images.to(device), labels.type(torch.LongTensor).to(device)
            
            output = model(data)
    
            test_loss += loss_function(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)

    val_log = {}
    val_log['loss'] =  test_loss
    val_log['acc'] = correct / len(test_loader.dataset)
    
    return val_log




def main():

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--epoch', type=int, default=50, help='total epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--gpu_n', type=str, default="0", help='gpu cuda number')
    parser.add_argument('--exp_name', type=str, default='test', help='experiment name')
    parser.add_argument('--model_name', type=str, default='resnet18', help='model name')
    parser.add_argument('--optim', type=str, default='adam', help='optimizer')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--loss_func', type=str, default='cross', help='loss function')
    parser.add_argument('--scheduler', type=str, default='reduce', help='lr scheduler')
    parser.add_argument('--patience', type=int, default=6, help='patience for reduceOnPleato')
    parser.add_argument('--lr_factor', type=float, default=0.1, help='lr decrease factor')
    parser.add_argument('--t_max', type=int, default=50, help='total epochs for cosine annealing')
    parser.add_argument('--lr_min', type=float, default=0, help='min lr')
    parser.add_argument('--save_queue', type=int, default=1, help='save or not 5 best checkpoints')
    parser.add_argument('--auto_aug', type=int, default=0, help='use or not autoaugmentaion')
    parser.add_argument('--cutmix', type=float, default=0, help='use or not cutmix')
        
    args = parser.parse_args()


    train_transform = Compose([
    RandomCrop(56, 56),
    HorizontalFlip(),
    Rotate(limit=(-20,20)),
    RandomContrast(),
    RandomGamma(gamma_limit=(90, 110)),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        ),
    ToTensorV2(),
    ])
    val_transform = Compose([
        CenterCrop(56, 56),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            ),
        ToTensorV2() 
    ])

 
    DATA_ROOT = Path("../input/tiny-imagenet-200")
    RESULTS_ROOT = Path("results/tensorboard_logs")
    TENSORBOARD_TAG = args.exp_name
    DEVICE = torch.device(f"cuda:{args.gpu_n}")
    
    
    
    model = get_model(args)
    model = model.to(DEVICE)

    if args.optim == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == "radam":
        optimizer = RAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    if args.scheduler == "reduce":
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_factor, patience=args.patience)
    elif args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.t_max, eta_min=args.lr_min)

    if args.loss_func == "cross":
        loss_function = loss.CrossEntropyLoss()
    elif args.loss_func == "smooth":
        loss_function = LabelSmoothing(smoothing=0.1)


    if args.auto_aug == 1:
        train_transform = None

    writer = SummaryWriter(RESULTS_ROOT / TENSORBOARD_TAG)

    summary = ''
    for k, v in args.__dict__.items():
        summary += f'{k}: {v}; \n'
    writer.add_text('exp params', summary)


    train_dataset = TinyImagenetDataset(DATA_ROOT / "train", train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=10,
    )

    test_dataset = TinyImagenetDataset(DATA_ROOT / "val" / "images", val_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=10,
    )



    best_models = deque(maxlen=5)
    best_acc = -1
    for epoch in range(args.epoch):
        

        train_log = train(args, model, DEVICE, train_loader, optimizer, loss_function, epoch, writer)
        val_log = test(model, DEVICE, test_loader, loss_function, epoch, writer)
        
        lr = get_lr(optimizer)
        
        print(f"epoch: {epoch:03d} lr: {lr:.05f} train_loss: {train_log['loss']:.04f} val_loss: {val_log['loss']:.04f} acc: {val_log['acc']:.04f}", end=' ')
    
        writer.add_scalar("Loss/train", train_log['loss'], global_step=epoch)
        writer.add_scalar("Acc/train",    train_log['acc'],  global_step=epoch)
        writer.add_scalar("Loss/valid", val_log['loss'], global_step=epoch)
        writer.add_scalar("Acc/valid",    val_log['acc'],  global_step=epoch)
        writer.add_scalar("LR", lr,  global_step=epoch)

        if isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        else:
            scheduler.step(val_log['acc'])
        
        writer.flush()

        if val_log['acc'] > best_acc:
            best_acc = val_log['acc']
            print("+")
            state_dict = copy.deepcopy(model.state_dict()) 

            if args.save_queue==1:
                best_models.append(state_dict)
                for i, m in enumerate(best_models):
                    path = f"result/checkpoints/{args.exp_name}"
                    os.makedirs(path, exist_ok=True)
                    torch.save(m, os.path.join(path, f"{i}.pt"))

            torch.save(state_dict, f"result/checkpoints/{args.exp_name}.pt")
        else:
            print()


if __name__ == '__main__':
    main()

