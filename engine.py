import math
import sys
from typing import Iterable

import torch
from utils.Tools import *

cons_lambda = 5

def cal_loss(criterion, raw_img_out, mask_img_out, target):
    loss1 = criterion(raw_img_out, target)
    loss2 = criterion(mask_img_out, target)
    loss_consistency = criterion( raw_img_out, mask_img_out)
    loss = ( loss1 +  loss2) + cons_lambda * loss_consistency
    return loss

def cal_ranking_loss(criterion, raw_img_out, mask_img_out, target, out_nograd):
    loss0 = criterion(raw_img_out, target)
    loss1 = criterion(mask_img_out, target)
    loss2 = criterion(raw_img_out, out_nograd)
    loss = ( loss0 +  loss1) + cons_lambda * loss2
    return loss

def train_one_epoch(model, data_loader, optimizer, device, epoch, criterion, args):
    
    model.train(True)

    losses = AverageMeter()
    masked_MAE = AverageMeter()
    raw_MAE = AverageMeter()

    model.train()
    for idx, (img, img_mask, sid, target, male) in enumerate(data_loader):

        # =========== convert male lable to one hot type =========== #
        img = img.to(device)
        img_mask = img_mask.to(device)
        
        target = target.type(torch.FloatTensor).to(device)

        male = torch.unsqueeze(male, 1)
        male = torch.zeros(male.shape[0], 2).scatter_(1, male, 1)
        male = male.to(device).type(torch.FloatTensor)

        # =========== compute output and loss =========== #
        model.zero_grad()
        
        raw_img_out = model(img,male)


        # mask部分
        mask_img_out = model(img_mask,male)
        out_nograd = mask_img_out.detach()
        out_nograd = out_nograd.to(device)

        loss = cal_loss(criterion, raw_img_out, mask_img_out, target)


        
        # masked_mae = metric(mask_img_out.detach(), target.detach().cpu())
        raw_mae = metric(raw_img_out.detach(), target.detach().cpu())
        losses.update(loss, img.size(0))

        # masked_MAE.update(masked_mae, img.size(0))
        raw_MAE.update(raw_mae, img.size(0))

        if idx % args.print_freq == 0:
            print(
                  'Epoch: [{0} / {1}]   [step {2}/{3}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                #   'mask MAE {mMAE.val:.3f} ({mMAE.avg:.3f})\t'
                  'raw MAE {rMAE.val:.3f}  ({rMAE.avg:.3f})\t'.format
                  ( epoch, args.epochs, idx, len(data_loader)
                  , loss=losses
                #   , mMAE=masked_MAE
                  , rMAE=raw_MAE)
                  )
        loss.backward()
        if args.accumulation_steps > 1:
            if ((idx + 1) % args.accumulation_steps) == 0:
                optimizer.step()
                optimizer.zero_grad()
                
        else:
            optimizer.step()
        
            
    return {'loss':losses.avg, 'mae':masked_MAE.avg}

def validate_one_epoch(model, data_loader, criterion, device, args):
    '''
    For validation process
    
    Args:
        valid_loader (data loader): validation data loader.
        model (CNN model): convolutional neural network.
        criterion1 (loss fucntion): main loss function.
        criterion2 (loss fucntion): aux loss function.
        device (torch device type): default: GPU
    Returns:
        [float]: training loss average and MAE average
    '''
    losses = AverageMeter()
    MAE = AverageMeter()

    # =========== switch to evaluate mode ===========#
    model.eval()

    with torch.no_grad():
        for _, (img, img_mask, sid, target, male) in enumerate(data_loader):
            img = img.to(device)

            
            target = target.type(torch.FloatTensor).to(device)
            male = torch.unsqueeze(male, 1)
            male = torch.zeros(male.shape[0], 2).scatter_(1, male, 1)
            male = male.to(device).type(torch.FloatTensor)

            # =========== compute output and loss =========== #
            out = model(img,male)

            # =========== compute loss =========== #
            loss = criterion(out, target)

            mae = metric(out.detach(), target.detach().cpu())

            # =========== measure accuracy and record loss =========== #
            losses.update(loss, img.size(0))
            MAE.update(mae, img.size(0))
        print(
                'Valid: [steps {0}], Loss {loss.avg:.4f},  MAE:  {MAE.avg:.4f}'.format(
                len(data_loader), loss=losses, MAE=MAE))

        return {'loss':losses.avg, 'mae':MAE.avg}