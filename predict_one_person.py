import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from DataSet import IMG_Folder, IMG_Folder_with_mask, Integer_Multiple_Batch_Size
from network.ScaleDense import ScaleDense
from utils.Tools import *
from loss.Matrix_loss import Matrix_distance_L2_loss, Matrix_distance_L3_loss, Matrix_distance_loss
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import nibabel as nib
#1 是男的，0 是女的
model_name = "/data/zhaojinxin/brain_prediction/super_cal/ckpt-best.pth.tar"
model = ScaleDense()
model.load_state_dict(torch.load(model_name)['state_dict'])

# male = torch.unsqueeze(male, 1)
#  male = torch.zeros(male.shape[0], 2).scatter_(1, male, 1).type(torch.FloatTensor)


def white0(image, threshold=0):
    "standardize voxels with value > threshold"
    image = image.astype(np.float32)
    mask = (image > threshold).astype(int)

    image_h = image * mask
    image_l = image * (1 - mask)

    mean = np.sum(image_h) / np.sum(mask)
    std = np.sqrt(np.sum(np.abs(image_h - mean)**2*mask) / np.sum(mask))

    if std > 0:
        ret = (image_h - mean) / std + image_l
    else:
        ret = image * 0.
    return ret

def nii_loader(path):
    img = nib.load(str(path))
    data = img.get_data()
    return data

img = nii_loader("/data/zhaojinxin/brain_prediction/bawd/pubulish/data/zhaojinxin/brain_prediction/bawd/Brain-Age-With-Disease-master/personal_img/WANG^AN---zz-test-20241217---20241217122617---2---0-MYMY-3D_Sag_T1_MP-RAGE.nii.gz")
img = white0(img)
img = np.expand_dims(img, axis=0)
img = np.ascontiguousarray(img, dtype= np.float32)
img = torch.from_numpy(img).type(torch.FloatTensor)
img = img.unsqueeze(0)
male = torch.tensor([1])
male = torch.unsqueeze(male, 1)
male = torch.zeros(male.shape[0], 2).scatter_(1, male, 1).type(torch.FloatTensor)

out = model(img, male)
print("MALE: ", out)


male = torch.tensor([0])
male = torch.unsqueeze(male, 1)
male = torch.zeros(male.shape[0], 2).scatter_(1, male, 1).type(torch.FloatTensor)

out = model(img, male)
print("FEMALE: ",out)

