import os
import torch
import nibabel as nib
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torchio as tio 
from utils.Crop_and_padding import resize_image_with_crop_or_pad
from utils.Interger_Multiple_Batch_Size import Integer_Multiple_Batch_Size
from utils.cube_mask import cube_mask


def nii_loader(path):
    img = nib.load(str(path))
    data = img.get_fdata()
    return data

def read_table(path):
    return(pd.read_excel(path).values) # default to first sheet

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

class IMG_Folder(torch.utils.data.Dataset):
    def __init__(self,excel_path, data_path, loader=nii_loader,transforms=None):
        self.root = data_path
        self.sub_fns = sorted(os.listdir(self.root))
        self.table_refer = read_table(excel_path)
        self.loader = loader
        self.transform = transforms

    def __len__(self):
        return len(self.sub_fns)

    def __getitem__(self,index):
        sub_fn = self.sub_fns[index]
        for f in self.table_refer:

            sid = str(f[0])
            slabel = (int(f[1]))
            smale = f[2]
            if sid not in sub_fn:
                continue
            sub_path = os.path.join(self.root, sub_fn)
            img = self.loader(sub_path)
            img = white0(img)

            img = np.expand_dims(img, axis=0)
            img = np.ascontiguousarray(img, dtype= np.float32)
            img = torch.from_numpy(img).type(torch.FloatTensor)

            if self.transform is not None:
                img = self.transform(img)
            break
        slabel = torch.tensor(np.array(slabel))
        slabel = torch.unsqueeze(slabel, dim=0)
        return (img, sid, slabel, smale)



class IMG_Folder_with_mask(torch.utils.data.Dataset):
    def __init__(self,excel_path, data_path, loader=nii_loader,transforms=None, mask_type='cube_mask'):
        self.root = data_path
        self.mask_type = mask_type

        self.sub_fns_img  = sorted(os.listdir(data_path))

        self.table_refer = read_table(excel_path)
        self.loader = loader
        self.transform = transforms

    def __len__(self):
        return len(self.sub_fns_img)

    def __getitem__(self,index):

        sub_fn_img = self.sub_fns_img[index]

        for f in self.table_refer:
            
            sid = str(f[0])
            t_id = str(f[0])
            slabel = (int(f[1]))
            smale = f[2]
            if sid not in sub_fn_img:
                continue

            sub_img_path = os.path.join(self.root,sub_fn_img)

            img = self.loader(sub_img_path)


            img = white0(img)

            img = np.expand_dims(img, axis=0)
            img = np.ascontiguousarray(img, dtype= np.float32)
            img = torch.from_numpy(img).type(torch.FloatTensor)

            masked_img = tio.Lambda(cube_mask)(img).type(torch.FloatTensor)


            if self.transform is not None:
                img = self.transform(img)
                masked_img = self.transform(masked_img)
            break
        slabel = torch.tensor(np.array(slabel))
        slabel = torch.unsqueeze(slabel, dim=0)
        return (img, masked_img, sid, slabel, smale)


# if __name__ == "__main__":
#     excel_path ="/data/zhaojinxin/brain_prediction/opt/zhaojinxin/TSAN/18_combine.xls"
#     train_folder = "/data/zhaojinxin/brain_prediction/opt/zhaojinxin/TSAN/brain_age_estimation_transfer_learning/train/"  
#     test_folder = "/data/zhaojinxin/brain_prediction/opt/zhaojinxin/TSAN/brain_age_estimation_transfer_learning/test/" 


#     val_dataset = IMG_Folder_with_mask(excel_path, test_folder
#                                          )

#     val_dataset = Integer_Multiple_Batch_Size(val_dataset, batch_size=8)

#     val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False,
#                               drop_last=False, num_workers=8)

#     test_data = IMG_Folder(excel_path, test_folder)
#     test_loader2 = torch.utils.data.DataLoader(test_data
#                                                , batch_size=8
#                                                , num_workers=1
#                                                , pin_memory=True
#                                                , drop_last=True
#                                                )
#     brk=0
#     for idx, (img,masked_img ,sid, target, male) in enumerate(val_loader):
#         print(img.shape)
#         print(masked_img.shape)
#         break

  