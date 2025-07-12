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


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Set Default Parameter
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
cudnn.enabled = True
cudnn.benchmark = True
import matplotlib.pyplot as plt


def plot_img(x,y,file_dir, Mae, pearson):
    plt.scatter(x, y, s=10, c=[(0,0.5,0.5)], linewidths=1, alpha=0.5, marker='o')
    plt.xticks(np.arange(15, 100, step=20))

    plt.yticks(np.arange(15, 100, step=20))
    plt.xlim(15,100)
    plt.ylim(15,100)
    plt.xlabel('Chronological age',fontdict={'size':18})
    plt.ylabel('Estimated age',fontdict={'size':18})

    x1 = np.linspace(15,100)
    y0 = x1
    y1 = x1 + 10
    y11 = x1 - 10
    y2 = x1 + 20
    y22 = x1 - 20


    Mae = round(Mae, 3)
    pearson = round(pearson, 3)
    MAE = 'MAE: '+"{:.3f}".format(Mae)
    Pearson = 'PCC: '+"{:.3f}".format(pearson)

    plt.plot(x1, y0, color=(209/255,102/255,117/255), linewidth=1.5,label = 'y = x')
    plt.legend(loc='upper left')

    plt.fill_between(x1,y1,y11,color='gray',alpha=0.25)
    plt.fill_between(x1,y2,y22,color='gray',alpha=0.1)
    plt.text(73,32, MAE, horizontalalignment='left', verticalalignment='center', fontsize=17, color='black')
    plt.text(73,26, Pearson, horizontalalignment='left', verticalalignment='center', fontsize=17, color='black')

    print('file_dir',file_dir)
    plt.savefig(file_dir)
    plt.clf()





def get_args_parser_pre():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # DataSet
    parser.add_argument('--train_folder'   ,default="/train/"       ,type=str, help="Train set data path ")
    parser.add_argument('--valid_folder'   ,default="/val/"        ,type=str, help="Validation set data path ")
    parser.add_argument('--test_folder'    ,default="/test/"        ,type=str, help="Test set data path ")
    parser.add_argument('--excel_path'     ,default="/18_combine.xls",type=str, help="Excel file path ")

    parser.add_argument('--mask_type', default='cube')
    parser.add_argument('--output_dir', type=str, default='./ckpt_scale_withgender/',
                        help='root path for storing checkpoints, logs')
    parser.add_argument('--num_workers', type=int, default=1, help='pytorch number of worker')

    # Model
    parser.add_argument('--model', type=str, default='scale', help='model name')
    parser.add_argument('--store_name', type=str, default='', help='experiment store name')
    parser.add_argument('--gpu', type=int, default=None)

    # ScaleDense
    parser.add_argument('--scale_block', type=int, default=5)
    parser.add_argument('--scale_channel', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')

    # Loss function
    parser.add_argument('--loss', type=str, default='l1', help='training loss type')
    parser.add_argument('--aux_loss', type=str, default='mse', help='Aux training loss type')
    parser.add_argument('--lambd', type=float, default=10.0, help='Loss weighted between main loss and aux loss')
    parser.add_argument('--beta', type=float, default=10.0,
                        help='Loss weighted between ranking loss and age difference loss')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Loss weighted between groud truth loss and constraint loss')
    parser.add_argument('--sorter', default='./Sodeep_pretrain_weight/best_lstmla_slen_32.pth.tar', type=str,
                        help="When use ranking, the pretrained SoDeep sorter network weight need to be appointed")





    return parser.parse_args()


def model_predict(args,model_name,save_path):
    best_metric = 1e+6
    saved_metrics, saved_epochs = [], []



    print('===== starting prediction ====== ')

    #  DataSet
    print('=====> Preparing data...')

    test_dataset = Integer_Multiple_Batch_Size(
        IMG_Folder_with_mask(args.excel_path, args.test_folder, mask_type=args.mask_type), args.batch_size)

  

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True, drop_last=False)


    print(f"Test data size: {len(test_dataset)}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



    # ========  build and set model  ======== #




    print("======== load trained parameters ======== ")
    model = ScaleDense(nb_block=args.scale_block, nb_filter=args.scale_channel)
    model.to(device)
    model.load_state_dict(torch.load(model_name)['state_dict'])
    print(model_name)



    # Define Loss function
    loss_func_dict = {'l1': nn.L1Loss().to(device)
        , 'mse': nn.MSELoss().to(device)
        , 'matrix_l1': Matrix_distance_loss(p=2)
        , 'matrix_l2': Matrix_distance_L2_loss(p=2)
        , 'matrix_l3': Matrix_distance_L3_loss(p=2)
                      }

    criterion1 = loss_func_dict[args.loss]  # l1


    # Begin to Train the model
    model.eval()  # switch to evaluate mode
    print('======= start prediction =============')
    ues_masked_img=False
    # ======= start test programmer ============= #
    for i in range(1):
        losses = AverageMeter()
        MAE = AverageMeter()

        print('use masked_img:', ues_masked_img)
        out, targ, ID = [], [], []
        target_numpy, predicted_numpy, ID_numpy = [], [], []

        with torch.no_grad():
            for _, (img, img_mask, sid, target, male) in enumerate(test_loader):

                img = img.to(device)
                img_mask = img_mask.to(device)
                
                target = target.type(torch.FloatTensor).to(device)

                male = torch.unsqueeze(male, 1)
                male = torch.zeros(male.shape[0], 2).scatter_(1, male, 1)
                male = male.to(device).type(torch.FloatTensor)
                male = male.to(device)


                # ======= compute output and loss ======= #
                if ues_masked_img:
                    output = model(img_mask,male)

                else:
                    output = model(img,male)
  

                out.append(output.cpu().numpy())
                targ.append(target.cpu().numpy())
                ID.append(sid)
                loss = criterion1(output, target)
                mae = metric(output.detach(), target.detach().cpu())

                # ======= measure accuracy and record loss ======= #
                losses.update(loss, img.size(0))
                MAE.update(mae, img.size(0))

            print('finishing for')

            targ = np.asarray(targ)
            out = np.asarray(out)
            ID = np.asarray(ID)

            for idx in ID:
                for i in idx:
                    ID_numpy.append(i)

            for idx in out:
                for i in idx:
                    predicted_numpy.append(i)

            for idx in targ:
                for i in idx:
                    target_numpy.append(i)

            target_numpy = np.asarray(target_numpy)
            predicted_numpy = np.asarray(predicted_numpy)
            ID_numpy = np.expand_dims(np.asarray(ID_numpy), axis=1)

            print(target_numpy.shape, predicted_numpy.shape, ID_numpy.shape)

            errors = predicted_numpy - target_numpy
            errors = np.squeeze(errors, axis=1)
            target_numpy = np.squeeze(target_numpy, axis=1)
            predicted_numpy = np.squeeze(predicted_numpy, axis=1)

            # ======= output several results  ======= #
            print('===============================================================\n')
            print(
                'TEST  : [steps {0}], Loss {loss.avg:.4f},  MAE:  {MAE.avg:.4f} \n'.format(
                    len(test_loader), loss=losses, MAE=MAE))

            print('STD_err = ', np.std(errors))
            print(' CC:    ', np.corrcoef(target_numpy, predicted_numpy)[0][1])
            print('PAD spear man cc', spearmanr(errors, target_numpy, axis=1))
            print('spear man cc', spearmanr(predicted_numpy, target_numpy, axis=1))

            
            print('\n =================================================================')


            mae_ = round(MAE.avg,4)
            pearson = np.corrcoef(target_numpy, predicted_numpy)[0][1]
            pearson = round(pearson,4)

            plot_img(target_numpy,predicted_numpy,save_path, mae_, pearson)


        ues_masked_img=False



if __name__ == "__main__":

   args = get_args_parser_pre()
   model_name = "/data/disk_2/zhaojinxin/brain_prediction/super_cal/ckpt-best.pth.tar"
   save_path = '/data/disk_2/zhaojinxin/brain_prediction/super_cal/1.jpg'
   args = get_args_parser_pre()
   model_predict(args, model_name, save_path)



