import torch
import numpy as np 
import torch.nn as nn
import scipy.stats.stats as stats
import torch.nn.functional as F

class Matrix_distance_loss(torch.nn.Module):
    def __init__(self, p=1):
        super().__init__()
        self.p = p
        self.criterion_mse = torch.nn.MSELoss()
        self.criterion_mae = nn.L1Loss()

    def forward(self, pred, gt):
        '''
        Computes the P-norm distance between every pair of row vector in the input.
        This is identical to the upper triangular portion, excluding the diagnoal,
        of
        torch.norm(input[:, None] - input, dim=2, p=p).
        This function will be faster if the row are contiguous.
        '''

        Matrix_Distance_pred = torch.norm(pred[:, None] - pred, dim=2, p=self.p)


        Matrix_Distance_gt = torch.norm(gt[:, None] - gt, dim=2, p=self.p)
        

        Matrix_Distance_pred = Matrix_Distance_pred * Matrix_Distance_pred
        Matrix_Distance_gt = Matrix_Distance_gt * Matrix_Distance_gt
        


        loss = self.criterion_mae(Matrix_Distance_gt, Matrix_Distance_pred)
       
        return loss


class Matrix_distance_L2_loss(torch.nn.Module):
    def __init__(self, p=1):
        super().__init__()
        self.p = p
        self.criterion_mae = nn.L1Loss()

    def forward(self, pred, gt):
        '''
        Computes the P-norm distance between every pair of row vector in the input.
        This is identical to the upper triangular portion, excluding the diagnoal,
        of
        torch.norm(input[:, None] - input, dim=2, p=p).
        This function will be faster if the row are contiguous.
        '''
        Matrix_Distance_pred = torch.norm(pred[:, None] - gt, dim=2, p=self.p)
        Matrix_Distance_gt = torch.norm(gt[:, None] - gt, dim=2, p=self.p)

        Matrix_length = Matrix_Distance_pred.size(0)
        Matrix_mask = 1 - torch.eye(Matrix_length)
        Matrix_Distance_pred = Matrix_Distance_pred * Matrix_mask
        Matrix_Distance_gt = Matrix_Distance_gt * Matrix_mask


        Matrix_Distance_pred = Matrix_Distance_pred * Matrix_Distance_pred
        Matrix_Distance_gt = Matrix_Distance_gt * Matrix_Distance_gt
        
  

        loss = self.criterion_mae(Matrix_Distance_gt, Matrix_Distance_pred)
       
        return loss


class Matrix_distance_L2_loss_MAE(torch.nn.Module):
    def __init__(self, p=1):
        super().__init__()
        self.p = p
        self.criterion_mae = nn.L1Loss()

    def forward(self, pred, gt):
        '''
        Computes the P-norm distance between every pair of row vector in the input.
        This is identical to the upper triangular portion, excluding the diagnoal,
        of
        torch.norm(input[:, None] - input, dim=2, p=p).
        This function will be faster if the row are contiguous.
        '''
        Matrix_Distance_pred = torch.norm(pred[:, None] - gt, dim=2, p=self.p)
        Matrix_Distance_gt = torch.norm(gt[:, None] - gt, dim=2, p=self.p)
        
        # print('Matrix_Distance:')
        # print(Matrix_Distance_pred)
        # print(Matrix_Distance_gt)
        # print(Matrix_Distance_pred-Matrix_Distance_gt)

        loss = self.criterion_mae(Matrix_Distance_gt, Matrix_Distance_pred)
       
        return loss


class Matrix_distance_L3_loss(torch.nn.Module):
    def __init__(self, p=1):
        super().__init__()
        self.p = p
        self.criterion_mse = torch.nn.MSELoss()

    def forward(self, pred, gt):
        '''
        Computes the P-norm distance between every pair of row vector in the input.
        This is identical to the upper triangular portion, excluding the diagnoal,
        of
        torch.norm(input[:, None] - input, dim=2, p=p).
        This function will be faster if the row are contiguous.
        '''
        Matrix_Distance_pred = torch.norm(pred[:, None] - pred, dim=2, p=self.p)
        Matrix_Distance_gt = torch.norm(gt[:, None] - pred, dim=2, p=self.p)
        
        loss = self.criterion_mse(Matrix_Distance_gt, Matrix_Distance_pred)
       
        return loss

def test_loss():
       
    # ======== use a simple neural network to test loss function =========
    input_size = 1
    output_size = 1
    num_epochs = 4000
    learning_rate = 0.001
    
    x_train = np.array(np.ones((100,1)),dtype=np.float32)
    print(x_train.shape)
    y_train = np.array(np.ones((100,1)) * 10 + np.random.rand(100,1),dtype=np.float32)
    print(y_train.shape)
    
    # ========  Linear regression model  ======== 
    model = nn.Sequential(nn.Linear(input_size, 128),
                          nn.Linear(128,1))

    criterion = Matrix_distance_loss()

    #  ======== Adam =========
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
    loss_dict = []

    # ========  Train the model 
    for epoch in range(num_epochs):
        inputs = torch.from_numpy(x_train)
        targets = torch.from_numpy(y_train)

    # ========  forward and loss ========  #
        outputs = model(inputs)
        loss = criterion(outputs, targets)
       
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #  =====  loss ==========  #
        loss_dict.append(loss.item())
        if (epoch+1) % 100 == 0:
            print ('Epoch [{}/{}], Loss: {:.6f}'.format(epoch+1, num_epochs, loss.item()))
    
# if __name__ == "__main__":  
    
#     X = torch.Tensor(((1.,),(2.,)))
#     Y = torch.Tensor(((3.,),(3.,)))
#     # X = torch.Tensor(((1.),(2.)))
#     # Y = torch.Tensor(((3.),(4.)))
#     # print(X)
#     # X=X.reshape(2,1)
#     # Y=Y.reshape(2,1)
#     # # print(X)
#     print(X.shape,Y.shape)
#     # print(X[:,None])
#     # print(X[:, None] - Y)
    
#     # matrix_A = torch.norm(X[:, None] - Y, dim=2, p=1)
#     # matrix_B = torch.norm(Y[:, None] - Y, dim=2, p=1)

#     # # criterion = nn.L1Loss()

#     # # print(X.shape, Y.shape)
#     # # print(X[:None].shape)
#     # print(matrix_A)
#     # print(matrix_B)
#     # print(matrix_A - matrix_B)
#     # print(criterion(matrix_A, matrix_B))
#     criterion = Matrix_distance_L2_loss()
#     criterion2 = Matrix_distance_L2_loss_MAE()

#     loss = criterion(X,Y)
#     loss2 = criterion2(X,Y)

#     print(loss)
#     print(loss2)

 