"""
Target:
    1. Create the setup for training and testing the model.
    2. Create a basic nn with keeping parameter cound near the target of 8k.
    3. Try to achieve the best accuracy possible.
Result:
    1. Parameters: 10.4k
    2. Best Train Accuracy: 98.55
    3. Best Test Accuracy: 98.43
Analysis:
    1. The model started showing over-fitting for last 5 layers.
    2. The model can now be tweaked to add batch-norm and dropout layers to reduce overfitting.
    3. The parameter count is slightly higher than the target which can be worked on by adding GAP layer.

"""
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 24
    
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 22

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 11

        # CONVOLUTION BLOCK 2
        
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 9

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 7

        # OUTPUT BLOCK
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 7
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(7, 7), padding=0, bias=False),            
        ) # output_size = 2        

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)        
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)        
        x = self.convblock8(x)        
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    
"""
Logs:
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 26, 26]              72
              ReLU-2            [-1, 8, 26, 26]               0
            Conv2d-3            [-1, 8, 24, 24]             576
              ReLU-4            [-1, 8, 24, 24]               0
            Conv2d-5           [-1, 16, 22, 22]           1,152
              ReLU-6           [-1, 16, 22, 22]               0
         MaxPool2d-7           [-1, 16, 11, 11]               0
            Conv2d-8            [-1, 8, 11, 11]             128
              ReLU-9            [-1, 8, 11, 11]               0
           Conv2d-10             [-1, 16, 9, 9]           1,152
             ReLU-11             [-1, 16, 9, 9]               0
           Conv2d-12             [-1, 16, 7, 7]           2,304
             ReLU-13             [-1, 16, 7, 7]               0
           Conv2d-14             [-1, 10, 7, 7]             160
             ReLU-15             [-1, 10, 7, 7]               0
           Conv2d-16             [-1, 10, 1, 1]           4,900
================================================================
Total params: 10,444
Trainable params: 10,444
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.34
Params size (MB): 0.04
Estimated Total Size (MB): 0.38
----------------------------------------------------------------
EPOCH: 0
Loss=0.28239867091178894 Batch_id=468 Accuracy=58.79: 100%|██████████| 469/469 [00:15<00:00, 30.81it/s]

Test set: Average loss: 0.2593, Accuracy: 9216/10000 (92.16%)

EPOCH: 1
Loss=0.37734320759773254 Batch_id=468 Accuracy=93.34: 100%|██████████| 469/469 [00:14<00:00, 31.78it/s]

Test set: Average loss: 0.1417, Accuracy: 9560/10000 (95.60%)

EPOCH: 2
Loss=0.15710221230983734 Batch_id=468 Accuracy=95.47: 100%|██████████| 469/469 [00:14<00:00, 31.52it/s]

Test set: Average loss: 0.1120, Accuracy: 9681/10000 (96.81%)

EPOCH: 3
Loss=0.10552438348531723 Batch_id=468 Accuracy=96.57: 100%|██████████| 469/469 [00:15<00:00, 29.85it/s]

Test set: Average loss: 0.0862, Accuracy: 9742/10000 (97.42%)

EPOCH: 4
Loss=0.0897369459271431 Batch_id=468 Accuracy=97.18: 100%|██████████| 469/469 [00:13<00:00, 33.89it/s]

Test set: Average loss: 0.0706, Accuracy: 9767/10000 (97.67%)

EPOCH: 5
Loss=0.04454916715621948 Batch_id=468 Accuracy=97.52: 100%|██████████| 469/469 [00:14<00:00, 33.46it/s]

Test set: Average loss: 0.0739, Accuracy: 9774/10000 (97.74%)

EPOCH: 6
Loss=0.0855405330657959 Batch_id=468 Accuracy=97.79: 100%|██████████| 469/469 [00:13<00:00, 33.75it/s]

Test set: Average loss: 0.0696, Accuracy: 9793/10000 (97.93%)

EPOCH: 7
Loss=0.07727101445198059 Batch_id=468 Accuracy=97.89: 100%|██████████| 469/469 [00:14<00:00, 32.60it/s]

Test set: Average loss: 0.0607, Accuracy: 9812/10000 (98.12%)

EPOCH: 8
Loss=0.11550569534301758 Batch_id=468 Accuracy=98.22: 100%|██████████| 469/469 [00:14<00:00, 33.43it/s]

Test set: Average loss: 0.0557, Accuracy: 9824/10000 (98.24%)

EPOCH: 9
Loss=0.03993288055062294 Batch_id=468 Accuracy=98.25: 100%|██████████| 469/469 [00:14<00:00, 33.35it/s]

Test set: Average loss: 0.0536, Accuracy: 9824/10000 (98.24%)

EPOCH: 10
Loss=0.12980400025844574 Batch_id=468 Accuracy=98.32: 100%|██████████| 469/469 [00:13<00:00, 33.66it/s]

Test set: Average loss: 0.0508, Accuracy: 9843/10000 (98.43%)

EPOCH: 11
Loss=0.028856180608272552 Batch_id=468 Accuracy=98.43: 100%|██████████| 469/469 [00:14<00:00, 33.31it/s]

Test set: Average loss: 0.0595, Accuracy: 9817/10000 (98.17%)

EPOCH: 12
Loss=0.027617596089839935 Batch_id=468 Accuracy=98.45: 100%|██████████| 469/469 [00:14<00:00, 32.22it/s]

Test set: Average loss: 0.0549, Accuracy: 9830/10000 (98.30%)

EPOCH: 13
Loss=0.02456125058233738 Batch_id=468 Accuracy=98.55: 100%|██████████| 469/469 [00:14<00:00, 33.26it/s]

Test set: Average loss: 0.0585, Accuracy: 9820/10000 (98.20%)

EPOCH: 14
Loss=0.018367648124694824 Batch_id=468 Accuracy=98.58: 100%|██████████| 469/469 [00:14<00:00, 33.22it/s]

Test set: Average loss: 0.0497, Accuracy: 9840/10000 (98.40%)
"""