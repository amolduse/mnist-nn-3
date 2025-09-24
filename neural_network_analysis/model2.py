"""
Target:
    1. Improve the architecture of previously trained network by adding following features one by one and observing the impact- 
        a. Add batch-norm layers
        b. Add dropout layers
        c. Add GAP layer
        d. Reduce the parameter count to less than 10k (by using 1x1 conv layers, GAP etc)
    2.Reduce the overfitting of the model by using above techniques like batch-norm, dropout etc.
Result:
    1. Parameters: 
        a. With only batch-norm 10.6k
        b. By adding dropout 10.6k
        c. By adding GAP 5.6k
    2. Best Train Accuracy: 
        a. With only batch-norm -  99.81
        b. By adding dropout 99.24
        c. By adding GAP 98.59
        d. By increasing capacity 98.86
    3. Best Test Accuracy:
        a. With only batch-norm -  99.18
        b. By adding dropout 99.2
        c. By adding GAP 98.73
        d. By increasing capacity 99.2

Analysis:
    1. The model showed over-fitting for when only batch norm was used.
    2. After adding a dropout the model started showing closer train and test accuracies but it was not able to pass the 
       training to test as it got closer to 15th epoch and started showing overfitting. Time to add GAP layer and reduce parameters 
       close to the target of 8k.
    3. Adding GAP layer reduced parameter count but decreased accuracy but we cannot compare it with our previous model because parameter
       count is too less. Its time to add some capacity to the model and see if we can reach closer to 99.4% accuracy.
    4. Adding some capacity to the model helped us to increase accuracy but it was capped to around 99.2 which is less than our target.
       But we have achieved our target of reducing parameters to less than 8k.
    5. Our model needs further improvement and in next trial we will try other things like image augmentation, schedulers etc.   


"""
import torch.nn as nn
import torch.nn.functional as F

dropout_value = 0.1
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 24
    
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 22

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 11

        # CONVOLUTION BLOCK 2
        
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 9

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 7

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 5

        # OUTPUT BLOCK
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 5
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=5) # 7>> 9... nn.AdaptiveAvgPool((1, 1))
        ) # output_size = 1       

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
        x = self.gap(x)        
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    
"""
Logs:
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 26, 26]              72
       BatchNorm2d-2            [-1, 8, 26, 26]              16
              ReLU-3            [-1, 8, 26, 26]               0
           Dropout-4            [-1, 8, 26, 26]               0
            Conv2d-5           [-1, 16, 24, 24]           1,152
       BatchNorm2d-6           [-1, 16, 24, 24]              32
              ReLU-7           [-1, 16, 24, 24]               0
           Dropout-8           [-1, 16, 24, 24]               0
            Conv2d-9           [-1, 16, 22, 22]           2,304
      BatchNorm2d-10           [-1, 16, 22, 22]              32
             ReLU-11           [-1, 16, 22, 22]               0
          Dropout-12           [-1, 16, 22, 22]               0
        MaxPool2d-13           [-1, 16, 11, 11]               0
           Conv2d-14            [-1, 8, 11, 11]             128
           Conv2d-15              [-1, 8, 9, 9]             576
      BatchNorm2d-16              [-1, 8, 9, 9]              16
             ReLU-17              [-1, 8, 9, 9]               0
          Dropout-18              [-1, 8, 9, 9]               0
           Conv2d-19             [-1, 16, 7, 7]           1,152
      BatchNorm2d-20             [-1, 16, 7, 7]              32
             ReLU-21             [-1, 16, 7, 7]               0
          Dropout-22             [-1, 16, 7, 7]               0
           Conv2d-23             [-1, 16, 5, 5]           2,304
      BatchNorm2d-24             [-1, 16, 5, 5]              32
             ReLU-25             [-1, 16, 5, 5]               0
          Dropout-26             [-1, 16, 5, 5]               0
           Conv2d-27             [-1, 10, 5, 5]             160
        AvgPool2d-28             [-1, 10, 1, 1]               0
================================================================
Total params: 8,008
Trainable params: 8,008
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.76
Params size (MB): 0.03
Estimated Total Size (MB): 0.80
----------------------------------------------------------------
EPOCH: 0
Loss=0.1639290153980255 Batch_id=468 Accuracy=79.11: 100%|██████████| 469/469 [00:17<00:00, 26.43it/s]

Test set: Average loss: 0.1268, Accuracy: 9691/10000 (96.91%)

EPOCH: 1
Loss=0.12630224227905273 Batch_id=468 Accuracy=96.91: 100%|██████████| 469/469 [00:19<00:00, 24.28it/s]

Test set: Average loss: 0.0599, Accuracy: 9830/10000 (98.30%)

EPOCH: 2
Loss=0.06779229640960693 Batch_id=468 Accuracy=97.64: 100%|██████████| 469/469 [00:17<00:00, 26.92it/s]

Test set: Average loss: 0.0504, Accuracy: 9863/10000 (98.63%)

EPOCH: 3
Loss=0.11628531664609909 Batch_id=468 Accuracy=98.07: 100%|██████████| 469/469 [00:18<00:00, 25.99it/s]

Test set: Average loss: 0.0531, Accuracy: 9841/10000 (98.41%)

EPOCH: 4
Loss=0.08715184777975082 Batch_id=468 Accuracy=98.24: 100%|██████████| 469/469 [00:16<00:00, 28.34it/s]

Test set: Average loss: 0.0480, Accuracy: 9852/10000 (98.52%)

EPOCH: 5
Loss=0.036554623395204544 Batch_id=468 Accuracy=98.31: 100%|██████████| 469/469 [00:18<00:00, 25.68it/s]

Test set: Average loss: 0.0349, Accuracy: 9897/10000 (98.97%)

EPOCH: 6
Loss=0.07117868214845657 Batch_id=468 Accuracy=98.52: 100%|██████████| 469/469 [00:17<00:00, 27.19it/s]

Test set: Average loss: 0.0364, Accuracy: 9879/10000 (98.79%)

EPOCH: 7
Loss=0.12529577314853668 Batch_id=468 Accuracy=98.57: 100%|██████████| 469/469 [00:17<00:00, 26.48it/s]

Test set: Average loss: 0.0293, Accuracy: 9906/10000 (99.06%)

EPOCH: 8
Loss=0.034706030040979385 Batch_id=468 Accuracy=98.63: 100%|██████████| 469/469 [00:17<00:00, 27.35it/s]

Test set: Average loss: 0.0297, Accuracy: 9911/10000 (99.11%)

EPOCH: 9
Loss=0.09916075319051743 Batch_id=468 Accuracy=98.75: 100%|██████████| 469/469 [00:17<00:00, 26.15it/s]

Test set: Average loss: 0.0341, Accuracy: 9900/10000 (99.00%)

EPOCH: 10
Loss=0.02140660583972931 Batch_id=468 Accuracy=98.77: 100%|██████████| 469/469 [00:18<00:00, 25.80it/s]

Test set: Average loss: 0.0284, Accuracy: 9917/10000 (99.17%)

EPOCH: 11
Loss=0.01686152070760727 Batch_id=468 Accuracy=98.71: 100%|██████████| 469/469 [00:18<00:00, 24.69it/s]

Test set: Average loss: 0.0297, Accuracy: 9919/10000 (99.19%)

EPOCH: 12
Loss=0.08477044105529785 Batch_id=468 Accuracy=98.83: 100%|██████████| 469/469 [00:17<00:00, 26.57it/s]

Test set: Average loss: 0.0245, Accuracy: 9917/10000 (99.17%)

EPOCH: 13
Loss=0.021237486973404884 Batch_id=468 Accuracy=98.84: 100%|██████████| 469/469 [00:18<00:00, 25.80it/s]

Test set: Average loss: 0.0252, Accuracy: 9919/10000 (99.19%)

EPOCH: 14
Loss=0.04194178059697151 Batch_id=468 Accuracy=98.86: 100%|██████████| 469/469 [00:17<00:00, 26.85it/s]

Test set: Average loss: 0.0262, Accuracy: 9920/10000 (99.20%)
"""