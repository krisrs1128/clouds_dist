import torch
import torch.nn as nn

class SupervisedModel(nn.Module):

    def __init__(self):
        super(SupervisedModel, self).__init__()
        self.height = 128
        self.final_conv = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding=1)
        self.model = nn.ModuleList([nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1) for i in range(20)])
        #self.layer1 = nn.Linear(in_features=8*self.height*self.height, out_features=1*self.height*self.height)

    def forward(self, x):
        #print(x.shape)
        #x_reshaped = x.reshape(x.shape[0], -1)
        #self.layer1(x_reshaped).reshape(x.shape[0], 1, self.height, self.height)
        for layer in self.model:
            x = nn.functional.relu(layer(x))
        return torch.sigmoid(self.final_conv(x))