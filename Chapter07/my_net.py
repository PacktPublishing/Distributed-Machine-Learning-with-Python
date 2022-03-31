import torch
import torch.nn as nn
import torch.nn.functional as F

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.seq1 = nn.Sequential(
		        nn.Conv2d(1,32,3,1),
		        nn.Dropout2d(0.5),
		        nn.Conv2d(32,64,3,1),
		        nn.Dropout2d(0.75)).to('cuda:0')
        self.seq2 = nn.Sequential(
		        nn.Linear(9216, 128),
		        nn.Linear(128,20),
		        nn.Linear(20,10)).to('cuda:2')

    def forward(self, x):
        x = self.seq1(x.to('cuda:0'))
        x = F.max_pool2d(x,2).to('cuda:1')
        x = torch.flatten(x,1).to('cuda:1')
        x = self.seq2(x.to('cuda:2'))
        output = F.log_softmax(x, dim = 1)
        return output
