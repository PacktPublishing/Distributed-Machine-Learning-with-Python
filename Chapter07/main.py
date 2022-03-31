import argparse
import os
import datetime

from my_net import *
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
import torch.multiprocessing as mp
from torchvision import datasets, transforms
from torch import optim
from torch.distributed.optim import DistributedOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler as DDP_sampler

train_all_set = datasets.MNIST('./mnist_data', download=True, train=True,
               transform = transforms.Compose([transforms.ToTensor(),
               transforms.Normalize((0.1307,),(0.3081,))]))
train_set, val_set = torch.utils.data.random_split( train_all_set,
        			 [50000, 10000])

test_set = datasets.MNIST('./mnist_data', download=True, train=False,
              transform = transforms.Compose([transforms.ToTensor(), 
              transforms.Normalize((0.1307,),(0.3081,))]))

def train(args):
    model = MyNet()
    model.train()
    trainset= datasets.MNIST('./mnist_data', download=True, train=True,
               transform = transforms.Compose([transforms.ToTensor(),
               transforms.Normalize((0.1307,),(0.3081,))]))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    criterion = nn.CrossEntropyLoss()			
    optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")
        for idx, (data, target) in enumerate(trainloader):
            data = data.to('cuda:0')
            optimizer.zero_grad()
            output = model(data)
            target = target.to(output.device)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            print(f"batch {idx} training :: loss {loss.item()}")
        print("Training Done!")
    return model
def test(args, model):
    model.eval()
    testset = datasets.MNIST('./mnist_data', download=True, train=True,
               transform = transforms.Compose([transforms.ToTensor(),
               transforms.Normalize((0.1307,),(0.3081,))]))
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,shuffle=False, num_workers=4)
    correct_total = 0
    with torch.no_grad():
        for idx, (data, target) in enumerate(testloader):
            output = model(data.to('cuda:0'))
            predict = output.argmax(dim=1, keepdim=True).to(output.device)
            target = target.to(output.device)
            correct = predict.eq(target.view_as(predict)).sum().item()
            correct_total += correct
            acc = correct_total/len(testloader.dataset)
            print(f"Test Accuracy {acc}")
    print("Test Done!")

def main():
    parser = argparse.ArgumentParser(description = 'model parallel training')
    parser.add_argument('-e', '--epochs', default = 4, type = int, help='number of epochs')
    args = parser.parse_args()
    trained_model = train(args)
    test(args, trained_model)

if __name__ == '__main__':
    main()
