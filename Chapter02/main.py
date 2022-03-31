import torch
from torchvision import datasets, transforms

from my_net import *
from worker import *
from parameter_server import *

train_loader = torch.utils.data.DataLoader(datasets.MNIST('./mnist_data', download=True, train=True,
               transform = transforms.Compose([transforms.ToTensor(), 
               transforms.Normalize((0.1307,),(0.3081,))])),
               batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('./mnist_data', download=True, train=False,
              transform = transforms.Compose([transforms.ToTensor(), 
              transforms.Normalize((0.1307,),(0.3081,))])),
              batch_size=128, shuffle=True)

def main():
	ps = ParameterServer()
	worker = Worker()
	
	for batch_idx, (data, target) in enumerate(train_loader):
		params = ps.get_weights()
		worker.pull_weights(params)
		grads = worker.push_gradients(batch_idx, data, target)
		ps.update_model(grads)
	print("Done Training")

if __name__ == '__main__':
	main()
