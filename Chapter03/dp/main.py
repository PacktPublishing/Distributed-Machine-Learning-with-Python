import torch
from torch.utils.data import Dataset, DataLoader
from my_net import *

from torchvision import datasets, transforms
from torch import optim

train_set = datasets.MNIST('./mnist_data', download=True, train=True,
               transform = transforms.Compose([transforms.ToTensor(),
               transforms.Normalize((0.1307,),(0.3081,))]))

test_set = datasets.MNIST('./mnist_data', download=True, train=False,
              transform = transforms.Compose([transforms.ToTensor(), 
              transforms.Normalize((0.1307,),(0.3081,))]))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = DataLoader(train_set, batch_size = 128, shuffle=True, pin_memory = True)

train_epoch = 2

def main():
	model = MyNet()
	print("Using ", torch.cuda.device_count(), "GPUs for data parallel training")
	optimizer = torch.optim.SGD(model.parameters(), lr = 5e-4)
	model = nn.DataParallel(model)
	model.to(device)
	#Training
	for epoch in range(train_epoch):
		print(f"Epoch {epoch}")
		for idx, (data, target) in enumerate(train_loader):
			data, target = data.cuda(), target.cuda()
			output = model(data)
			loss = F.cross_entropy(output, target)
			loss.backward()
			optimizer.step()
			print(f"batch {idx}, loss {loss.item()}")
	print("Training Done!")
	

if __name__ == '__main__':
	main()

