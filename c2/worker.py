import torch
import torch.nn as nn
#import torch.optim as optim

from my_net import *

class Worker(nn.Module):
	def __init__(self):
		super().__init__()
		self.model = MyNet()
		if torch.cuda.is_available():
			self.input_device = torch.device("cuda:0")
		else:
			self.input_device = torch.device("cpu")

	def pull_weights(self, model_params):
		self.model.load_state_dict(model_params)

	def push_gradients(self, batch_idx, data, target):
		data, target = data.to(self.input_device), target.to(self.input_device)		
		output = self.model(data)
		data.requires_grad = True
		loss = F.nll_loss(output, target)
		loss.backward()
		grads = []
		for layer in self.parameters():
			grad = layer.grad
			grads.append(grad)
		print(f"batch {batch_idx} training :: loss {loss.item()}")
		return grads

