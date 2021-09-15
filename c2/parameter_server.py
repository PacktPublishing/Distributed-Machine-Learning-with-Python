import torch
import torch.nn as nn
import torch.optim as optim
from my_net import *

class ParameterServer(nn.Module):
	def __init__(self):
		super().__init__()
		self.model = MyNet()

		if torch.cuda.is_available():
			self.input_device = torch.device("cuda:0")
		else:
			self.input_device = torch.device("cpu")

		self.optimizer = optim.SGD(self.model.parameters(), lr = 0.05)

	def get_weights(self):
		return self.model.state_dict()

	def update_model(self, grads):
		for para, grad in zip(self.model.parameters(), grads):
			para.grad = grad
		self.optimizer.step()
		self.optimizer.zero_grad()
				
