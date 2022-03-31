import torch
import torch.nn as nn
import torch.nn.functional as F

class MyNet(nn.Module):
	def __init__(self):
		super(MyNet, self).__init__()
		if torch.cuda.is_available():
			device = torch.device(f"cuda")
		else:
			device = torch.device("cpu")
		self.conv1 = nn.Conv2d(1,32,3,1).to(device)
		self.dropout1 = nn.Dropout2d(0.5).to(device)
		self.conv2 = nn.Conv2d(32,64,3,1).to(device)
		self.dropout2 = nn.Dropout2d(0.75).to(device)
		self.fc1 = nn.Linear(9216, 128).to(device)
		self.fc2 = nn.Linear(128,20).to(device)
		self.fc3 = nn.Linear(20,10).to(device)

	def forward(self, x):
		x = self.conv1(x)
		x = self.dropout1(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = self.dropout2(x)
		x = F.max_pool2d(x,2)
		x = torch.flatten(x,1)

		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		x = self.fc3(x)

		output = F.log_softmax(x, dim = 1)
		return output

