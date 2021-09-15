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

def net_setup():
	os.environ['MASTER_ADDR'] = '172.31.26.15'
	os.environ['MASTER_PORT'] = '12345'

def checkpointing(rank, epoch, net, optimizer, loss):
	path = f"model{rank}.pt"
	torch.save({
				'epoch':epoch,
				'model_state':net.state_dict(),
				'loss': loss,
				'optim_state': optimizer.state_dict(),
				}, path)
	print(f"Checkpointing model {rank} done.")

def load_checkpoint(rank, machines):
	path = f"model{rank}.pt"
	checkpoint = torch.load(path)
	model = torch.nn.DataParallel(MyNet(), device_ids=[rank%machines])
	optimizer = torch.optim.SGD(model.parameters(), lr = 5e-4)

	epoch = checkpoint['epoch']
	loss = checkpoint['loss']
	model.load_state_dict(checkpoint['model_state'])
	optimizer.load_state_dict(checkpoint['optim_state'])
	return model, optimizer, epoch, loss
	
def validation(model, val_set):
	model.eval()
	val_loader = torch.utils.data.DataLoader(val_set, batch_size=128)
	correct_total = 0
	with torch.no_grad():
		for idx, (data, target) in enumerate(val_loader):
			output = model(data)
			predict = output.argmax(dim=1, keepdim=True).cuda()
			target = target.cuda()
			correct = predict.eq(target.view_as(predict)).sum().item()
			correct_total += correct
		acc = correct_total/len(val_loader.dataset)
	print(f"Validation Accuracy {acc}")

def train(local_rank, args):
	torch.manual_seed(123)
	world_size = args.machines*args.gpus
	rank = args.mid * args.gpus + local_rank
	dist.init_process_group('nccl', rank =rank, world_size = world_size,
                            timeout=datetime.timedelta(seconds=60))
	
	torch.cuda.set_device(local_rank)
	model = MyNet()
	local_train_sampler = DDP_sampler(datasets.MNIST('./mnist_data', download=True, train=True,
               transform = transforms.Compose([transforms.ToTensor(),
               transforms.Normalize((0.1307,),(0.3081,))])), rank = rank, num_replicas = world_size) 
	local_train_loader = torch.utils.data.DataLoader(datasets.MNIST('./mnist_data', download=True, train=True,
               transform = transforms.Compose([transforms.ToTensor(),
               transforms.Normalize((0.1307,),(0.3081,))])),
							batch_size = 128,
							shuffle = False,
							sampler = local_train_sampler)

	optimizer = torch.optim.SGD(model.parameters(), lr = 5e-4)
	model = DDP(model, device_ids=[local_rank])

	for epoch in range(args.epochs):
		print(f"Epoch {epoch}")
		for idx, (data, target) in enumerate(local_train_loader):
			data = data.cuda()
			target = target.cuda()
			output = model(data)
			loss = F.cross_entropy(output, target)
			loss.backward()
			optimizer.step()
			print(f"batch {idx} training :: loss {loss.item()}")
		checkpointing(rank, epoch, model, optimizer, loss.item())
		validation(model, val_set)
	print("Training Done!")
	dist.destroy_process_group()
	
def test(local_rank, args):
	world_size = args.machines*args.gpus
	rank = args.mid * args.gpus + local_rank
	dist.init_process_group('nccl', rank =rank, world_size = world_size,
                            timeout=datetime.timedelta(seconds=60))

	torch.cuda.set_device(local_rank)
	print(f"Load checkpoint {rank}")
	model, optimizer, epoch, loss = load_checkpoint(rank, args.machines)
	print("Checkpoint loading done!")

	local_test_sampler = DDP_sampler(test_set, rank = rank, num_replicas = world_size)

	model.eval()
	local_test_loader = torch.utils.data.DataLoader(test_set, 
							batch_size=128,
							shuffle = False, 
							sampler = local_test_sampler)
	correct_total = 0
	with torch.no_grad():
		for idx, (data, target) in enumerate(local_test_loader):
			output = model(data)
			predict = output.argmax(dim=1, keepdim=True).cuda()
			target = target.cuda()
			correct = predict.eq(target.view_as(predict)).sum().item()
			correct_total += correct
		acc = correct_total/len(local_test_loader.dataset)
	print(f"GPU {rank}, Test Accuracy {acc}")
	print("Test Done!")
	dist.destroy_process_group()

def main():
	parser = argparse.ArgumentParser(description = 'distributed data parallel training')
	parser.add_argument('-m', '--machines', default=2, type=int, help='number of machines')
	parser.add_argument('-g', '--gpus', default = 4, type=int, help='number of GPUs in a machine')
	parser.add_argument('-id', '--mid', default = 0, type=int, help='machine id number')
	parser.add_argument('-e', '--epochs', default = 10, type = int, help='number of epochs')
	args = parser.parse_args()
	net_setup()
	mp.spawn(train, nprocs=args.gpus, args=(args,), join=True)
	mp.spawn(test, nprocs=args.gpus, args=(args,), join=True)

if __name__ == '__main__':
    main()
