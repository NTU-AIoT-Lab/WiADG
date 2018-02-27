""" Train source for ADDA """
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from models import LeNetClassifier, LeNetEncoder
from GestureDataset import GestureDataset
from utils import save_model

# super parameters
batch_size = 75
num_workers = 2
num_epochs = 40
lr = 0.001
weight_decay = 1e-6

# load data
data = sio.loadmat('dataset/gesture_l.mat')
x_train, y_train, x_test, y_test = data['x_train'], data['y_train'], data['x_test'], data['y_test']

train_dataset = GestureDataset(x_train, y_train)
test_dataset = GestureDataset(x_test, y_test)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

encoder = LeNetEncoder()
classifier = LeNetClassifier()
optimizer = optim.Adam(list(encoder.parameters())+list(classifier.parameters()),lr=lr, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()

# train network
if torch.cuda.is_available():
	encoder.cuda()
	classifier.cuda()

def train(epoch):
	total_num = 0
	correct_num = 0
	encoder.train()
	classifier.train()
	for step, (inputs, targets) in enumerate(train_loader):
		inputs = Variable(inputs).cuda()
		targets = Variable(targets).cuda()

		# zero gradients for optimizer
		optimizer.zero_grad()

		# compute loss
		outputs = classifier(encoder(inputs))
		loss = criterion(outputs, torch.max(targets, 1)[1])

		# optimize src classifier
		loss.backward()
		optimizer.step()

		_, predicted = torch.max(outputs.data, 1)
		_, labels = torch.max(targets.data, 1)
		total_num += targets.size(0)
		correct_num += predicted.eq(labels).cpu().sum()

		print("Epoch [{}/{}] Step [{}/{}]: loss={:.5f}, accuracy={:.2f}"
			  .format(epoch + 1,
					  num_epochs,
					  step + 1,
					  len(train_loader),
					  loss.data[0],
					  100.*correct_num/total_num))

	# save model parameters
	if ((epoch + 1) > 20):
		save_model(encoder, "large-source-encoder-{}.pt".format(epoch + 1))
		save_model(classifier, "large-source-classifier-{}.pt".format(epoch + 1))
		print ('Save model of epoch {}'.format(epoch + 1))

def test(epoch):
	total_num = 0
	correct_num = 0
	encoder.eval()
	classifier.eval()
	criterion = nn.CrossEntropyLoss()

	for step, (inputs, targets) in enumerate(test_loader):
		inputs = Variable(inputs).cuda()
		targets = Variable(targets).cuda()
		
		outputs = classifier(encoder(inputs))
		loss = criterion(outputs, torch.max(targets, 1)[1])

		total_num += targets.size(0)
		_, predicted = torch.max(outputs.data, 1)
		_, labels = torch.max(targets.data, 1)
		correct_num += predicted.eq(labels).cpu().sum()

	print("Test: Avg Loss = {:.5f}, Avg Accuracy = {:.2f}".format(loss.data[0], 100.*correct_num/total_num))

if __name__ == '__main__':
	for epoch in range(num_epochs):
		train(epoch)
		test(epoch)