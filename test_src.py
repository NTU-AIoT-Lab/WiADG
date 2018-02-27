""" 
	Test original source for ADDA 
	Source CNN + Target dataset 
"""

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

# 0 test small/ 1 test large
test_flag = 0

# super parameters
batch_size = 75
num_workers = 2
num_epochs = 40
lr = 0.001
weight_decay = 1e-6

if test_flag == 0:
	data = sio.loadmat('dataset/gesture_s.mat')
else:
	data = sio.loadmat('dataset/gesture_l.mat')
x_train, y_train, x_test, y_test = data['x_train'], data['y_train'], data['x_test'], data['y_test']

train_dataset = GestureDataset(x_train, y_train)
test_dataset = GestureDataset(x_test, y_test)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

encoder = LeNetEncoder()
classifier = LeNetClassifier()
if test_flag == 0:
	encoder.load_state_dict(torch.load('parameters/large-source-encoder-final.pt'))
	classifier.load_state_dict(torch.load('parameters/large-source-classifier-final.pt'))
else:
	# encoder.load_state_dict(torch.load('parameters/s-l-target-encoder-50.pt'))
	encoder.load_state_dict(torch.load('parameters/small-source-encoder-final.pt'))
	classifier.load_state_dict(torch.load('parameters/small-source-classifier-final.pt'))

criterion = nn.CrossEntropyLoss()

# train network
if torch.cuda.is_available():
	encoder.cuda()
	classifier.cuda()

total_num = 0
correct_num = 0
encoder.eval()
classifier.eval()
criterion = nn.CrossEntropyLoss()

for step, (inputs, targets) in enumerate(train_loader):
	inputs = Variable(inputs).cuda()
	targets = Variable(targets).cuda()
	
	outputs = classifier(encoder(inputs))
	loss = criterion(outputs, torch.max(targets, 1)[1])

	total_num += targets.size(0)
	_, predicted = torch.max(outputs.data, 1)
	_, labels = torch.max(targets.data, 1)
	correct_num += predicted.eq(labels).cpu().sum()

print("Test: Avg Loss = {:.5f}, Avg Accuracy = {:.2f}".format(loss.data[0], 100.*correct_num/total_num))