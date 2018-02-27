"""
	Train encoder for target domain
"""
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from models import LeNetClassifier, LeNetEncoder, Discriminator
from GestureDataset import GestureDataset
from utils import save_model, make_variable
from test_tgt import test_tgt

####################
# 1. setup network #
####################

# super parameters
batch_size = 75
num_workers = 2
num_epochs = 4000
d_learning_rate = 1e-4
c_learning_rate = 1e-4
weight_decay = 1e-6
beta1 = 0.5
beta2 = 0.9
save_step = 100
eva_step = 10
num_train_encoder = 1

# load data
data = sio.loadmat('dataset/gesture_l.mat')
x_train, y_train, x_test, y_test = data['x_train'], data['y_train'], data['x_test'], data['y_test']
train_dataset = GestureDataset(x_train, y_train)
test_dataset = GestureDataset(x_test, y_test)
src_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
src_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

data = sio.loadmat('dataset/gesture_s.mat')
x_train, y_train, x_test, y_test = data['x_train'], data['y_train'], data['x_test'], data['y_test']
train_dataset = GestureDataset(x_train, y_train)
test_dataset = GestureDataset(x_test, y_test)
tgt_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
tgt_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

# define model
src_encoder = LeNetEncoder()
tgt_encoder = LeNetEncoder()
discriminator = Discriminator()
src_encoder.load_state_dict(torch.load('parameters/large-source-encoder-final.pt'))
tgt_encoder.load_state_dict(torch.load('parameters/large-source-encoder-final.pt'))

if torch.cuda.is_available():
	src_encoder.cuda()
	tgt_encoder.cuda()
	discriminator.cuda()

tgt_encoder.train()
discriminator.train()
print (tgt_encoder)
print (discriminator)

# setup criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer_tgt = optim.Adam(tgt_encoder.parameters(),
						   lr=c_learning_rate,
						   betas=(beta1, beta2))
optimizer_dc = optim.Adam(discriminator.parameters(),
							  lr=d_learning_rate,
							  betas=(beta1, beta2))

len_data = len(src_train_loader)

####################
# 2. train network #
####################

for epoch in range(num_epochs):
	# generate source and target pair
	data_zip = enumerate(zip(src_train_loader, tgt_train_loader))
	for step, ((input_src, _), (input_tgt, _)) in data_zip:
		###########################
		# 2.1 train discriminator #
		###########################
		input_src = make_variable(input_src)
		input_tgt = make_variable(input_tgt)

		# concatenate features
		optimizer_dc.zero_grad()
		feat_src = src_encoder(input_src)
		feat_tgt = tgt_encoder(input_tgt)
		feat_concat = torch.cat((feat_src, feat_tgt), 0)

		# generate real/fake label
		label_src = make_variable(torch.ones(feat_src.size(0)).long())
		label_tgt = make_variable(torch.zeros(feat_tgt.size(0)).long())
		label = torch.cat((label_src, label_tgt), 0)

		pred = discriminator(feat_concat.detach())
		loss_dc = criterion(pred, label)
		loss_dc.backward()

		optimizer_dc.step()
		pred_cls = torch.squeeze(pred.max(1)[1])
		acc = (pred_cls == label).float().mean()

		############################
		# 2.2 train target encoder #
		############################

		for i in range(num_train_encoder):
			optimizer_dc.zero_grad()
			optimizer_tgt.zero_grad()

			feat_tgt = tgt_encoder(input_tgt)
			pred_tgt = discriminator(feat_tgt)

			label_tgt = make_variable(torch.ones(feat_tgt.size(0)).long())
			loss_tgt = criterion(pred_tgt, label_tgt)
			loss_tgt.backward()

			optimizer_tgt.step()

		if ((step + 1) % 12 == 0):
			print("Epoch [{}/{}] Step [{}/{}]:"
				  "d_loss={:.5f} g_loss={:.5f} acc={:.5f}"
				  .format(epoch + 1,
						  num_epochs,
						  step + 1,
						  len_data,
						  loss_dc.data[0],
						  loss_tgt.data[0],
						  acc.data[0]))
	if ((epoch + 1) % eva_step == 0):
		test_tgt(epoch + 1, tgt_train_loader, tgt_encoder)
	if ((epoch + 1) % save_step == 0 and (epoch + 1) > 2000):
		# save_model(discriminator, "discriminator-s-l-{}.pt".format(epoch + 1))
		save_model(tgt_encoder, "l-s-target-encoder-{}.pt".format(epoch + 1))

if ((epoch + 1) % save_step == 0):
	# save_model(discriminator, "discriminator-s-l-final.pt")
	save_model(tgt_encoder, "l-s-target-encoder-final.pt")
