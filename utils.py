import os
import random

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

def save_model(net, filename):
	"""Save trained model."""
	model_root = "parameters/"
	torch.save(net.state_dict(),
			   os.path.join(model_root, filename))
	print("save pretrained model to: {}".format(os.path.join(model_root,
															 filename)))

def make_variable(tensor, volatile=False):
	"""Convert Tensor to Variable."""
	if torch.cuda.is_available():
		tensor = tensor.cuda()
	return Variable(tensor, volatile=volatile)