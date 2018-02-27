""" Best model for source test """
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

class LeNetEncoder(nn.Module):
	""" LeNet encoder for ADDA """
	def __init__(self):
		super(LeNetEncoder, self).__init__()
		self.restored = False
		self.encoder = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=11, stride=3),
			nn.MaxPool2d(kernel_size=3),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=3),
			nn.MaxPool2d(kernel_size=2),
			nn.ReLU()
		)
		self.fc1 = nn.Linear(64 * 11 * 20, 1024)

	def forward(self, input):
		""" Forward the encoder """
		out = self.encoder(input)
		out = self.fc1(out.view(-1, 64 * 11 * 20))
		return out

class LeNetClassifier(nn.Module):
	""" LeNet classifier for ADDA """
	def __init__(self):
		super(LeNetClassifier, self).__init__()
		self.fc2 = nn.Linear(1024, 6)
		# self.softmax = nn.Softmax()

	def forward(self, input):
		""" Forward the classifier """
		out = F.dropout(F.relu(input), training=self.training)
		out = self.fc2(out)
		return out

if __name__ == '__main__':
	net = LeNetEncoder()
	classifier = LeNetClassifier()
	x = Variable(torch.ones(30, 1, 228, 400))
	out = net(x)
	out = classifier(out)
	print (out)
