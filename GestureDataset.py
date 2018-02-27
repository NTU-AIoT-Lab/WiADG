from torch.utils import data as data_utils
import torch

class GestureDataset(torch.utils.data.Dataset):
	"""docstring for GestureDataset"""
	def __init__(self, X, Y):
		super(GestureDataset, self).__init__()
		self.X = X
		self.Y = Y

	def __getitem__(self, idx):
		x, y = self.X[idx], self.Y[idx]
		x, y = torch.from_numpy(x), torch.from_numpy(y)
		return x, y

	def __len__(self):
		return len(self.X)