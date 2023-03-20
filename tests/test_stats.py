from flaregun.flaregun import ModelStats
import torch

def test_model():
	class Model(torch.nn.Module):
		def __init__(self, input, hidden, output):
			super().__init__()
			self.linear1 = torch.nn.Linear(input, hidden)
			self.activation = torch.nn.ReLU()
			self.linear2 = torch.nn.Linear(hidden, hidden)
			self.linear3 = torch.nn.Linear(hidden, output)
			self.softmax = torch.nn.Softmax()

	model = Model(1, 10, 1)
	assert ModelStats(model).total() == 1 * 10 + 10 + 10 * 10 + 10 + 10 * 1 + 1

	model = Model(100, 200, 10)
	assert ModelStats(model).total() == 100 * 200 + 200 + 200 * 200 + 200 + 200 * 10 + 10
