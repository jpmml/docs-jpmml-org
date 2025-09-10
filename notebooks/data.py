from abc import ABC, abstractmethod
from ucimlrepo import fetch_ucirepo

class Dataset(ABC):

	def __init__(self, ds):
		self.ds = ds

	def getData(self):
		X = self.getFeatures()
		y = self.getTargets()

		return (X, y)

	def getFeatures(self):
		return self.ds.data.features

	def getTargets(self):
		return self.ds.data.targets

	@abstractmethod
	def getCategoricalColumns(self):
		raise NotImplementedError()

	@abstractmethod
	def getContinuousColumns(self):
		raise NotImplementedError()

	def getFeatureColumns(self, types):
		variables = self.ds.variables

		role_mask = variables["role"] == "Feature"
		type_mask = variables["type"].isin(types)

		return variables[role_mask & type_mask]["name"].values

def fetch_bank_marketing():
	class BankMarketingDataset(Dataset):

		def __init__(self):
			super().__init__(ds = fetch_ucirepo(id = 222))

		def getCategoricalColumns(self):
			return self.getFeatureColumns(["Categorical"])

		def getContinuousColumns(self):
			return self.getFeatureColumns(["Integer"])

	return BankMarketingDataset()
