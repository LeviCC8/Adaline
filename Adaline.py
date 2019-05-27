import numpy as np
from random import shuffle
from functools import reduce

class Adaline:

	def __init__(self, realizations, epochs, learningRate):
		self.realizations = realizations
		self.epochs = epochs
		self.learningRate = learningRate
		self.w = []
		self.MSE = {}
		self.RMSE = {}

	def train(self, X, Y, samples):
		totalError = 1
		self.w = np.random.rand(len(X[0]) + 1) 
		for i in range(self.epochs):
			if totalError == 0:
				break
			totalError = 0
			shuffle(samples)
			for j in samples:
				x = np.append(-1, X[j])
				d = Y[j]
				y = self.predict(x)
				error = d - y
				totalError += abs(error)
				self.w = self.w + self.learningRate*error*x

	def predict(self, x):
		y = np.dot(self.w, x)
		return y

	def fit(self, X, Y):
		MSEList = []
		RMSEList = []
		listW = []
		dataNumber = len(X)
		dataNumber08 = int(0.8*dataNumber)
		for i in range(self.realizations):
			lista = np.random.permutation(dataNumber)
			samples = lista[:dataNumber08]
			tests = lista[dataNumber08:]
			self.train(X, Y, samples)
			listW.append(self.w)
			squaredErrorList = [(Y[i] - self.predict(np.append(-1, X[i])))**2 for i in tests]
			MSEi = np.mean(squaredErrorList)
			MSEList.append(MSEi)
			RMSEi = MSEi**(1/2)
			RMSEList.append(RMSEi)
		self.MSE = {"value": np.mean(MSEList), "standardDeviation": np.std(MSEList)}
		self.RMSE = {"value": np.mean(RMSEList), "standardDeviation": np.std(RMSEList)}
		k = reduce(lambda i, j: i if (abs(MSEList[i] - self.MSE["value"]) < abs(MSEList[j] - self.MSE["value"])) else j, range(len(MSEList)))
		greatW = listW[k]
		self.w = greatW