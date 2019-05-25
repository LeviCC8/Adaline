import numpy as np
from random import shuffle
from functools import reduce

class Adaline:

	def __init__(self, X, Y):
		self.__X = X
		self.__Y = Y
		self.__dataNumber = len(X)
		self.__dataNumber08 = int(0.8*self.__dataNumber)
		self.MSE = {}
		self.RMSE = {}

	def train(self, samples, epochs, learningRate):
		totalError = 1
		w = np.random.rand(len(self.__X[0]) + 1) 
		for i in range(epochs):
			if totalError == 0:
				break
			totalError = 0
			shuffle(samples)
			for j in samples:
				x = np.append(-1, self.__X[j])
				d = self.__Y[j]
				y = self.predict(w, x)
				error = d - y
				totalError += abs(error)
				w = w + learningRate*error*x
		return w

	@staticmethod
	def predict(w, x):
		y = np.dot(w, x)
		return y

	def solution(self, realizations, epochs, learningRate):
		MSEList = []
		RMSEList = []
		listW = []
		for i in range(realizations):
			lista = np.random.permutation(self.__dataNumber)
			samples = lista[:self.__dataNumber08]
			tests = lista[self.__dataNumber08:]
			w = self.train(samples, epochs, learningRate)
			listW.append(w)
			squaredErrorList = [(self.__Y[i] - self.predict(w, np.append(-1, self.__X[i])))**2 for i in tests]
			MSEi = np.mean(squaredErrorList)
			MSEList.append(MSEi)
			RMSEi = MSEi**(1/2)
			RMSEList.append(RMSEi)
		self.MSE = {"value": np.mean(MSEList), "standardDeviation": np.std(MSEList)}
		self.RMSE = {"value": np.mean(RMSEList), "standardDeviation": np.std(RMSEList)}
		j = reduce(lambda i, j: i if (abs(MSEList[i] - self.MSE["value"]) < abs(MSEList[j] - self.MSE["value"])) else j, range(len(MSEList)))
		greatW = listW[j]
		return greatW