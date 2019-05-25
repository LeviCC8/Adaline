import numpy as np
import matplotlib.pyplot as plt
from Adaline import Adaline
from random import uniform


def f(x):
	return 3*x + 2

X = np.random.rand(100, 1)
Y = [f(x) + uniform(-0.1, 0.1) for x in X] 

f = Adaline(X, Y)
w = f.solution(realizations = 20, epochs = 100, learningRate = 0.1)
MSE = f.MSE
RMSE = f.RMSE

print("f(x) = 3*x + 2")
print("MSE: {}, standardDeviation: {}".format(MSE["value"], MSE["standardDeviation"]))
print("RMSE: {}, standardDeviation: {}".format(RMSE["value"], RMSE["standardDeviation"]))

fig = plt.figure()
F = [Adaline.predict(w, np.append(-1, x)) for x in X]
plt.plot(X, Y, "r.", X, F, "k")
plt.title("f(x) = {:.3f}*x + {:.3f}".format(w[1], -w[0]))
plt.xlabel("x")
plt.ylabel("f(x)")

plt.show()