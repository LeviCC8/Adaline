import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Adaline import Adaline
from random import uniform


def g(x1, x2):
	return 5*x1 + 4*x2 + 2

X = np.random.rand(100, 2)
Y = [g(x[1], x[0]) + uniform(-0.1, 0.1) for x in X] 

g = Adaline(X, Y)
w = g.solution(realizations = 20, epochs = 100, learningRate = 0.1)
MSE = g.MSE
RMSE = g.RMSE

print("g(x) = 5*x1 + 4*x2 + 2")
print("MSE: {}, standardDeviation: {}".format(MSE["value"], MSE["standardDeviation"]))
print("RMSE: {}, standardDeviation: {}".format(RMSE["value"], RMSE["standardDeviation"]))

fig = plt.figure()
ax = fig.add_subplot(111, projection = "3d")
X1 = np.array([[i, 0] for i in X[: , 0]])
X2 = np.array([[i, 0] for i in X[: , 1]])
G = np.array([[Adaline.predict(w, np.append(-1, x)), 0] for x in X])
Axes3D.scatter(ax, X[: , 0], X[: , 1], Y, c = "r")
Axes3D.plot_surface(ax, X1, X2, G, color = "b")

plt.title("g(x1, x2) = {:.3f}*x1 + {:.3f}*x2 + {:.3f}".format(w[2], w[1], -w[0]))
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("g(x1, x2)")

plt.show()