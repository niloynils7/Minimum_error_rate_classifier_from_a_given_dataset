import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
graph = Axes3D(fig)

test_data = pd.read_csv('test.txt', header=None, sep=',', dtype='float64')
test_arr = test_data.values

d = 2
mu1 = [0, 0]
sigma1 = [[.25, .3], [.3, 1]]

mu2 = [2, 2]
sigma2 = [[0.5, 0], [0, 0.5]]


def dist_function(mu, sigma, x):
    Is = np.linalg.inv(sigma)
    Ds = np.linalg.det(sigma)
    return (2 * np.pi) ** (-d / 2) * Ds ** (-.5) * np.exp(- 0.5 * np.matmul(np.matmul(x - mu, Is), x - mu))


class_A = []
class_B = []
for x in test_arr:
    g1 = dist_function(mu1, sigma1, x)
    g2 = dist_function(mu2, sigma2, x)
    if g1 > g2:
        class_A.extend([x])
    elif g1 < g2:
        class_B.extend([x])
class_A = np.array(class_A)
class_B = np.array(class_B)
     
graph.scatter(class_A[:,0], class_A[:,1], label='class A', c='r', marker='o')
graph.scatter(class_B[:,0], class_B[:,1], label='class B', c='b', marker='*')

x = np.arange(-6, 6, 0.05)
y = np.arange(-6, 6, 0.05)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)
L = np.zeros_like(X)
for i in range(len(x)):
    for j in range(len(y)):
        a = dist_function(mu1, sigma1, np.array([x[i], y[j]]))
        b = dist_function(mu2, sigma2, np.array([x[i], y[j]]))
        if a>b:
            Z[j][i] = a
        else:
            Z[j][i] = b
        L[j][i] = a-b
        
color = 'hsv'
graph.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True, cmap=color, alpha=0.8)
graph.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=color, alpha=0.8)
graph.contour3D(X, Y, L, zdir='z', offset=-0.15, cmap='Greens')
graph.set_zlim(-0.15,0.2)
graph.set_zticks(np.linspace(0,0.2,5))

graph.set_xlabel('X axis')
graph.set_ylabel('Y axis')
graph.set_zlabel('Probability Density')
graph.legend()
graph.view_init(40, 240)

plt.show()
