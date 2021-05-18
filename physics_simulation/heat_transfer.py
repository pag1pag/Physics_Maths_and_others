import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib.animation as animation

tau = 120  # s
M = 62
tc = 120 * 30  # s

N = 60
L = 0.5  # cm
h = L / N

D = 1.15e-4  # m²/s
k = D * tau / (h * h)

Ta = 273 + 20  # K
Tc = 273 + 100  # K


def g(t):
    if t > tc:
        return Tc
    else:
        return Ta


def get_differential_matrix():
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                A[i, j] = 2
            if abs(i - j + 1) == 0:
                A[i, j] = -1
            if abs(i - j - 1) == 0:
                A[i, j] = -1
    A[N - 1, N - 2] = -2
    return A


A = get_differential_matrix()
I = np.identity(N)

Inv = linalg.inv(I + k * A)

# CI
T = np.zeros((N, M))
T[:, 0] = Ta

F = np.zeros((N, M))
F[:, 0] = Ta
F[0, 0] = Ta + k * g(tau)


for j in range(1, M):
    T[:, j] = Inv.dot(F[:, j-1])
    F[:, j] = T[:, j]
    F[0, j] = T[0, j] + k * g(j*(tau+1))



fig = plt.figure() # initialise la figure
line, = plt.plot([],[]) 
plt.xlim(0, L)
plt.ylim(273,373)

# fonction à définir quand blit=True
# crée l'arrière de l'animation qui sera présent sur chaque image
def init():
    line.set_data([],[])
    return line,

x = [h*i for i in range(N)]
def animate(j): 
    t = j * tau
    y = T[:, j]
    line.set_data(x, y)
    return line,
 
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=M, blit=True, interval=100, repeat=False)

plt.show()


# P, L, U = linalg.lu(I + k*A)

# print(P)
# print(L)
# print(U)
