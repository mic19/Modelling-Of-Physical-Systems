import numpy as np
import matplotlib.pyplot as plt
from typing import List


L_in = 10  # m
L_out = 90  # m
length = 1000
A = 5  # m^2
U = 0.2  # m/s
D = 0.5  # m^2/s
m = 10  # kg
dx = 1
dt = 1
n = 10
time_steps = 1000
c_in = m/A/dx/n
ca = U * dt / dx
cd = D * dt / dx ** 2


def simulate_quickest() -> (List[float], float):
	ca = U * dt / dx
	cd = D * dt / dx ** 2

	c = np.zeros(length)
	c_next = c.copy()

	data = []
	mass = 0

	# 1
	for t in range(time_steps):
		if t < n:
			c[L_in] = c_in
		else:
			c[L_in] = 0

		for j in range(2, length - 1):
			c_next[j] = c[j] + (cd * (1 - ca) - ca / 6 * (ca ** 2 - 3 * ca + 2)) * c[j + 1]
			c_next[j] -= (cd * (2 - 3 * ca) - ca / 2 * (ca ** 2 - 2 * ca - 1)) * c[j]
			c_next[j] += (cd * (1 - 3 * ca) - ca / 2 * (ca ** 2 - ca - 2)) * c[j - 1]
			c_next[j] += (cd * ca + ca / 6 * (ca ** 2 - 1)) * c[j - 2]

		data.append(c[L_out])
		mass += c[L_out]
		c, c_next = c_next, c

	return data, mass


def initial_matrixes():
	AA = np.zeros((length, length), float)
	BB = np.zeros((length, length), float)

	for i in range(length):
		AA[i][i] = 1 + cd
		BB[i][i] = 1 - cd

	for i in range(length - 1):
		AA[i + 1][i] = -cd/2 - ca/4
		AA[i][i + 1] = -cd/2 + ca/4
		BB[i + 1][i] = cd/2 + ca/4
		BB[i][i + 1] = cd/2 - ca/4

	return AA, BB


def simulate_cn() -> (List[float], float):
	AA, BB = initial_matrixes()
	AB = np.linalg.matrix_power(AA, -1).dot(BB)
	c = np.zeros((length, 1))
	c_next = c.copy()
	data = []
	mass = 0

	for t in range(time_steps):
		if t < n:
			c[L_in][0] = c_in
		else:
			c[L_in][0] = 0

		data.append(c[L_out][0])
		mass += c[L_out][0]
		c_next = AB.dot(c)
		c, c_next = c_next, c

	return data, mass


if __name__ == "__main__":

	n = 10
	data_10, mass_10 = simulate_quickest()
	n = 1
	c_in = m / A / dx / n
	data_1, mass_1 = simulate_quickest()

	print('Masa początkowa: ', m)
	print('Masa w punkcie L_out dla n = 10: ', mass_10 * A * dx)
	print('Masa w punkcie L_out dla n = 1: ', mass_1 * A * dx)

	# 2
	plt.title('Krzywa przejścia zanieczyszczenia - QUICKEST')
	plt.xlabel('t [s]')
	plt.ylabel('c [kg/m^3]')
	plt.grid()

	plt.plot(range(time_steps), data_10, label='n = 10')
	plt.plot(range(time_steps), data_1, label='n = 1')
	plt.legend()

	plt.show()

