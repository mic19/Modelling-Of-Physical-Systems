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
gif_flag = False


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

	ca = U * dt / dx
	cd = D * dt / dx ** 2

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


def task_plot(data_10, data_1, data_10_cn, data_1_cn):
	# QUICKEST
	plt.title('Krzywa przejścia zanieczyszczenia - QUICKEST')
	plt.xlabel('t [s]')
	plt.ylabel('c [kg/m^3]')
	plt.grid()

	plt.plot(range(time_steps), data_10, label='QUICKEST n = 10')
	plt.plot(range(time_steps), data_1, label='QUICKEST n = 1')
	plt.legend()
	plt.show()

	# CRANK-NICHOLSON
	plt.title('Krzywa przejścia zanieczyszczenia - metoda Cranka-Nicholsona')
	plt.xlabel('t [s]')
	plt.ylabel('c [kg/m^3]')
	plt.grid()

	plt.plot(range(time_steps), data_10_cn, label='CRANK-NICHOLSON n = 10')
	plt.plot(range(time_steps), data_1_cn, label='CRANK-NICHOLSON n = 1')
	plt.legend()
	plt.show()

	# BOTH
	plt.title('Krzywa przejścia zanieczyszczenia')
	plt.xlabel('t [s]')
	plt.ylabel('c [kg/m^3]')
	plt.grid()

	plt.plot(range(time_steps), data_10, label='QUICKEST n = 10')
	plt.plot(range(time_steps), data_1, label='QUICKEST n = 1')
	plt.plot(range(time_steps), data_10_cn, label='CRANK-NICHOLSON n = 10', ls='dashed', )
	plt.plot(range(time_steps), data_1_cn, label='CRANK-NICHOLSON n = 1', ls='dashed')
	plt.legend()

	plt.show()


if __name__ == "__main__":

	# 1 WYKONANIE SYMULACJI
	n = 100
	c_in = m / A / dx / n
	data_10, mass_10 = simulate_quickest()
	data_10_cn, mass_10_cn = simulate_cn()
	n = 1
	c_in = m / A / dx / n
	data_1, mass_1 = simulate_quickest()
	data_1_cn, mass_1_cn = simulate_cn()

	print('Masa początkowa: ', m)
	print('QUICKEST Masa w punkcie L_out dla n = 10: ', mass_10 * A * dx)
	print('QUICKEST Masa w punkcie L_out dla n = 1: ', mass_1 * A * dx)
	print('CRANK-NICHOLSON Masa w punkcie L_out dla n = 10: ', mass_10_cn * A * dx)
	print('CRANK-NICHOLSON Masa w punkcie L_out dla n = 1: ', mass_1_cn * A * dx)

	# 2 PRZEDSTAWIENIE KRZYWEJ PRZEJŚCIA
	task_plot(data_10, data_1, data_10_cn, data_1_cn)

	# 3 ZMIANA PARAMETRÓW U I D
	print('-- U D --')
	U = 0.2  # m/s
	D = 0.5  # m^2/s
	n = 10
	step = 0.01
	gif_data = []

	if gif_flag is True:
		for i in range(int(0.2 / step) + 1):
			D = 0.6 + step * i
			data_10, mass_10 = simulate_quickest()
			data_10_cn, mass_10_cn = simulate_cn()

			plt.clf()
			plt.title('Krzywa przejścia zanieczyszczenia - U = 0.2, D = ' + str(round(D, 3)) + '\nn = 10')
			plt.xlabel('t [s]')
			plt.ylabel('c [kg/m^3]')
			plt.ylim(0, 0.009)
			plt.grid()

			plt.plot(range(time_steps), data_10, label='QUICKEST masa L_out = ' + str(round(mass_10 * A * dx, 3)))
			plt.plot(range(time_steps), data_10_cn, label='CRANK-NICHOLSON masa L_out = ' + str(round(mass_10_cn * A * dx, 3)))
			plt.legend()

			name = 'output/d_change/gif' + str(i) + '.png'
			plt.savefig(name, dpi=80)
			gif_data.append(name)

		D = 0.5
		step = 0.1
		for i in range(int(2 / step) + 1):
			U = 0.2 + step * i
			data_10, mass_10 = simulate_quickest()
			data_10_cn, mass_10_cn = simulate_cn()

			plt.clf()
			plt.title('Krzywa przejścia zanieczyszczenia - U = ' + str(round(U, 3)) + ', D = 0.5\nn = 10')
			plt.xlabel('t [s]')
			plt.ylabel('c [kg/m^3]')
			plt.ylim(0, 0.007)
			plt.grid()

			plt.plot(range(time_steps), data_10, label='QUICKEST masa L_out = ' + str(round(mass_10 * A * dx, 3)))
			plt.plot(range(time_steps), data_10_cn, label='CRANK-NICHOLSON masa L_out = ' + str(round(mass_10_cn * A * dx, 3)))
			plt.legend()

			name = 'output/u_change/gif' + str(i) + '.png'
			plt.savefig(name, dpi=80)
			gif_data.append(name)


