import math
import numpy as np
import matplotlib.pyplot as plt


length = 100  # Długość rzeki [m]
width = 3  # Szerokość rzeki [m]
dx = 0.1
size = int(length / dx)  # Rozmiar siatki
angle = 0.01  # Nachylenia rzeki [%]
xt = int(10 / dx)  # Położenie tamy (10 metr)
ht = 10  # Wysokość tamy
h0 = 2  # Poziom wody w rzece
t = 0

time = 40  # [s]
dt = 1
g = 9.81
fdw = 40

h = np.zeros(size, dtype=float)
T = np.zeros(size, dtype=float)
T.fill(width)

Q = np.zeros(size, dtype=float)
Q_next = Q.copy()
Q0 = 1

V = np.zeros(size, dtype=float)
S = np.zeros(size, dtype=float)
S.fill(angle)
R = np.zeros(size, dtype=float)

h[0: xt] = ht
h[xt: size] = h0
A = h * T

level_10_over_time = []
index10 = int(10 / dx)
level_20_over_time = []
index20 = int(20 / dx)
level_50_over_time = []
index50 = int(50 / dx)
time_list = []


def draw_level(time: float, counter: int):
	plt.clf()
	plt.title('Poziom wody w całej długości rzeki w chwili ' + str("%.2f" % round(time, 2)) + 's')
	plt.plot(np.arange(0, length, dx), h, color='blue')
	plt.xlabel('Odległość [m]')
	plt.ylabel('Poziom wody [m]')
	plt.ylim(0, 1.2 * ht)
	plt.grid()

	filename = 'output/out' + str(counter) + '.png'
	plt.savefig(filename, dpi=80)

	if counter % 50 == 0:
		plt.pause(0.1)


np.seterr('raise')


if __name__ == "__main__":

	iteration = 0
	draw_level(0, 0)
	level_10_over_time.append(h[index10])
	level_20_over_time.append(h[index20])
	level_50_over_time.append(h[index50])
	time_list.append(t)

	while t < time:
		maks = abs(V[0] + math.sqrt(g * h[0]))
		for i in range(1, size):
			candidate = abs(V[i] + math.sqrt(abs(g * h[i])))
			if candidate > maks:
				maks = candidate
		for i in range(0, size):
			candidate = abs(V[i] - math.sqrt(abs(g * h[i])))
			if candidate > maks:
				maks = candidate

		dt = dx/maks
		t = t + dt

		for i in range(0, size - 2):
			h[i] = 2 * (-dt) / (T[i] + T[i + 1]) * (Q[i + 1] - Q[i]) / dx + h[i]

			A[i] = T[i] * h[i]
			R[i] = A[i]/(T[i] + 2 * h[i])
			V[i] = Q[i]/A[i]

		h[size - 1] = h[size - 2]

		for i in range(1, size - 2):
			temp = ((math.pow(Q[i + 1], 2) / A[i + 1]) - (math.pow(Q[i - 1], 2) / A[i - 1])) / (2 * dx)
			temp += (fdw/8/R[i] * abs(V[i]) * Q[i]) + g*A[i] * (((h[i] - h[i - 1]) / dx) - S[i])
			temp = temp * (-dt) + Q[i]
			Q[i] = temp

		Q[size - 1] = Q[size - 2]
		iteration += 1

		draw_level(t, iteration)
		level_10_over_time.append(h[index10])
		level_20_over_time.append(h[index20])
		level_50_over_time.append(h[index50])
		time_list.append(t)

	plt.clf()
	plt.title('Poziom wody 10, 20 i 50 m od tamy w funkcji czasu')
	plt.plot(time_list, level_10_over_time, color='cyan', label='10 m')
	plt.plot(time_list, level_20_over_time, color='blue', label='20 m')
	plt.plot(time_list, level_50_over_time, color='magenta', label='50 m')
	plt.legend()
	plt.xlabel('Czas [s]')
	plt.ylabel('Poziom wody [m]')
	plt.grid()
	plt.show()

