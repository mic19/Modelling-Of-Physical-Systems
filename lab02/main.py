import numpy as np
import matplotlib.pyplot as plt
from enum import Enum


# Warunek na roznice temperatur w kolejnych krokach czasowych
def get_avg_temp(matrix):
	sum = 0

	for i in matrix:
		for j in i:
			sum += j

	return sum / len(matrix) / len(matrix[0])


def is_next_to_heater(x_heater, y_heater, heater_size, x, y):
	if x == x_heater - heater_size:
		if y > y_heater - heater_size and y < y_heater + heater_size:
			return True

	if x == x_heater + heater_size:
		if y > y_heater - heater_size and y < y_heater + heater_size:
			return True

	if y == y_heater - heater_size:
		if x > x_heater - heater_size and x < x_heater + heater_size:
			return True

	if y == y_heater + heater_size:
		if x > x_heater - heater_size and x < x_heater + heater_size:
			return True

	return False


class Condition(Enum):
	Dirichlet = 1
	Neumann = 2


def lab02(temp_heater = 80, temp_blade = 20, cond = Condition.Dirichlet, file_name="", q = None):
	x_size , y_size = 80, 80
	a = 0.1
	b = 0.02
	h = 0.002

	matrix = np.full((x_size, y_size), temp_blade, dtype='float64')
	matrix_next = matrix.copy()

	# polozenie grzalki w blaszce
	x_heater = int(x_size / 2)
	y_heater = int(y_size / 2)

	heater_size = int(b / a * x_size)

	# rowananie
	time = 0
	delta_t = 0.01
	delta_x = a / x_size
	delta_y = a / y_size
	delta_temp = 100
	K = 110 #110 W/(m * K) mosiadz
	density = 8700 #8700 kg/m**3 mosiadz
	c_w = 400 #400J/kg * C mosiadz
	count = 0

	temp_prev = None
	temp_curr = get_avg_temp(matrix)

	while delta_temp > 0.01:
		for i in range(heater_size):
			for j in range(heater_size):
				matrix[x_heater + i, y_heater + j] = temp_heater
				matrix[x_heater - i, y_heater + j] = temp_heater
				matrix[x_heater + i, y_heater - j] = temp_heater
				matrix[x_heater - i, y_heater - j] = temp_heater


		for i in range(1, x_size - 1):
			for j in range(1, y_size - 1):
				matrix_next[i][j] = matrix[i][j]
				matrix_next[i][j] += K * delta_t / c_w / density / delta_x ** 2 * (matrix[i+1][j] - 2 * matrix[i][j] + matrix[i-1][j])
				matrix_next[i][j] += K * delta_t / c_w / density / delta_y ** 2 * (matrix[i][j+1] - 2 * matrix[i][j] + matrix[i][j-1])

				if q is not None and is_next_to_heater(x_heater, y_heater, heater_size, i, j):
					matrix_next[i][j] += q / K / h / density

		if cond == Condition.Neumann:
			for k in range(x_size):
				matrix_next[k][0] = matrix_next[k][1]
				matrix_next[k][-1] = matrix_next[k][-2]#

			for k in range(y_size):
				matrix_next[0][k] = matrix_next[1][k]
				matrix_next[-1][k] = matrix_next[-2][k]#

		matrix, matrix_next = matrix_next, matrix

		time += delta_t
		count += 1

		if count % 1 == 0:
			temp_prev = temp_curr
			temp_curr = get_avg_temp(matrix)
			delta_temp = abs(temp_curr - temp_prev)

	# rysowanie
	Z = matrix[:, :]

	plt.imshow(Z, origin='lower', interpolation='bilinear')
	plt.colorbar()
	plt.title("Rozkład temperatury w chwili czasowej t = " + str(round(time, 2)) + "s")
	plt.xlabel("x")
	plt.ylabel("y")

	plt.savefig(file_name)
	plt.show()


# A
lab02(temp_heater=80, temp_blade=20, cond=Condition.Dirichlet, file_name="output_dirichlet.png")

# B
lab02(temp_heater=80, temp_blade=20, cond=Condition.Neumann, file_name="output_neumann.png")

# C
lab02(temp_heater=80, temp_blade=20, cond=Condition.Dirichlet, file_name="output_C.png", q=1000)
