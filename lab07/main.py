import math
import matplotlib.pyplot as plt
import numpy as np


T_hf = 148  # mouths, half-life of tritium
lam = math.log(2, math.e) / T_hf
tt = 1.0
Pe = 1.0
C_in = []
out = []


def tlok(t: float):
	output = 0
	if t - tt is 0:
		output = 1
	return output


def expo(t: float):
	global tt
	return math.exp(-t / tt) / tt


def dysp(t: float):
	global tt
	out = (4 * math.pi * Pe * t / tt) ** -1/2 * 1 / t
	out *= math.exp(-((1 - t/tt)**2 / (4 * Pe * t / tt)))
	return out


def read_files():
	global C_in, out
	file_c = open('opady.prn', 'r')
	file_dunaj = open('dunaj.prn', 'r')
	inputs_dict = {
		file_c: C_in,
		file_dunaj: out
	}

	for input_file in inputs_dict:
		for line in input_file:
			temp = line.split(' ')
			temp = list(filter(lambda elem: elem != '', temp))
			inputs_dict[input_file].append(float(temp[1]))


def calculate_integral_rect(t: "month (int)", g: "response function") -> float:
	sum = 0
	dt = 1
	for t_prim in range(t):
		sum += C_in[t_prim] * g(t - t_prim) * math.exp(-lam * (t - t_prim)) * dt

	return sum


def get_fit_coeff(data: list) -> float:
	global out
	n = len(out)
	rests = [(data[i] - out[i]) ** 2 for i in range(n)]

	return math.fsum(rests) / n


def reverse_model(tt_range: list, Pe_range: list, g: "response function", integral_method: "model calculating integral (function)"):
	global tt, Pe

	i, j = 0, 0
	fit_array = np.zeros((len(tt_range), len(Pe_range)), dtype=float)

	for tt_value in tt_range:
		tt = tt_value
		j = 0
		for Pe_value in Pe_range:
			Pe = Pe_value
			data = simulate(g, integral_method)
			fit_coef = get_fit_coeff(data)
			fit_array[i][j] = fit_coef
			j += 1
		i += 1

	return fit_array


def simulate(g: "response function", integral_method: "model calculating integral (function)"):
	data = []
	for month in range(len(C_in)):
		data.append(integral_method(month, g))

	return data


def draw(data: list, model: str, tt_value: "tt used in simulation (str)", Pe_value: "Pe used in simulation (str)"):
	global C_in, out
	plt.title('Tryt w Dunaju, ' + model + '\nśredni czas przebywania = ' + tt_value + ', wartość Pe = ' + Pe_value)
	plt.scatter(range(len(out)), out, label='observations', s=2)
	plt.plot(range(len(C_in)), data, label='model', c='orange')

	plt.xlabel('Time [months]')
	plt.ylabel('Tritium content [TU]')
	plt.grid()
	plt.legend()

	plt.show()


if __name__ == "__main__":

	read_files()

	# 1, 2, 3, 4, 5 - dobór tt metodą prób i błędów, zestawienie z opadami
	tt = 110
	Pe = 0.1
	data = simulate(dysp, calculate_integral_rect)

	plt.title('Tryt w Dunaju, model tłokowy\nśredni czas przebywania = ' + str(tt) + ', wartość Pe = ' + str(Pe))
	plt.scatter(range(len(out)), out, label='observations', s=2)
	plt.plot(range(len(C_in)), data, label='model', c='orange')
	#plt.plot(range(len(C_in)), C_in, label='rainfall', c='green')

	plt.xlabel('Time [months]')
	plt.ylabel('Tritium content [TU]')
	plt.grid()
	plt.legend()

	plt.show()

	# 6
	g = dysp
	tt_range = np.arange(1, 11, 2, dtype=float)
	Pe_range = np.arange(0.1, 2.6, 0.5, dtype=float)
	fit = reverse_model(tt_range, Pe_range, g, calculate_integral_rect)
	minimum = fit.min()

	i, j = np.where(fit == minimum)
	tt, Pe = tt_range[i[0]], Pe_range[j[0]]
	data = simulate(g, calculate_integral_rect)
	draw(data, 'model dyspersyjny', str(round(tt, 2)), str(round(Pe, 3)))
	print('tt = ' + str(tt) + ', Pe = ' + str(Pe) + ', odch = ' + str(minimum))

	tts, Pes = np.meshgrid(tt_range, Pe_range)

	from matplotlib import cm

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.set_title('RMSE dla dobranych wartości Liczby Pecleta\ni średniego czasu przebywania')
	ax.set_xlabel('Średni czas przebywania')
	ax.set_ylabel('Liczba Pecleta')
	ax.set_zlabel('Odchylenie RMSE')

	surf = ax.plot_surface(tts, Pes, fit.transpose(), cmap=cm.coolwarm, linewidth=0, antialiased=False)
	fig.colorbar(surf, shrink=0.5, aspect=5)
	plt.show()

	Z = fit[:, :]

	plt.imshow(Z, origin='lower', interpolation='bilinear', extent=[1, 11, 0.1, 2.6])
	plt.colorbar()
	plt.title("Siatka średniego czasu przebywania i liczby Pecleta")
	plt.xlabel("tt")
	plt.ylabel("Pe")

	plt.show()

	fig, ax = plt.subplots()
	plt.pcolor(tt_range, Pe_range, Z, cmap='viridis')
	plt.colorbar()
	plt.title("Siatka średniego czasu przebywania i liczby Pecleta")
	plt.xlabel("tt")
	plt.ylabel("Pe")

	plt.show()

