import math
import matplotlib.pyplot as plt
import numpy as np


square_size = 50  # rozmiar kwadratu siatki [m]
n = 20
h = 10  # wysokość wylotów kominów [m]
dh = 0
Eg = 1  # maksymalna emisja substancji gazowej [mg/s]
Ep = 1  # maksymalna emisja pyłu zawieszonego [mg/s]
H = h + dh  # efektywna wysokość emitora

# stałe zależne od stanu równowagi atmosfery
m = [0.080, 0.143, 0.196, 0.270, 0.363, 0.440]
a = [0.888, 0.865, 0.845, 0.818, 0.784, 0.756]
b = [1.284, 1.108, 0.978, 0.822, 0.660, 0.551]

Ua_state = [
	{'min': 1, 'max': 3, 'state': 'silnie chwiejna'},
	{'min': 1, 'max': 5, 'state': 'chwiejna'},
	{'min': 1, 'max': 8, 'state': 'lekko chwiejna'},
	{'min': 1, 'max': 11, 'state': 'obojętna'},
	{'min': 1, 'max': 5, 'state': 'lekko stała'},
	{'min': 1, 'max': 4, 'state': 'stała'}
]

state = 0

Ua = Ua_state[state]['min']  # prędkość wiatru na wysokości anemometru (stan równowagi atmosfery)
Uh = Ua * (h / 14) ** m[state]
wind_velocity = Uh  # średnia prędkość wiatru (na wysokości od h do H)
z_o = 1  # średnia szorstkość


def get_horizontal_diffusion(x):
	A = 0.088 * (6 * m[state] ** -0.3 + 1 - math.log(H/z_o, math.e))
	return A * x ** a[state]


def get_vertical_diffusion(x):
	B = 0.38 * m[state] ** 1.3 * (8.7 - math.log(H/z_o, math.e))
	return B * x ** b[state]

# bezo(alfa)piren (równanie 4.2)
def get_concetration_bezo(x, y):  # stężenie substancji w powietrzu uśrednione dla 1h [mikro_g/m**3]
	diff_coeff_y = get_horizontal_diffusion(x)  # współczynnik poziomej dyfuzji
	diff_coeff_z = get_vertical_diffusion(x)  # współczynnik pionowej dyfuzji

	output = Eg / (math.pi * wind_velocity * diff_coeff_y * diff_coeff_z)
	output *= math.exp(-y**2/(2 * diff_coeff_y**2))
	output *= math.exp(-H**2/(2*diff_coeff_z**2))*1000
	return output


# pył zawieszony (równanie 4.6)
def get_concetration_dust(x, y):
	diff_coeff_y = get_horizontal_diffusion(x)  # współczynnik poziomej dyfuzji
	diff_coeff_z = get_vertical_diffusion(x)  # współczynnik pionowej dyfuzji

	output = Ep / (2 * math.pi * wind_velocity * diff_coeff_y * diff_coeff_z)
	output *= math.exp(-y ** 2 / (2 * diff_coeff_y ** 2))
	output *= math.exp(-H ** 2 / (2 * diff_coeff_z ** 2)) * 1000
	return output


def interpolate(output):
	final = np.zeros((2 * n, 2 * n), dtype=float)

	for i in range(1, 2 * n - 1):
		for j in range(1, 2 * n - 1):
			if output[i][j] == 0:
				avg = math.fsum((output[i][j+1], output[i][j-1], output[i+1][j], output[i-1][j]))
				final[i][j] = avg / 4
			else:
				final[i][j] = output[i][j]

	return final


def adjust(S):
	output = np.zeros((2 * n, 2 * n), dtype=float)

	k, l = 1, 0

	for i in range(n):
		k_in = k
		l_in = l
		for j in range(i + 1):
			output[k_in][l_in] = S[i][j]
			k_in -= 1
			l_in += 1

		k += 1
		l += 1

	final = np.zeros((n, n), dtype=float)

	plt.pcolormesh(range(2*n), range(2*n), output[range(0, 2*n - 1), :][:, range(0, 2*n - 1)], shading='flat')
	plt.show()

	# Make grid
	for i in range(n - 1):
		for j in range(0, n - 1):
			final[i][j] = output[2 * i + 2][2 * j + 1]

	# Rotate to match NE
	final = np.flip(final, axis=1)

	size = int(n / 2) + 2
	plt.pcolormesh(range(size), range(size), final[range(0, size-1), :][:, range(1, size)], shading='flat')
	plt.show()

	# Mirror simetria along wind direction
	for i in range(1, n - 1):
		for j in range(i, n - 1):
			final[j][i] = final[i - 1][j + 1]

	return final


if __name__ == "__main__":

	# 6 stanów równowagi
	for sym in range(1):
		# SYMULACJA
		Sbezo = np.zeros((n, n), dtype=float)
		Sdust = np.zeros((n, n), dtype=float)
		state = sym

		for i in range(n):
			for j in range(n):
				Sbezo[i][j] = get_concetration_bezo(i + 1, j + 1)
				#Sdust[i][j] = get_concetration_dust(i + 1, j + 1)

		# WYKRES, USTALANIE KIERUNKU WIATRU
		Sbezo_final = adjust(Sbezo)
		size = int(n/2) + 2
		plt.pcolormesh(range(1, size), range(0, size-1), Sbezo_final[range(0, size-1), :][:, range(1, size)], shading='flat')
		plt.show()



