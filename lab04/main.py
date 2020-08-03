import matplotlib.pyplot as plt
import scipy.optimize as sp


# No influence from the atmosphere
def no_atmosphere():
	power_sun = None
	power_earth = None
	albedo = 0.3
	S = 1366  # W/m^2
	earth_surface = 510100000  # km^2
	boltzman_const = 5.67 * 10**-8  # W/(m^2 * K^2)

	power_sun = S * earth_surface / 4 * (1 - albedo)

	earth_temperature = (power_sun / earth_surface / boltzman_const) ** (1/4)
	return earth_temperature


transmission_a = 0.53
transmission_long_a = 0.06

albedo_s = 0.19
albedo_a = 0.30
albedo_long_a = 0.31

S = 1366  # W/m^2
c = 2.7
sigma = 5.67 * 10 ** -8


def equation_a(temperature_a, temperature_s):
	global albedo_s, albedo_a, albedo_long_a
	global transmission_a, transmission_long_a, sigma, S, c
	equation = -transmission_a * (1 - albedo_s) * S/4 + c * (temperature_s - temperature_a)
	equation += sigma * temperature_s**4 * (1 - albedo_long_a) - sigma * temperature_a**4
	return equation


def equation_b(temperature_a, temperature_s):
	global albedo_s, albedo_a, albedo_long_a
	global transmission_a, transmission_long_a, sigma, S, c
	equation = -(1 - albedo_a - transmission_a + albedo_s * transmission_a) * S/4
	equation -= c * (temperature_s - temperature_a)
	equation -= sigma * temperature_s**4 * (1 - transmission_long_a - albedo_long_a)
	equation += 2 * sigma * temperature_a**4
	return equation


def equations(parameters):
	temperature_a, temperature_s = parameters
	return equation_a(temperature_a, temperature_s), equation_b(temperature_a, temperature_s)


def draw(S_range, data_a, data_s, title):
	plt.clf()
	plt.title(title)
	plt.xlabel('S [W/m^2]')
	plt.ylabel('Temperatura [°C]')
	plt.grid()

	plt.plot(S_range, data_a, S_range, data_s)
	plt.legend(['Temperatura atmosfery', 'Temperatura powierzchni'])
	plt.show()


if __name__ == "__main__":
	# 1
	temperature = no_atmosphere()
	print('Model bez atmosfery: \n', temperature - 272.15)

	# 2, 3
	x, y = sp.fsolve(equations, (1, 1))

	print('Model z atmosferą: ')
	print('Temperatura atmosfery: ', x - 272.15, '\nTemperatura powierzchni: ', y - 272.15)

	# 4
	S_range = [S * i / 100 for i in range(80, 120)]
	data_a, data_s = [], []

	for i in S_range:
		S = i
		x, y = sp.fsolve(equations, (1, 1))
		data_a.append(x - 272.15)
		data_s.append(y - 272.15)

	draw(S_range[:], data_a[:], data_s[:], 'Zależność temperatury atmosfery i powierzchni od S')

	# 6
	data_a, data_s = [], []
	reverse = S_range.copy()
	reverse.reverse()

	# Zmiania temperatury - najpierw rośnie, potem maleje
	for i in S_range + reverse:
		S = i
		x, y = sp.fsolve(equations, (1, 1))
		x -= 272.15
		y -= 272.15

		# Temperatura krytyczna
		if y < -5:
			albedo_s = 0.7
		else:
			albedo_s = 0.19

		data_a.append(x)
		data_s.append(y)

	plt.clf()
	plt.title('Zależność temperatury atmosfery i powierzchni z uwzględnieniem\n temperatury krytycznej')
	plt.xlabel('S [W/m^2]')
	plt.ylabel('Temperatura [°C]')
	plt.grid()

	axes_atm_up = plt.scatter(S_range[:], data_a[:len(S_range)], s=30, c='#3b41ff', marker='+')
	axes_atm_down = plt.scatter(reverse[:], data_a[len(S_range):], s=15, c='#ce0aff', marker='x')

	axes_surf_up = plt.scatter(S_range[:], data_s[:len(S_range)], s=30, c='#5bff45', marker='+')
	axes_surf_down = plt.scatter(reverse[:], data_s[len(S_range):], s=15, c='#ffb30f', marker='x')

	axes_atm_up.set_label("Atmosfera (zwiększany S)")
	axes_atm_down.set_label("Atmosfera (zmniejszany S)")
	axes_surf_up.set_label("Powierzchnia (zwiększany S)")
	axes_surf_down.set_label("Powierzchnia (zmniejszany S)")
	plt.legend()

	plt.show()
