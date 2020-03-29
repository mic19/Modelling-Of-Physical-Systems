import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats

# 1
steps = 1000
particles = 1000

points = np.zeros((particles, steps, 2), dtype='float64')
distances = np.zeros((particles, steps))
avsq = np.zeros(steps)

for i in range(1, steps):
	for j in range(0, particles):
		movement = np.random.normal(size=2)
		points[j][i] = points[j][i - 1] + movement
		points[j][i] = points[j][i - 1] + movement

		distances[j][i] = math.hypot(*movement)
		avsq[i] += points[j][i][0]**2 + points[j][i][1]**2
	avsq[i] /= particles

# 2 - trajektoria dla 10, 100, 1000
for i in range(particles):
	plt.plot(points[i][:, 0], points[i][:, 1], linewidth=0.5)

plt.xlabel('x')
plt.ylabel('y')

plt.title("Trajektoria dla " + str(particles) + " cząstek")
plt.legend()
plt.savefig('images/trajektoria.png')
plt.show()

# 3 - zaleznosc sredniego kwadratu polozenia czastki od czasu
plt.plot(range(0, steps), avsq)
plt.xlabel('t')
plt.ylabel('r')
plt.title("Średni kwadrat położnia cząstki dla " + str(particles) + " cząstek")
plt.legend()
plt.savefig('images/sredni_kwadrat.png')
plt.show()

# 4 - liczba czastek dla ktorej dopasowanie do prostej spelnia: R2 - 0.999
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(range(0, steps), avsq)
print(r_value)

# 5 - liczba czastek przypadajaca na jednostke powierzchni
all_x_points = []
all_y_points = []
for i in range(particles):
	all_x_points = np.append(all_x_points, points[i, :, 0])
	all_y_points = np.append(all_y_points, points[i, :, 1])

plt.hist2d(all_x_points, all_y_points, (50, 50), cmap=plt.cm.jet)
plt.title("Rozkład gęstości cząstek dla " + str(particles) + " cząstek \ni " + str(steps) + " kroków czasowych")
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.savefig('images/histogram.png')
plt.show()

# 6 - wspolczynik dyfuzji
D = avsq[steps - 1]/4
print(D)
