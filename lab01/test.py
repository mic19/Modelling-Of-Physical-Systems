import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import linregress

def randNum():
    return np.random.normal(0, 1)

def rSquared(x, y):
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return r_value ** 2

numOfSteps = 100
numOfParticles = 100

x = 0
y = 0

dx = 0
dy = 0

x_coor_all = []
y_coor_all = []

dx_values = []
dy_values = []

for i in range(numOfParticles):
    x = 0
    y = 0
    x_coor = []
    y_coor = []
    dx_all = []
    dy_all = []
    for j in range(0, numOfSteps):
        dx = randNum()
        dy = randNum()
        x += dx
        y += dy

        x_coor.append(x)
        y_coor.append(y)
        dx_all.append(dx)
        dy_all.append(dy)

    x_coor_all.append(x_coor)
    y_coor_all.append(y_coor)
    dx_values.append(dx_all)
    dy_values.append(dy_all)

plt.figure(1)
for i in range(numOfParticles):
    plt.plot(x_coor_all[i], y_coor_all[i])
plt.show()

# Średni kwadrat położenia po n krokach czasowych dla m cząstek
r2 = []

for i in range(numOfParticles):
    x_all = x_coor_all[i]
    y_all = x_coor_all[i]
    r = 0
    steps_coor = []
    for j in range(0, len(x_all)):
        # Kwadrat położenia dla pojedynczego punktu cząstki
        sum_squared = np.power(x_all[j], 2) + np.power(y_all[j], 2)
        # Dodawanie elementu do wektora położeń
        steps_coor.append(sum_squared)
    r2.append(steps_coor)

# obliczanie średniej dla wszystkich cząstek
r2 = np.asmatrix(r2)
r2 = np.mean(r2, axis=0)
r2 = np.asarray(r2)
r2 = r2.tolist()

t = []

# dziedzina czasu
for i in range(0, numOfSteps):
    t.append(i)

# rysowanie
plt.figure(2)
plt.plot(t, r2[0])
plt.show()

dr2 = []

# # Średni kwadrat przemieszczen po n krokach czasowych dla 1 czastki
# for i in range(numOfParticles):
#     dx_all = dx_values[i]
#     dy_all = dy_values[i]
#     r = 0
#     steps_coor = []
#     for j in range(0, len(x_all)):
#         # Kwadrat położenia dla pojedynczego punktu cząstki
#         sum_squared = np.power(dx_all[j], 2) + np.power(dy_all[j], 2)
#         # Dziele przez liczbe kroków
#         # sum_squared = sum_squared / (j+1)
#         # Suma poprzedniego położenia z nowym
#         r += sum_squared
#         # Dodawanie elementu do wektora położeń
#         steps_coor.append(r)
#     dr2.append(steps_coor)
#
# plt.figure(3)
# plt.plot(t, dr2[0])
# plt.show()

# R^2 coefficient must be 0.999 for how many particles
R_2 = rSquared(t, r2[0])
print(R_2)

