import numpy as np
import matplotlib.pyplot as plt
import math

size = 1000 # steps
n = 100 # particles

points = np.zeros((n, size, 2), dtype='float64')
distances = np.zeros((size, n))

for j in range(n):
	for i in range(size):
		move = np.random.normal(size=2)
		points[j][i] = points[j][i - 1] + move
		distances[i][j] = math.pow((math.hypot(*points[j][i - 1])), 2)

# plt.scatter(points[0][:, 0], points[0][:, 1], s=5, marker='o')

# for i in range(n):
# 	plt.scatter(points[i][:, 0], points[i][:, 1], s=1, marker='o')
# 	plt.plot(range(size), distances[:,i])

mean_distances = np.zeros(size)
for i in range(size):
	mean_distances[i] = np.mean(distances[i])
# plt.plot(range(size), mean_distances)

result = np.correlate(range(size), mean_distances, mode='full')
# plt.plot(result)

all_x_points = []
all_y_points = []
for i in range(n):
	all_x_points = np.append(all_x_points, points[i, :, 0])
	all_y_points = np.append(all_y_points, points[i, :, 1])

# plt.hist2d(all_x_points, all_y_points, (50, 50), cmap=plt.cm.jet)
# plt.colorbar()
plt.show()