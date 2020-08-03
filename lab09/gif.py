import imageio


images = []
num = int(2790 / 100)
gif_data = ['output/out' + str(i * 100) + '.png' for i in range(0, num)]

for filename in gif_data:
	images.append(imageio.imread(filename))
	imageio.mimsave('water_level.gif', images)
