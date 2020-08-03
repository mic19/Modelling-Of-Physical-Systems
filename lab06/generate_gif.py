import imageio

if __name__ == "__main__":

	# d change
	gif_data = ['output/d_change/gif' + str(i) + '.png' for i in range(20)]
	images = []

	for filename in gif_data:
		images.append(imageio.imread(filename))
	imageio.mimsave('output_d.gif', images, duration=0.5)

	# u change
	gif_data = ['output/u_change/gif' + str(i) + '.png' for i in range(20)]
	images = []

	for filename in gif_data:
		images.append(imageio.imread(filename))
	imageio.mimsave('output_u.gif', images, duration=0.5)

