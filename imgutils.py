import matplotlib.pyplot as plt


def color_grayscale_img(grayscale_img):
    cmap = plt.get_cmap("viridis")

    return cmap(grayscale_img)
