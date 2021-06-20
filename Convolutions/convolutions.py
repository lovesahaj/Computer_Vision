import cv2
import numpy as np
import os
from skimage.exposure import rescale_intensity
import seaborn as sns
import matplotlib.pyplot as plt

vid = cv2.VideoCapture(0)


"""
gaussianArray : funcition
args:   muu -> mean of the normal distribution
        sigma -> standard deviation of the normal distribution
        size -> size of the return gaussian array

returns: Gaussain Distributed Array    
"""


def gaussianArray(muu: float, sigma: float, size: tuple):
    # np.linspace -> evenly spaced values given a starting and ending point
    # np.meshgird -> cooridinate matrices from vectors
    x, y = np.meshgrid(
        np.linspace(-2, 2, size[0]), np.linspace(-2, 2, size[1]))

    dst = np.sqrt(x*x+y*y)

    # this is the formula for a gaussian distribution
    return np.exp(-0.5 * ((dst-muu)**2 / (2.0 * sigma**2)))


def convolve_and_show(image, kernel):
    output = cv2.filter2D(image.astype('float64') / 255, -1, kernel)

    if output.min() < 0:
        output = np.abs(output)

    output = (output - output.min()) / (output.max() - output.min())

    # print(output.max())
    # print(output.min())

    output = (output * 255).astype('uint8')
    output = rescale_intensity(output, in_range=(0, 255))

    return output


def show_kernel(kernel):
    sns.heatmap(kernel, cmap='coolwarm')
    plt.show()


if __name__ == '__main__':
    laplacian = np.asarray([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0],
    ])

    sharpen = np.array((
        [0, -1, 0],
        [-1, 2, -1],
        [0, -1, 0]), dtype="int")

    haarX = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
    ])

    sobelX = np.array((
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ))

    Identity = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ])

    edge1 = np.array([
        [1, 0, -1],
        [0, 0, 0],
        [-1, 0, 1],
    ])

    edge2 = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0],
    ])

    edge3 = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1],
    ])

    unsharp = np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, -476, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1],
    ])

    gaussian = gaussianArray(0, 1, (4, 4))

    blur = np.ones((10, 10), dtype="float") * (1.0 / (10 * 10))

    kernel = laplacian

    show_kernel(kernel)

    while True:
        ret, image = vid.read()

        ########### COOOL STUFF ############
        image_blurred = convolve_and_show(image, gaussian)
        image_convolved = convolve_and_show(image_blurred, kernel)

        # image = image - image_convolved

        # cv2.imshow('Image', image_blurred)
        cv2.imshow('Convolved Image', image_convolved)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        print(image.shape)
        os.system("cls")

vid.release()
cv2.destroyAllWindows()
