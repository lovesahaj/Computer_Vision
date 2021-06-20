from PIL import Image
from tqdm import tqdm
from solver import print_board, solve
import cv2
import numpy as np
from utils import preProcess, reorder, show_image, biggestContour
from cnn import give_arr
import pytesseract

size = (800, 800)
newsize = (900, 900)


# PROCESSING THE IMAGE
image = cv2.imread("1.png")  # Reading image
# image = image[30:-400, 100:-100]
image = cv2.resize(image, size)  # Resizing the image

whiteboard = np.zeros_like(image)

# appling the preprocesing and getting the threshold
imgThreshold = preProcess(image)
show_image(imgThreshold)

# Finding the countours
imgContours = image.copy()
imgBigContours = image.copy()

contour, hierarchy = cv2.findContours(
    imgThreshold, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours, contour, -1, (0, 255, 0), 3)

show_image(imgContours)

biggest, maxArea = biggestContour(contour)
print(biggest)

if biggest.size != 0:
    biggest = reorder(biggest)
    cv2.drawContours(imgBigContours, biggest, -1, (0, 0, 255), 25)
    show_image(imgBigContours)

    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [size[0], 0], [0, size[1]], [size[0], size[1]]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpColoured = cv2.warpPerspective(image, matrix, size)

    imgDetectedDigits = whiteboard.copy()
    imgWarpColoured = cv2.cvtColor(imgWarpColoured, cv2.COLOR_BGR2GRAY)

    imgWarpColoured = cv2.resize(imgWarpColoured, newsize)
    show_image(imgWarpColoured)

board = give_arr(imgWarpColoured)
print_board(board)

# test_img = Image.fromarray(imgWarpColoured[:100, :100])
# test_img.show()

# print(pytesseract.image_to_string(test_img))

# print_board(solve(board, tqdm(total=81)))
