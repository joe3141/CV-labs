import numpy as np
import cv2
from matplotlib import pyplot as plt


# https://docs.opencv.org/trunk/d3/dc1/tutorial_basic_linear_transform.html
# https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/

def show_img(img):
	cv2.imshow('auto',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


#Loading and displaying the image

guc = cv2.imread('GUC.png', 0)
img = cv2.imread('roi.jpg', -1)
calc = cv2.imread('calculator.png', 0)
bond = cv2.imread('james.png', 0)
london = cv2.imread('london1.png', 0)

# cv2.imshow('GUC',guc)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# plt.imshow(guc, cmap='gray', interpolation='bicubic')
# plt.xticks([]), plt.yticks([])
# plt.show()


# guc[100, 100] # Pixel <100, 100>, usually 3vec however it's a scalar since we are dealing with a greyscale img.
# guc[100, 100, 0] # Acessing a particular channel, meaningless in this case and would raise an error.

# guc.item(100, 100) # Faster way to acess. Make sure to use a third index when appropriate.
# guc.itemset((10, 10), 100) # Set a pixel value. pos, val

# print(guc.shape)
# print(guc.size)
# print(guc.dtype)

# print(img.item(10, 10, 0))
# print(img.dtype)

# b, g, r = cv2.split(img)
# img = cv2.merge((b, g, r))
# ball = img[280:340, 330:390]
# b = img[:,:,0]
# img[:,:,2] = 0
# show_img(ball)


# img2gray = cv2.cvtColor(calc,cv2.COLOR_BGR2GRAY)
# ret, mask = cv2.threshold(img2gray, 200, 255, cv2.THRESH_BINARY)
# mask_inv = cv2.bitwise_not(mask)
# tst = cv2.bitwise_and(calc, calc, mask=mask_inv)
# show_img(img)
# show_img(tst)

# e1 = cv2.getTickCount()
# # your code execution
# e2 = cv2.getTickCount()
# time = (e2 - e1)/ cv2.getTickFrequency()


# Change background
mask = cv2.inRange(bond, 245, 255)  # could also use threshold

mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))  # "erase" the small white points in the resulting mask
mask = cv2.bitwise_not(mask)  # invert mask

bk = london  # white bk
fg_masked = cv2.bitwise_and(bond, bond, mask=mask)

mask = cv2.bitwise_not(mask)
bk_masked = cv2.bitwise_and(bk, bk, mask=mask)
final = cv2.bitwise_or(fg_masked, bk_masked)
show_img(final)
cv2.imwrite("./test.bmp", final)