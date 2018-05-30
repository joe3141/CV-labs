import numpy as np
import cv2

# How to use!!!!!
# Just run and keep pressing any buttons to scroll through the output pics.

def show_img(img, title='auto'):
	cv2.imshow(title, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# Credit from : https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
def adjust_gamma(img, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(img, table)

def print_hist(img):
	frq = np.bincount(img.ravel())
	unq = np.unique(img)
	for i in unq:
		print(i, frq[i])

	return unq

###### Dim GUC: ######

guc = cv2.imread('GUC.png', 0) 
gamma = 0.5
adjusted = adjust_gamma(guc, gamma=gamma)
show_img(adjusted, "Dimmed")

###### Shadow removal: ######

calc = cv2.imread('calculator.png', 0)
# print(calc.item(62, 120))
mask = cv2.inRange(calc, 183, 255)  # Threshold

# Noise removal
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))  # "erase" the small white points in the resulting mask
mask_inv = cv2.bitwise_not(mask)  # invert mask

white = np.full(calc.shape, 255, dtype=np.uint8)  # white bk
fg_masked = cv2.bitwise_and(calc, calc, mask=mask_inv)

bk_masked = cv2.bitwise_and(white, white, mask=mask)
removed = cv2.bitwise_or(calc, bk_masked)
show_img(removed)

###### Brighten coat: ######
coat = cv2.imread('cameraman.png', 0)

bcoat = cv2.add(coat, 100)

for i in range(256):
	for j in range(256):
		x = bcoat.item(i, j)
		if not (x <= 120 and x >= 108 and i >= 68 and i <= 211 and j <= 125 ):
			bcoat.itemset(i, j, coat.item(i, j))

show_img(bcoat)


###### Segmentation: ######

lake = cv2.imread('lake.png', 0)
dlake = cv2.add(lake, -100) # this number needs tuning
ret, thresh = cv2.threshold(dlake,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


# Noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

opening = 255 - opening

for i in range(opening.shape[0]):
	for j in range(opening.shape[1]):
		if i <= 320:
			opening.itemset(i, j, lake.item(i, j))
		elif i > 320:
			x = opening.item(i, j)
			if x != 0:
				opening.itemset(i, j, lake.item(i, j))

show_img(opening)

###### Combination: ######

bond = cv2.imread('james.png', 0)
london1 = cv2.imread('london1.png', 0)
london2 = cv2.imread('london2.png', 0)

mask = cv2.inRange(bond, 245, 255)  # could also use threshold

mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))  # "erase" the small white points in the resulting mask
mask = cv2.bitwise_not(mask)  # invert mask

bk_1 = london1  # white bk
fg_masked_1 = cv2.bitwise_and(bond, bond, mask=mask)

mask = cv2.bitwise_not(mask)
bk_masked_1 = cv2.bitwise_and(bk_1, bk_1, mask=mask)
final1 = cv2.bitwise_or(fg_masked_1, bk_masked_1)
show_img(final1)




flipped_bond = cv2.flip(bond, 1)

mask = cv2.inRange(flipped_bond, 245, 255)  # could also use threshold
mask = cv2.bitwise_not(mask)  # invert mask

bk_2 = london2
fg_masked_2 = cv2.bitwise_and(flipped_bond, flipped_bond, mask=mask)

mask = cv2.bitwise_not(mask)
bk_masked_2 = cv2.bitwise_and(bk_2, bk_2, mask=mask)
final2 = cv2.bitwise_or(fg_masked_2, bk_masked_2)
show_img(final2)