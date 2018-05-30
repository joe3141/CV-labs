import cv2
import numpy as np
from matplotlib import pyplot as plt

from timeit import default_timer as timer

def show_img(img, title='auto'):
	cv2.imshow(title, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def normalize(arr):
	mins = np.min(arr)
	maxs = np.max(arr)

	return ((arr - mins)/maxs)*128

def show_histograms(img, title = "auto"):
	hist = normalize(cv2.calcHist([img],[0],None,[256],[0,256]))
	cum_hist = normalize(np.cumsum(hist))

	img = np.zeros((128, 256), np.uint8) 
	for j in range(255):
		for i in range(int(cum_hist[j])):
			img.itemset(i, j, 100)

	for j in range(255):
		for i in range(int(hist[j])):
			img.itemset(i, j, 200)

	img = cv2.flip(img, 0)
	img = cv2.resize(img, (0,0), fx=4, fy=4) 
	show_img(img, title)

def mean_filter(img):
	kernel = np.ones((5,5),np.float32)/25
	return cv2.filter2D(img,-1,kernel)

def gaussian_filter(img):
	return cv2.GaussianBlur(img,(5,5),0)

def median_filter(img):
	out = np.zeros_like(img, np.uint8)
	
	for i in range(out.shape[0]):
		for j in range(out.shape[1]):
			neighbors = []

			for ki in range(5):
				for kj in range(5):
					if (i-2+ki) >= 0 and (i-2+ki) < (out.shape[0]) and (j-2+kj) < (out.shape[1]) and (j-2+kj) >= 0:
						neighbors.append(img.item(i-2+ki, j-2+kj))

			neighbors = np.array(neighbors)
			out.itemset(i, j, np.median(neighbors))
	return out

def median_filter_enhanced(img):
	out = np.zeros_like(img, np.uint8)
	
	for i in range(out.shape[0]):
		for j in range(out.shape[1]):
			neighbors = []
			pixel = img.item(i, j)

			if (pixel >= 0 and pixel < 33) or (pixel >= 223): # Enhancement
				for ki in range(5):
					for kj in range(5):
						if (i-2+ki) >= 0 and (i-2+ki) < (out.shape[0]) and (j-2+kj) < (out.shape[1]) and (j-2+kj) >= 0:
							neighbors.append(img.item(i-2+ki, j-2+kj))

				neighbors = np.array(neighbors)
				out.itemset(i, j, np.median(neighbors))
			else:
				out.itemset(i, j, pixel)

	return out

def hist_eq(img):
	hist = cv2.calcHist([img],[0],None,[256],[0,256])
	cum_hist = np.cumsum(hist)

	out = np.zeros_like(img)
	unq = np.unique(img)
	num_pix = img.shape[0] * img.shape[1]

	for val in unq:
		out[[img == val]] = (cum_hist[val] * 255) / num_pix

	return out

def contrast_stretching(img):
	a = 0
	b = 255
	c = np.min(img)
	d = np.max(img)
	scale_fact = ((b-a) / (d-c))
	img = img - (c*scale_fact)
	return np.uint8(img + a)

cameraman = cv2.imread('cameraman.png', 0)
fognoise  = cv2.imread('fognoise.png', 0)
frostfog  = cv2.imread('frostfog.png', 0)
tree      = cv2.imread('treeM.png', 0)
t         = cv2.imread('tree.png', 0)

# cameraman_mean = mean_filter(cameraman)
# cameraman_gaussian = gaussian_filter(cameraman)

# first_filter_time = 0
# second_filter_time = 0

# start = timer()
# fognoise_median = median_filter(fognoise)
# end = timer()
# first_filter_time = end - start

# show_histograms(cameraman, "Image's Histogram")

# show_img(cameraman_mean, "Mean Filter Application")
# show_histograms(cameraman_mean, "Histogram After Mean Filter")
# show_img(cameraman_gaussian, "Gaussian Filter Application")
# show_histograms(cameraman_gaussian, "Histogram After Guassian Filter")


# start = timer()
# fog1 = median_filter_enhanced(fognoise)
# end = timer()
# second_filter_time = end - start

# show_img(fognoise_median, "Fognoise After Applying Median Filter")
# show_img(fog1, "Fognoise After Applying second version of Median Filter")

# print("Time taken for standard median filter: ", first_filter_time)
# print("Time taken for enhanced median filter: ", second_filter_time)

# frost_hist = hist_eq(frostfog)
# frost_contrast = contrast_stretching(frostfog)

# show_img(frost_hist, "Applying Histogram Equalization on FrostFog")
# show_img(frost_contrast, "Applying Contrast Stretching on FrostFog")

# show_histograms(frost_hist, "Histogram After Applying Histogram Equalization")
# show_histograms(frost_contrast, "Histogram After Applying Contrast Stretching")

show_img((tree - t)*50, "Mystery. Please Check the console for runtime comparisons") # The mystery
