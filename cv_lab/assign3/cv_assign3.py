import cv2
import numpy as np
from matplotlib import pyplot as plt

import itertools

def show_img(img, title='auto'):
	cv2.imshow(title, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def SAD(a, b):
	return np.sum(np.absolute(a-b).flatten())

def get_surrounding_window(img, pxl_i, pxl_j, d):
	"""
	Assuming the window is of odd size.
	"""
	img = np.array(img) # To appease Python's type gods.
	ref_point_offset = int(d/2)
	ref_point_i = pxl_i - ref_point_offset
	ref_point_j = pxl_j - ref_point_offset

	# if ref_point_i < 0:
	# 	ref_point_i = 0
	# if ref_point_j < 0:
	# 	ref_point_j = 0

	return img[ref_point_i:(ref_point_i+d), ref_point_j:(ref_point_j+d)]

def check_bounds(vector, x_bound, y_bound):
	return vector[0] >= 0 and vector[0] < x_bound and vector[1] >= 0 and vector[1] < y_bound

def pad(img, d):
	pad_size = int(d/2)
	return cv2.copyMakeBorder(img, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=0)

def disparity_estimation(img1, img2, xmin, xmax, ymin, ymax, d):
	"""
	Returns disparity map.
	Assuming both images are the same size.
	"""
	img1_pad = pad(img1, d)
	img2_pad = pad(img2, d)
	pad_offset = int(d/2)
	# print(pad_offset)
	out = np.zeros_like(img1)
	for j in range(pad_offset, img1.shape[0]+pad_offset):
		for i in range(pad_offset, img1.shape[1]+pad_offset):
			disparity_vectors = [v for v in itertools.product(range(xmin+i, xmax+i+1), range(ymin+j, ymax+j+1)) \
			if check_bounds(v, img1.shape[1]+pad_offset, img1.shape[0]+pad_offset)]
			vals = []

			for v in disparity_vectors:
				vals.append(SAD(get_surrounding_window(img1_pad, j, i, d), \
				get_surrounding_window(img2_pad, v[1], v[0], d)))

			out.itemset(j - pad_offset, i - pad_offset, min(vals))

	return out


left = cv2.imread("Tsukuba_L.png", 1)
right = cv2.imread("Tsukuba_R.png", 1)

l_gs = cv2.imread("Tsukuba_L.png", 0)
r_gs = cv2.imread("Tsukuba_R.png", 0)

anaglyph = np.zeros_like(left)

anaglyph[:,:,:2] = left[:,:,:2]
anaglyph[:,:,2] = right[:,:,2]

# show_img(anaglyph, "3D Anaglyph") 


# Depth map using an opencv function
# stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
# disparity = stereo.compute(l_gs, r_gs)

# plt.imshow(disparity,'gray')
# plt.show()


disparity_map = disparity_estimation(l_gs, r_gs, 0, 16, 0, 16, 15)
# tst1 = [[9, 8, 9, 2, 1], [8, 7, 9, 3, 1], [9, 9, 2, 2, 1], [5, 8, 1, 1, 1], [9, 9, 9, 1, 2]]
# tst2 = [[2, 3, 1, 3, 1], [2, 8, 7, 8, 1], [1, 9, 8, 9, 1], [1, 9, 8, 2, 2], [1, 9, 9, 1, 1]]

# tst1 = np.array([np.array(a) for a in tst1])
# tst2 = np.array([np.array(a) for a in tst2])

# disparity_map = disparity_estimation(tst1, tst2, 0, 0, 0, 2, 3)
# disparity_map = disparity_estimation(l_gs, r_gs, 0, 0, 0, 2, 3)
# print(pad(l_gs, 3).shape)

plt.imshow(disparity_map,'gray')
plt.show()

# print(disparity_map)
