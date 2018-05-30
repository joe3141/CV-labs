import cv2
import numpy as np
from matplotlib import pyplot as plt

def show_img(img, title='auto'):
	cv2.imshow(title, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

left = cv2.imread("Tsukuba_L.png", 1)
right = cv2.imread("Tsukuba_R.png", 1)

l_gs = cv2.imread("Tsukuba_L.png", 0)
r_gs = cv2.imread("Tsukuba_R.png", 0)

anaglyph = np.zeros_like(left)

anaglyph[:,:,:2] = left[:,:,:2]
anaglyph[:,:,2] = right[:,:,2]

# show_img(anaglyph, "3D Anaglyph") 

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(l_gs, r_gs)

plt.imshow(disparity,'gray')
plt.show()
