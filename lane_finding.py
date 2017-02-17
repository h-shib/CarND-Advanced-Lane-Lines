import cv2
import numpy as np
import glob
import os
import pickle
import matplotlib.pyplot as plt

def camera_calibration():
	if os.path.exists('dist_pickle.p'):
		print('file exists.')
		return

	objp = np.zeros((6*9,3), np.float32)
	objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

	objpoints = [] # 3d points in real world space
	imgpoints = [] # 2d points in image plane.

	images = glob.glob('./camera_cal/calibration*.jpg')

	for fname in images:
		image = cv2.imread(fname)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		retval, corners = cv2.findChessboardCorners(gray, (9, 6), None)

		if retval:
			imgpoints.append(corners)
			objpoints.append(objp)

	# save imgpoints and objpoints into pickle data
	dist_data = {'imgpoints': imgpoints, 'objpoints': objpoints}
	pickle.dump(dist_data, open('dist_pickle.p', 'wb'))
	print('dist data saved.')


def undistort(img, objpoints, imgpoints):
	retval, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[0:2], None, None)
	dst = cv2.undistort(img, mtx, dist, None, mtx)
	return dst

def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
	# Calculate gradient magnitude
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    mag = np.sqrt(sobelx**2 + sobely**2)
    scaled_sobel = np.uint8(255*mag/np.max(mag))
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel>=thresh[0]) & (scaled_sobel<=thresh[1])] = 1
    return mag_binary

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
	# Calculate gradient direction
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	abs_sobelx = np.absolute(sobelx)
	abs_sobely = np.absolute(sobely)
	direction = np.arctan2(abs_sobely, abs_sobelx)
	dir_binary = np.zeros_like(direction)
	dir_binary[(direction>=thresh[0]) & (direction<=thresh[1])] = 1
	return dir_binary

def color_threshold(img, thresh=(0, 255)):
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	s_channel = hls[:, :, 2]
	s_binary = np.zeros_like(s_channel)
	s_binary[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
	return s_binary

def apply_threshold(img, sobel_kernel=3):
	mag_binary = mag_thresh(img, sobel_kernel=sobel_kernel, thresh=(30, 100))
	dir_binary = dir_threshold(img, sobel_kernel=sobel_kernel, thresh=(0.7, 1.3))
	s_binary = color_threshold(img, thresh=(170, 255))
	combined_binary = np.zeros_like(dir_binary)
	combined_binary[((mag_binary == 1) & (dir_binary == 1)) | (s_binary == 1)] = 1
	return combined_binary


def main():
	# run pipeline
	camera_calibration()

	dist_data = pickle.load(open('dist_pickle.p', 'rb'))
	objpoints = dist_data['objpoints']
	imgpoints = dist_data['imgpoints']

	image = plt.imread('test_images/test1.jpg')
	plt.imshow(image)
	plt.show()
	dst = undistort(image, objpoints, imgpoints)
	thresh_binary = apply_threshold(dst)
	plt.imshow(thresh_binary, cmap='gray')
	plt.show()


if __name__ == '__main__':
	main()