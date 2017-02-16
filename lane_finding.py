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

def main():
	# run pipeline
	camera_calibration()

	dist_data = pickle.load(open('dist_pickle.p', 'rb'))
	objpoints = dist_data['objpoints']
	imgpoints = dist_data['imgpoints']

	images = glob.glob('./camera_cal/calibration*.jpg')

	for fname in images:
		image = cv2.imread(fname)
		dst = undistort(image, objpoints, imgpoints)
		plt.imshow(dst)
		plt.show()

if __name__ == '__main__':
	main()