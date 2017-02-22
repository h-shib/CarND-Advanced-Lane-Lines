import cv2
import os
import numpy as np
import pickle

def camera_calibration():
	# skip camera calibration if pickled file exists.
	if os.path.exists('dist_pickle.p'):
		print('file exists.')
		return

	# prepare for chessboard corners
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


if __name__ == '__main__':
	camera_calibration()