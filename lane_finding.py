import cv2
import numpy as np
import glob
import os
import pickle
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

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

def perspective_transform(undist_img):
	offset_x = 100
	offset_y = 130
	offset = 200
	img_size = (undist_img.shape[1], undist_img.shape[0])
	center_x = img_size[0]/2.
	center_y = img_size[1]/2.
	src = np.float32([[center_x-offset_x, center_y+offset_y], [center_x+offset_x, center_y+offset_y],
						[img_size[0]-200, img_size[1]], [200, img_size[1]]])
	dst = np.float32([[offset, offset], [img_size[0]-offset, offset],
						[img_size[0]-offset, img_size[1]],
						[offset, img_size[1]]])
	M = cv2.getPerspectiveTransform(src, dst)
	Minv = cv2.getPerspectiveTransform(dst, src)
	warped = cv2.warpPerspective(undist_img, M, img_size)
	return warped, Minv

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
	mag_binary = mag_thresh(img, sobel_kernel=sobel_kernel, thresh=(50, 200))
	dir_binary = dir_threshold(img, sobel_kernel=sobel_kernel, thresh=(0.7, 1.3))
	s_binary = color_threshold(img, thresh=(170, 255))
	combined_binary = np.zeros_like(dir_binary)
	combined_binary[((mag_binary == 1) & (dir_binary == 1)) | (s_binary == 1)] = 1
	return combined_binary


def draw_lane_line(warped, left_fitx, right_fitx, ploty, Minv, image, undist):
	warp_zero = np.zeros_like(warped).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
	# Combine the result with the original image
	result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
	#plt.imshow(result)
	#plt.show()
	return result

def find_lines(binary_warped, Minv, image, undist):
	# Assuming you have created a warped binary image called "binary_warped"
	# Take a histogram of the bottom half of the image
	histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
	# Create an output image to draw on and  visualize the result
	out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	# Choose the number of sliding windows
	nwindows = 9
	# Set height of windows
	window_height = np.int(binary_warped.shape[0]/nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	# Set the width of the windows +/- margin
	margin = 100
	# Set minimum number of pixels found to recenter window
	minpix = 50
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(nwindows):
	    # Identify window boundaries in x and y (and right and left)
	    win_y_low = binary_warped.shape[0] - (window+1)*window_height
	    win_y_high = binary_warped.shape[0] - window*window_height
	    win_xleft_low = leftx_current - margin
	    win_xleft_high = leftx_current + margin
	    win_xright_low = rightx_current - margin
	    win_xright_high = rightx_current + margin
	    # Draw the windows on the visualization image
	    cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
	    cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
	    # Identify the nonzero pixels in x and y within the window
	    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
	    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
	    # Append these indices to the lists
	    left_lane_inds.append(good_left_inds)
	    right_lane_inds.append(good_right_inds)
	    # If you found > minpix pixels, recenter next window on their mean position
	    if len(good_left_inds) > minpix:
	        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
	    if len(good_right_inds) > minpix:        
	        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds] 

	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)

	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
	plt.imshow(out_img)
	plt.plot(left_fitx, ploty, color='yellow')
	plt.plot(right_fitx, ploty, color='yellow')
	plt.xlim(0, 1280)
	plt.ylim(720, 0)

	result = draw_lane_line(binary_warped, left_fitx, right_fitx, ploty, Minv, image, undist)
	return result


def process_image(image):
	dist_data = pickle.load(open('dist_pickle.p', 'rb'))
	objpoints = dist_data['objpoints']
	imgpoints = dist_data['imgpoints']

	#image = plt.imread('test_images/straight_lines2.jpg')
	# image = plt.imread('test_images/straight_lines2.jpg')
	dst = undistort(image, objpoints, imgpoints)
	thresh_binary = apply_threshold(dst)
	warped_binary, Minv = perspective_transform(thresh_binary)
	result = find_lines(warped_binary, Minv, image, dst)
	return result

def main():
	# run pipeline
	camera_calibration()

	"""
	dist_data = pickle.load(open('dist_pickle.p', 'rb'))
	objpoints = dist_data['objpoints']
	imgpoints = dist_data['imgpoints']

	#image = plt.imread('test_images/straight_lines2.jpg')
	image = plt.imread('test_images/straight_lines2.jpg')
	plt.imshow(image)
	plt.show()
	dst = undistort(image, objpoints, imgpoints)
	thresh_binary = apply_threshold(dst)
	warped_binary, Minv = perspective_transform(thresh_binary)
	find_lines(warped_binary, Minv, image, dst)
	plt.imshow(warped_binary, cmap='gray')
	plt.show()
	"""
	output = 'test.mp4'
	clip1 = VideoFileClip("project_video.mp4")
	print(clip1)
	clip = clip1.fl_image(process_image)
	clip.write_videofile(output, audio=False)


if __name__ == '__main__':
	main()