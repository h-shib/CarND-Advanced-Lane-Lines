import cv2
import numpy as np
import glob
import os
import pickle
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip


class LaneLineTracker():
	def __init__(self):
		self.found = False
		self.ploty = None
		self.left_fitx = None
		self.right_fitx = None

	def calc_curvature(self, ploty, leftx, rightx):
		# Define conversions in x and y from pixels space to meters
		ym_per_pix = 30/720 # meters per pixel in y dimension
		xm_per_pix = 3.7/700 # meters per pixel in x dimension

		# Fit new polynomials to x,y in world space
		left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
		right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
		# Calculate the new radii of curvature
		left_curverad = ((1 + (2*left_fit_cr[0]*np.max(ploty)*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
		right_curverad = ((1 + (2*right_fit_cr[0]*np.max(ploty)*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
		# Now our radius of curvature is in meters
		return (left_curverad, 'm', right_curverad, 'm')

	def find_lines(self, binary_warped):
		if self.found:
			nonzero = binary_warped.nonzero()
			nonzeroy = np.array(nonzero[0])
			nonzerox = np.array(nonzero[1])
			margin = 100
			left_lane_inds = ((nonzerox > (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + self.left_fit[2] - margin)) &
								(nonzerox < (self.left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
			right_lane_inds = ((nonzerox > (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + self.right_fit[2] - margin)) &
								(nonzerox < (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + self.right_fit[2] + margin)))

			# Again, extract left and right line pixel positions
			leftx = nonzerox[left_lane_inds]
			lefty = nonzeroy[left_lane_inds] 
			rightx = nonzerox[right_lane_inds]
			righty = nonzeroy[right_lane_inds]
			# Fit a second order polynomial to each
			left_fit = np.polyfit(lefty, leftx, 2)
			right_fit = np.polyfit(righty, rightx, 2)
			# Generate x and y values for plotting
			ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
			left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
			right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

		else:
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
			print(left_lane_inds)

			# Extract left and right line pixel positions
			leftx = nonzerox[left_lane_inds]
			lefty = nonzeroy[left_lane_inds]
			rightx = nonzerox[right_lane_inds]
			righty = nonzeroy[right_lane_inds]

			# Fit a second order polynomial to each
			print(lefty)
			left_fit = np.polyfit(lefty, leftx, 2)
			right_fit = np.polyfit(righty, rightx, 2)

			# Generate x and y values for plotting
			ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
			left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
			right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

		self.ploty = ploty
		self.left_fitx = left_fitx
		self.right_fitx = right_fitx

		curvatures = self.calc_curvature(self.ploty, self.left_fitx, self.right_fitx)
		print(curvatures)
		offset_from_center = (binary_warped.shape[1]/2 - (left_fitx[-1]+right_fitx[-1])/2) * 3.7/700
		print(self.left_fitx[-1], self.right_fitx[-1], offset_from_center, 'm')
		return binary_warped


class Lane():
	def __init__(self, img_size, objpoints, imgpoints):
		self.img_size = img_size
		self.objpoints = objpoints
		self.imgpoints = imgpoints
		self.M = None
		self.Minv = None
		self.lines = LaneLineTracker()

	def undistort(self, image):
		retval, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, image.shape[0:2], None, None)
		return cv2.undistort(image, mtx, dist, None, mtx)

	def mag_thresh(self, image, sobel_kernel=3, thresh=(0, 255)):
		# detect lane lines by gradient magnitude
		gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
		sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
		mag = np.sqrt(sobelx**2 + sobely**2)
		scaled_sobel = np.uint8(255*mag/np.max(mag))
		mag_binary = np.zeros_like(scaled_sobel)
		mag_binary[(scaled_sobel>=thresh[0]) & (scaled_sobel<=thresh[1])] = 1
		return mag_binary

	def dir_threshold(self, image, sobel_kernel=3, thresh=(0, np.pi/2)):
		# detect lane lines by gradient direction
		gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
		sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
		abs_sobelx = np.absolute(sobelx)
		abs_sobely = np.absolute(sobely)
		direction = np.arctan2(abs_sobely, abs_sobelx)
		dir_binary = np.zeros_like(direction)
		dir_binary[(direction>=thresh[0]) & (direction<=thresh[1])] = 1
		return dir_binary

	def color_threshold(self, image, thresh=(0, 255)):
		# detect lane lines by saturation
		hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
		s_channel = hls[:, :, 2]
		s_binary = np.zeros_like(s_channel)
		s_binary[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
		return s_binary

	def apply_threshold(self, image, sobel_kernel=3):
		# apply multiple threshold functions
		mag_binary = self.mag_thresh(image, sobel_kernel=sobel_kernel, thresh=(50, 200))
		dir_binary = self.dir_threshold(image, sobel_kernel=sobel_kernel, thresh=(0.7, 1.3))
		s_binary = self.color_threshold(image, thresh=(170, 255))
		combined_binary = np.zeros_like(dir_binary)
		combined_binary[((mag_binary == 1) & (dir_binary == 1)) | (s_binary == 1)] = 1
		return combined_binary

	def warp_image(self, undist_img):
		if self.M:
			return cv2.warpPerspective(undist_img, self.M, self.img_size)
		offset_x = 100 # offset for x direction of undistorted image
		offset_y = 130 # offset for y direction of undistorted image
		offset = 200 # offset for transformed image
		center_x = self.img_size[0]/2.
		center_y = self.img_size[1]/2.
		
		src = np.float32([[center_x-offset_x, center_y+offset_y], [center_x+offset_x, center_y+offset_y],
							[self.img_size[0]-200, self.img_size[1]], [200, self.img_size[1]]])
		dst = np.float32([[offset, offset], [self.img_size[0]-offset, offset],
							[self.img_size[0]-offset, self.img_size[1]], [offset, self.img_size[1]]])

		self.M = cv2.getPerspectiveTransform(src, dst)
		self.Minv = cv2.getPerspectiveTransform(dst, src)
		return cv2.warpPerspective(undist_img, self.M, self.img_size)

	def unwarp_image(self, undist_img):
		return cv2.warpPerspective(undist_img, self.Minv, self.img_size)

	def draw_lane_line(self, warped, undist):
		warp_zero = np.zeros_like(warped).astype(np.uint8)
		color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

		# Recast the x and y points into usable format for cv2.fillPoly()
		pts_left = np.array([np.transpose(np.vstack([self.lines.left_fitx, self.lines.ploty]))])
		pts_right = np.array([np.flipud(np.transpose(np.vstack([self.lines.right_fitx, self.lines.ploty])))])
		pts = np.hstack((pts_left, pts_right))

		# Draw the lane onto the warped blank image
		cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

		# Warp the blank back to original image space using inverse perspective matrix (Minv)
		newwarp = cv2.warpPerspective(color_warp, self.Minv, self.img_size) 
		# Combine the result with the original image
		result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
		#plt.imshow(result)
		#plt.show()
		return result

	def process_image(self, image):
		"""
		image processing pipeline
		read image frames from video and draw lane line
		"""
		undist_img = self.undistort(image)
		thresh_img = self.apply_threshold(undist_img) # find lane line by threshold function
		warped_img = self.warp_image(thresh_img) # get warped binary image
		lines = self.lines.find_lines(warped_img) # find lane lines and draw area
		lane = self.draw_lane_line(lines, undist_img)
		return lane


def main():
	# read calibration data
	dist_data = pickle.load(open('dist_pickle.p', 'rb'))
	objpoints = dist_data['objpoints']
	imgpoints = dist_data['imgpoints']

	# debug
	image = plt.imread('test_images/test5.jpg')
	img_size = (image.shape[1], image.shape[0])
	lane = Lane(img_size, objpoints, imgpoints)
	result = lane.process_image(image)
	plt.imshow(result)
	plt.show()

	"""
	# apply process_image function to the test images
	input_dir = "test_images"
	output_dir = "output_images"

	for fname in os.listdir(input_dir):
		image = plt.imread(os.path.join(input_dir, fname))
		lane = Lane(objpoints, imgpoints)
		result_image = lane.process_image(image)
		plt.imsave(os.path.join(output_dir, fname), result_image)
	"""
	# apply process_image function to the video
	"""
	output = 'test2.mp4'
	clip_input = VideoFileClip("project_video.mp4")
	clip = clip_input.fl_image(process_image)
	clip.write_videofile(output, audio=False)
	"""

if __name__ == '__main__':
	main()