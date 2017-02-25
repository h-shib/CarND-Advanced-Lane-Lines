**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/compare_caliblation.jpg
[image2]: ./test_images/straight_lines1.jpg
[image3]: ./output_images/undistort_straight_lines1.jpg
[image4]: ./output_images/thresh_straight_lines1.jpg
[image5]: ./output_images/warped_straight_lines1.jpg
[image6]: ./output_images/fit_polynomial_test5.png
[image7]: ./output_images/processed_test5.jpg

### [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
---

### Camera Calibration

#### 1. Have the camera matrix and distortion coefficients been computed correctly and checked on one of the calibration images as a test?

The code for camera calibration step is in the file named "calibrate.py".
First of all, I prepared "object points", which will be the (x, y, z) coordinates of the chessboard corners in the 3D world. Then, I applied `cv2.findChessboardCorners()` to original image, and store the corners on the 2D image. "imgpoints" will be the (x, y) coordinates which represent 2D points in image plane.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.
I applied this distortion correction to the test image using the `cv2.undistort()` function in the "lane_finding.py" file and obtained this result: 

![compare original and undistorted image][image1]

### Pipeline (single images)

#### 1. Has the distortion correction been correctly applied to each image?
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
original straight_lines1.jpg
![original straight_lines1.jpg][image2]

undistorted straight_lines1.jpg
![undistorted straight_lines1.jpg][image3]


#### 2. Has a binary image been created using color transforms, gradients or other methods?

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 164-202 in "lane_finding.py"). Here's an example of my output for this step. It clearly shows the lane lines.

![thresholded straight_lines1.jpg][image4]


#### 3. Has a perspective transform been applied to rectify the image?

The code for my perspective transform includes a function called `warp_image()`, which appears in lines 204-220 in the file "lane_finding.py". The `warp_image()` function takes as inputs an undistorted image (`undist_img`) and apply `cv2.warpPerspective()` function. I chose the hardcode the source and destination points in the following manner:

```
offset_x = 100 # offset for x direction of undistorted image
offset_y = 130 # offset for y direction of undistorted image
offset = 200 # offset for transformed image
center_x = self.img_size[0]/2.
center_y = self.img_size[1]/2.

src = np.float32([[center_x-offset_x, center_y+offset_y],
				  [center_x+offset_x, center_y+offset_y],
				  [self.img_size[0]-200, self.img_size[1]],
				  [200, self.img_size[1]]])

dst = np.float32([[offset, offset],
				  [self.img_size[0]-offset, offset],
				  [self.img_size[0]-offset, self.img_size[1]],
				  [offset, self.img_size[1]]])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![warped straight_laines1.jpg][image5]


#### 4. Have lane line pixels been identified in the rectified image and fit with a polynomial?

I applied `find_lines()` function in lines 38-148 to get curvatures and polynomial lines.

![alt text][image6]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 19-31 in `calc_curvature()` function. Also, I calculated offset from the center in lines 146-147 in `find_lines()` function.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 225-241 in my code in "lane_finding.py" in the function `process_image()`.  Here is an example of my result on a test image:

![result image][image7]

---

### Pipeline (video)

Here's a [link to my video result](./project_video_result.mp4)

---

### Discussion

This was the good exercise to understand how to find lane lines with computer vision approach. I learned a lot of techniques from the lesson and could apply those into this problem.
One of the hardest steps of this exercise was perspective transformation. Althogh there might be more robust mathmatical approach, I applied `cv2.getPerspectiveTransform()` with fixed src and dst parameters. I thought this was the reason why that deeplearning approach is prefered in recent studies because we don't have to tweak the parameters to get lane lines.
The case that this pipeline would fail is when the solid shade lines are in the frame such as in the `challenge_video.mp4`. When it happened, `find_lines()` function would shade as the lane line and fail to detect the actual lane.
To overcome it, I could apply region of interest mask to the image and store some previous lane line parameters to estimate next lane lines.
Anyway, I could implement basic lane finding pipline in this project.
