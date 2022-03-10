import cv2 as cv
import numpy as np
import pdb
from matplotlib import pyplot as plt


def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes	
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")


img_rgb = cv.imread("./data/thermal_ext_calib_frames/frame0014.jpg")
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
r = cv.selectROI(img_gray)
img_gray = img_gray[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
r = cv.selectROI(img_gray)
template = img_gray[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

# template = cv.imread('template.jpg',0)
# template = cv.medianBlur(template,5)
w, h = template.shape[::-1]
res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)
# pdb.set_trace()
boxes = [[pt[0], pt[1], pt[0] + w, pt[1] + h] for pt in zip(*loc[::-1])]
boxes = np.array(boxes)
nms_box = non_max_suppression_fast(boxes, 0.4)
detector = cv.SimpleBlobDetector_create()
for box in nms_box:
    # cv.rectangle(img_rgb, (box[0],box[1]), (box[2], box[3]), (0,0,255), 1)
    cropped = img_gray[box[1]:box[3], box[0]:box[2]]

    # th3 = cv.adaptiveThreshold(cropped,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            # cv.THRESH_BINARY,11,2)
    # blurred = cv.GaussianBlur(cropped, (3, 3), 0)
    # thresh = cv.adaptiveThreshold(cropped,255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #          cv.THRESH_BINARY,11,2)
    # find contours in the thresholded image and initialize the
    # shape detector
    # contours, hierarchy = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[-2:]

    # pdb.set_trace()
    # keypoints = detector.detect(cropped)
  
# Draw blobs on our image as red circles
    gray_blurred = cv.blur(cropped, (3, 3))
  
# Apply Hough transform on the blurred image.
    detected_circles = cv.HoughCircles(gray_blurred, 
                    cv.HOUGH_GRADIENT, 1, 20, param1 = 50,
                param2 = 30, minRadius = 1, maxRadius = 40)
    
    # Draw circles that are detected.
    if detected_circles is not None:
    
        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
    
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
    
            # Draw the circumference of the circle.
            cv.circle(cropped, (a, b), r, (0, 255, 0), 2)
    
    cv.imshow("res", cropped)
# cv.imwrite('res.png',img_rgb)
    cv.waitKey(0)