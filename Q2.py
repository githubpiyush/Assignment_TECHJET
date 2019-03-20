import cv2
import imutils
import time


greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
x = input('Please enter video file name/path: ')
video_input = cv2.VideoCapture(x)
time.sleep(2.0)
while True:
	image = video_input.read()
	image = image[1]
	if image is None:
		break
	image = imutils.resize(image, width=600)
	blur = cv2.GaussianBlur(image, (11, 11), 0)
	hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

	temp = cv2.inRange(hsv, greenLower, greenUpper)
	temp = cv2.erode(temp, None, iterations=2)
	temp = cv2.dilate(temp, None, iterations=2)
	contours = cv2.findContours(temp.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	contours = imutils.grab_contours(contours)
	center = None
	if len(contours) > 0:
		max_contour = max(contours, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(max_contour)
		moment = cv2.moments(max_contour)
		p1 = int(moment["m10"] / moment["m00"])
		p2 = int(moment["m01"] / moment["m00"])
		center = (p1, p2)
		pt1 = (int(p1+radius), int(p2+radius))
		pt2 = (int(p1+radius+radius), int(p2+radius+radius))
		if radius > 10:
			cv2.circle(image, (int(x), int(y)), int(radius),(0, 0, 255), 2)

	cv2.imshow("output", image)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("s"):
		break
	
	if key == ord("p"):
		cv2.arrowedLine(image, pt2, pt1, (0,0,255), 3)
		cv2.imshow("output", image)
		cv2.waitKey(5)
		time.sleep(5)

video_input.release()
cv2.destroyAllWindows()
