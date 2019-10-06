import numpy as np
import cv2
img = cv2.imread('/home/navaneeth/work/imgMax/mine.jpg') # Read in your image
# contours, _ = cv2.findContours(...) # Your call to find the contours using OpenCV 2.4.x
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
ret, thresh1 = cv2.threshold(blur,70,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Your call to find the contours
max_area = 0
ci = 0
for i in range(len(contours)):
    cnt = contours[i]
    area = cv2.contourArea(cnt)
    if(area > max_area):
        max_area = area
        ci = i
    cnt = contours[ci]
#
#
# print(cnt)
# The index of the contour that surrounds your object
mask = np.zeros_like(img) # Create mask where white is what we want, black otherwise
cv2.drawContours(mask,[cnt],0,(0,255,0),2) # Draw filled contour in mask
out = np.zeros_like(img) # Extract out the object and place into output image
out[mask == 255] = img[mask == 255]

a, d, alpha = cv2.split(img)
gray_layer = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
dst = cv2.merge((gray_layer, gray_layer, gray_layer, a))
# Show the output image
# cv2.imshow('Output', out)
cv2.imwrite('/home/navaneeth/work/imgMax/test.png', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
