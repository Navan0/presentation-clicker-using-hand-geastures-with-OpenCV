import numpy as np
import cv2
import os

video_capture = cv2.VideoCapture(0)
lower_color = np.array([0, 50, 120], dtype=np.uint8)
upper_color = np.array([180, 150, 250], dtype=np.uint8)

hand_cascade = cv2.CascadeClassifier('hand.xml')


while True:
    max_area = 0
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    ret, thresh1 = cv2.threshold(blur,70,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
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
    mask = np.zeros_like(frame) # Create mask where white is what we want, black otherwise
    n =cv2.drawContours(mask,[cnt],0,(0,255,0),2) # Draw filled contour in mask
    m =cv2.drawContours(frame,[cnt],0,(0,255,0),2)
    out = np.zeros_like(frame) # Extract out the object and place into output image
    out[mask == 255] = frame[mask == 255]

    a, d, alpha = cv2.split(frame)
    gray_layer = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    dst = cv2.merge((gray_layer, gray_layer, gray_layer, alpha))


    # cv2.drawContours(drawing,[hull],0,(0,0,255),2)
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 300,300)
    cv2.imshow('image', dst)
    cv2.namedWindow('two',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('two', 300,300)
    cv2.imshow('two', m)
    cv2.namedWindow('three',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('three', 300,300)
    cv2.imshow('three', out)
    # hands = hand_cascade.detectMultiScale(gray, 1.5, 2)
    # contour = hands

    # contour = np.array(contour)
    # if contour.ndim >= 2:
    #     print(contour[0][0])
    #     if contour[0][0] > 500:
    #         os.system('xdotool key Down')
    #     elif contour[0][0] < 500:
    #         os.system('xdotool key Up')
    # print(contour.shape)
    # print(contour.ndim)
    #cv2.putText(frame, 'Welcome', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                # (255, 0, 0), 3, cv2.LINE_AA)
    # key = cv2.waitKey(1)
    # if key != 27:
    #     cv2.destroyAllWindows()
    #     video_capture.release()
    #     break
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
