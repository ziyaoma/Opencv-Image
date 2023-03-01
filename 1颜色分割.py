import cv2
from  matplotlib import pyplot as plt
from PIL import Image
import numpy as np
imgpath = r"images\1person.jpg"

#img = Image.open(imgpath)
img = cv2.imread(imgpath)
blur = cv2.blur(img,(5,5))
blur0 = cv2.medianBlur(blur,5)
blur1 = cv2.GaussianBlur(blur0,(5,5),0)
blur2 = cv2.bilateralFilter(blur1,9,75,75)

# cv2.imshow("blur",blur)
# cv2.imshow("blur0",blur0)
# cv2.imshow("blur1",blur1)
# cv2.imshow("blur2",blur2)
# cv2.waitKey(0)

hsv = cv2.cvtColor(blur2, cv2.COLOR_BGR2HSV)
cv2.imshow("H",hsv[:,:,0])
cv2.imshow("S",hsv[:,:,1])
cv2.imshow("V",hsv[:,:,2])
low_blue = np.array([55, 0, 0])
high_blue = np.array([118, 255, 255])
mask = cv2.inRange(hsv, low_blue, high_blue)
res = cv2.bitwise_and(img,img, mask= mask)

cv2.imshow("mask",mask)
cv2.imshow("res",res)
# cv2.imshow("blur2",blur2)
cv2.waitKey(0)

