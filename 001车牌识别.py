import cv2
import imutils
import numpy as np
import pytesseract


image = cv2.imread("images\\001\\1.png")
image = cv2.resize(image,(620,480))#跟图像大小相关，（600,400）时检测不到


imagea = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray1 = cv2.bilateralFilter(imagea, 13, 15, 15)
edged = cv2.Canny(gray1, 30, 200)#杂纹
cv2.imwrite("images\\001\\edged.jpg",edged)

#车牌定位很差，大部分都找不到
#找面积最大的4四边形
contours,_ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#contours = imutils.grab_contours(contours)#返回轮廓
contours = sorted(contours,key=cv2.contourArea,reverse=True)
screenCnt = []
for i in range(10):
    c = contours[i]
    peri = cv2.arcLength(c,True)#闭合的周长
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    x, y, w, h = cv2.boundingRect(c)
    image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 6)
    #cv2.drawContours(image, [c], -1, (0, 0, 255), 10)
    if len(approx)==4:
        screenCnt.append(c)
        #image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 6)
        # break#不能确定是车牌,有可能是格栅
cv2.imwrite("images\\001\\imagea.jpg",image)


##图像裁剪
for i in range(len(screenCnt)):
    x, y, w, h = cv2.boundingRect(screenCnt[i])
    Cropped = imagea[y:y+h,x:x+w]
    text = pytesseract.image_to_string(Cropped, config='--psm 11')
    text = filter(str.isalnum, text)
    text = ''.join(list(text))
    if len(text)==7:
        print("车牌结果：", text)
        cv2.imshow("Cropped",Cropped)
        cv2.waitKey(0)


