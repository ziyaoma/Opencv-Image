#导入库
import cv2
import numpy as np
import sys
from PIL import Image
#图像显示函数
def show(name,img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#显示匹配到的特征点
def drawMatches(imageA, imageB, kpsA, kpsB, matches):
    # 初始化可视化图片，将A、B图左右连接到一起
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB
    # 联合遍历，画出匹配对
    for (trainIdx, queryIdx) in matches:
        # 当点对匹配成功时，画到可视化图上
        # 画出匹配对
        ptA = (int(kpsA[trainIdx][0]), int(kpsA[trainIdx][1]))
        ptB = (int(kpsB[queryIdx][0]) + wA, int(kpsB[queryIdx][1]))
        cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
    # 返回可视化结果
    return vis


# 检测A、B图片的SIFT关键特征点，并计算特征描述子
def detectAndDescribe(image):
    # 建立SIFT生成器
    #sift = cv2.xfeatures2d.SURF_create()
    sift = cv2.SIFT_create()
    # 检测SIFT特征点，并计算描述子
    (kps, features) = sift.detectAndCompute(image, None)
    # 将结果转换成NumPy数组
    # cv2.imshow("kpsA", kps)
    # #cv2.imshow("kpsB", kpsB)
    # cv2.waitKey(0)

    kps = np.float32([kp.pt for kp in kps])
    # 返回特征点集，及对应的描述特征
    return (kps, features)


#读取输入图片
ima = cv2.imread("images//3//view81.png")
imb = cv2.imread("images//3//view8.png")
# ima = cv2.imread("images//3//1_02.jpg")
# imb = cv2.imread("images//3//1_01.jpg")
A = ima.copy()
B = imb.copy()
imageA = cv2.resize(A,(0,0),fx=0.2,fy=0.2)
imageB = cv2.resize(B,(imageA.shape[1],imageA.shape[0]))

#检测A、B图片的SIFT关键特征点，并计算特征描述子
kpsA, featuresA = detectAndDescribe(imageA)
kpsB, featuresB = detectAndDescribe(imageB)
#
# cv2.imshow("kpsA",kpsA)
# cv2.imshow("kpsB",kpsB)
# cv2.waitKey(0)


# 建立暴力匹配器
bf = cv2.BFMatcher()
# 使用KNN检测来自A、B图的SIFT特征匹配对，K=2
matches = bf.knnMatch(featuresA, featuresB, 2)
good = []
for m in matches:
    # 当最近距离跟次近距离的比值小于ratio值时，保留此匹配对
    if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
        # 存储两个点在featuresA, featuresB中的索引值
        good.append((m[0].trainIdx, m[0].queryIdx))


vis = drawMatches(imageB, imageA, kpsB, kpsA, good)
cv2.imwrite("images\\3\\res81.jpg",vis)

# 当筛选后的匹配对大于4时，计算视角变换矩阵
if len(good) > 4:
    # 获取匹配对的点坐标
    ptsA = np.float32([kpsA[i] for (_, i) in good])
    ptsB = np.float32([kpsB[i] for (i, _) in good])
    # 计算视角变换矩阵
    H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,4.0)

# 匹配两张图片的所有特征点，返回匹配结果
M = (matches, H, status)
# 如果返回结果为空，没有匹配成功的特征点，退出程序
if M is None:
    print("无匹配结果")
    sys.exit()
# 否则，提取匹配结果
# H是3x3视角变换矩阵
(matches, H, status) = M
# 将图片A进行视角变换，result是变换后图片
result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
# 将图片B传入result图片最左端
result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
#show('res',result)
cv2.imwrite("images\\3\\res8.jpg",result)
print(result.shape)

