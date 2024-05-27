from rembg import remove 
from PIL import Image
import cv2
import numpy as np
import imutils

def sharpen_image_laplacian(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    sharpened_image = np.uint8(np.clip(image - 0.3*laplacian, 0, 255))
    return sharpened_image  # Return the sharpened image

img1 = cv2.imread('testdefect.jpg')
img_reszie = cv2.resize(img1, (200, 300))
cv2.imwrite('img.png', img_reszie, [cv2.IMWRITE_PNG_COMPRESSION, 0])

inp = r'img.png'
out = r'imgok.png'

input_img = Image.open(inp)
output = remove(input_img)
output.save(out)

img = cv2.imread('imgok.png')
Img = np.array(img)
curImg = np.array(img)

# Sharpen the image
sharpen_img = sharpen_image_laplacian(Img)
# Convert to grayscale
gray = cv2.cvtColor(np.array(sharpen_img), cv2.COLOR_BGR2GRAY)
# Thresholding
_, thresh = cv2.threshold(gray,150,255, cv2.THRESH_BINARY)

# remove_noise:
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,kernel, iterations=1)
curImg = opening

# # Find contours
# contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Detect Defect 
#vùng sure background
sure_bg = cv2.dilate(thresh,kernel, iterations=3)
 # Tìm vùng sure foreground
dist_transform = cv2.distanceTransform(curImg, cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform, 0.05* dist_transform.min(),125,0)

# tìm vùng chưa xác định
sure_fg = np.uint8(sure_fg)
#vùng backround - vùng foreground còn lại là vùng không xác định
unknown = cv2.subtract(sure_bg, sure_fg)

# ghi nhãn điểm đánh dấu
ret, markers = cv2.connectedComponents(sure_fg)

# thêm 1 vào tất cả các nhãn để sure background không phải là 0 mà là 1
markers = markers + 1

# giờ đánh dấu vùng chưa xác định = 0
markers[unknown == 255] = 0
markers = cv2.watershed(Img, markers)
Img[markers == -1] = [255, 0, 0]


processImage = Img
#  Code chay dc chua toi uu 
contours = cv2.findContours(sure_bg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
cnts =imutils.grab_contours(contours)
img_result = sharpen_img.copy()
countContours = 0
for cnt in cnts:
    c_area = cv2.contourArea(cnt)
    print(c_area)
    if c_area >=100 and c_area <3000:
        countContours += 1
        cv2.drawContours(img_result,[cnt], -1, (0,255,0) , 2)
    if c_area < 100 and c_area >400:
        countContours += 1
        cv2.drawContours(img_result,[cnt], -1, (0,255,0) , 2)

# cv2.imshow('shapern_img', sharpen_img)
# cv2.imshow("img", img_reszie)
cv2.imshow("dilate",sure_bg)
cv2.imshow("Thresh", thresh)
cv2.imshow("Opening",opening)
cv2.imshow('img_draw', img_result)
#cv2.imshow("processImage",processImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
