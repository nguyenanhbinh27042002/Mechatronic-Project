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
# img_yuv = cv2.cvtColor(sharpen_img, cv2.COLOR_BGR2YUV)
# img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
# # convert the YUV image back to RGB format
# img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

# Thresholding
_, thresh = cv2.threshold(gray,150,255, cv2.THRESH_BINARY)



# # Find contours
# contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
kernel = np.ones((3, 3), np.uint8)
# Detect Defect 
#vùng sure background
sure_bg = cv2.dilate(thresh,kernel, iterations=3)
# remove_noise:

opening = cv2.morphologyEx(sure_bg, cv2.MORPH_OPEN,kernel, iterations=1)
curImg = opening
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
max_area = 0
# Ngưỡng khoảng cách để xác định contour gần nhau
distance_threshold = 50
for i,cnt in enumerate(cnts):
    for cnt1 in cnt[i+1:]:
        # Tính toán khoảng cách giữa hai contour
        distance = cv2.pointPolygonTest(cnt1, tuple(cnt[0][0]), True)
        # Nếu khoảng cách nhỏ hơn ngưỡng, nối contour2 vào contour1
        if abs(distance) < distance_threshold:
            contours[i] = np.concatenate((cnt, cnt1))
            contours.remove(cnt1)
# Vẽ các contour đã nối lại lên ảnh gốc
result = np.copy(sharpen_img)
cv2.drawContours(result, cnts, -1, (0, 255, 0), 2)
    # c_area = cv2.contourArea(cnt)
# Calculate bounding rectangle
    # x, y, w, h = cv2.boundingRect(cnt)
    # # Count white pixels within the bounding rectangle
    # dem = np.sum(thresh[y:y+h, x:x+w] == 255)
    # # Update max area and calculate the real area
    # if dem > max_area:
    #     max_area = dem
    #     area = max_area * (1369 / 4260) * (851 / 2810) * 0.1
    # print("Area of defect",area)
    
#     print(c_area)
#     # code chua toi uu dc area : tinhs lai 
#     if c_area >=100:
#         countContours += 1
#     cv2.drawContours(img_result,[cnt], -1, (0,255,0) , 2)
#     # if c_area < 100 and c_area >400:
#     #     countContours += 1
#     #     cv2.drawContours(img_result,[cnt], -1, (0,255,0) , 2)
# img_resulthsv= img_result.copy()

# cv2.imshow('shapern_img', sharpen_img)
# cv2.imshow("img", img_reszie)
cv2.imshow("dilate",sure_bg)
cv2.imshow("Thresh", thresh)
cv2.imshow("Opening",opening)
cv2.imshow('img_draw', img_result)
#cv2.imshow("processImage",processImage)
# Hiển thị ảnh kết quả
cv2.imshow("Connected Contours", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
