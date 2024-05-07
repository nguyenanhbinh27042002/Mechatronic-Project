import cv2
import numpy as np
import rembg
from scipy.spatial import distance
from rembg import remove 
from PIL import Image
import imutils
import os


# def sharpen_image_laplacian(image):
#     laplacian = cv2.Laplacian(image, cv2.CV_64F)
#     sharpened_image = np.uint8(np.clip(image - 0.3*laplacian, 0, 255))
#     return sharpened_image  # Return the sharpened image

# Path = '..\Project Push Git\Image'
# Files = os.listdir(Path)
# for File in Files : 
#     imgPath = os.path.join(Path,File)
#     print(imgPath)
#     image = cv2.imread(imgPath)
#     rm = File.rsplit('.', maxsplit=1)[0]
#     #cv2.imshow("Image not Remove Background",image)
#     # Remove Background and xoa bong den
#     input_path =  Path + File
#     output_path = 
#     #input_img = Image.open(image)
#     # output = remove(input_img)
#     # output.save(out)   
#     out = cv2.imwrite(f'../Project Push Git/Result Remove Background/rmbg_{File}',image)   # Write the image rm background in folder Result

def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: 
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: 
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


def getcoutours(img, imgContour):
    contours, hierachy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print("Area of Contour",area)
        selected_contour = max(contours, key=lambda x: cv2.contourArea(x))
        areaMin = 1000 # Config area

        if area > areaMin:  
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
            M = cv2.moments(cnt)
            cx= int(M["m10"]/M["m00"])
            cy= int(M["m01"]/M["m00"])  
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            ellipse = cv2.fitEllipse(selected_contour)
            cv2.ellipse(image, ellipse, (0, 255, 0), 3)
            cv2.circle(image,(cx,cy),7,(0,0,255),-1)
            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 5)  
            # cv2.putText(imgContour, "Area: " + str(int(area)), (x + w + 40, y + 65), cv2.FONT_HERSHEY_COMPLEX, 0.7,
            #             (0, 255, 0), 2)
            

while True:
    image = cv2.imread("Result Remove Background\sample No.16.png")
    image = cv2.resize(image,(400,300))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Warming threshold needed apdative
    # Convert Binary Image using 3 method
    #thresh, output_binthresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)    
    thresh, output_otsuthresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    output_adapthresh = cv2.adaptiveThreshold (gray,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,51, 0)  #51
    

    # Erosion image to detect Object Elipse for Durian 
    kernel = np.ones((3,3),np.uint8)
    output_erosion = cv2.morphologyEx(output_adapthresh, cv2.MORPH_OPEN,kernel)
    output_erosion = cv2.erode(output_adapthresh, kernel,iterations=2)
    output_erosion = cv2.dilate(output_adapthresh, kernel,iterations=4)
    
    # Detect Contour and measure the area durian object 
    getcoutours(output_erosion,image)
    imgstack = stackImages(0.8, ([image, gray, output_erosion], [output_adapthresh, image, gray]))

    #cv2.imshow("Binary Threshold (fixed)", output_binthresh)
    # cv2.imshow("Image original",image)
    # cv2.imshow("Binary Threshold (otsu)", output_otsuthresh)
    # cv2.imshow("Adaptive Thresholding", output_adapthresh)
    # cv2.imshow("Erosion", output_erosion)
    cv2.imshow("Result Image", imgstack)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
cv2.destroyAllWindows()

