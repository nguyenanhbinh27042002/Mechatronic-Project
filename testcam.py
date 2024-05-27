import cv2 
import numpy as np

video = cv2.VideoCapture(0)
framewidth = 640
frameheight = 480
video.set(3,framewidth)
video.set(4,frameheight)


def empty(a):
    pass
cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters",640,240)
cv2.createTrackbar("Thershold1","Parameters",100,255,empty)
cv2.createTrackbar("Thershold2","Parameters",60,255,empty)
cv2.createTrackbar("Area","Parameters",5000,30000,empty)

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver



def getcoutours(img,imgContour):
    contours,hierachy = cv2.findContours(img,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        areaMin = cv2.getTrackbarPos("Area","Parameters")
        if area > areaMin: # nho cau hinh lai
            cv2.drawContours(imgContour,cnt,-1,(255,0,255),7) # 255,0,255 : mau tim
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            x,y,w,h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),5) # 0,255,0 : mau xanh la cay
            cv2.putText(imgContour,"Area: "+ str(int(area)),(x+w+40,y+65),cv2.FONT_HERSHEY_COMPLEX,0.7,
                        (0,255,0),2)

while True: 
    ret,img = video.read()
    imgContour = img.copy()
    imgblur = cv2.GaussianBlur(img,(7,7),1)
    gray = cv2.cvtColor(imgblur, cv2.COLOR_BGR2GRAY)

    threshold1 = cv2.getTrackbarPos("Thershold1","Parameters")
    threshold2 = cv2.getTrackbarPos("Thershold2","Parameters")

    imgCany = cv2.Canny(gray,threshold1,threshold2)
    # Create the array 3x3 vailue 1
    kernel = np.ones((3, 3), np.uint8)
    # Dilation the object 
    imgDil = cv2.dilate(imgCany,kernel,iterations=2)
    opening = cv2.morphologyEx(imgDil,cv2.MORPH_OPEN,kernel, iterations=2)
    getcoutours(imgDil,imgContour)
    imgstack = stackImages(0.8,([img,gray,imgCany],
                                [imgDil,imgContour,opening]))


    cv2.imshow("Result Camera",imgstack)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release
cv2.destroyAllWindows()