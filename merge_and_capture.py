import cv2 
import numpy as np
from snap7.util import *
from snap7.types import *
import snap7.client as c
import snap7
import os



class IMG_Processing():
    def __init__(self):
        pass

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



    def getcoutours(frame_gray,frame_countour):
        _,contours,hierachy = cv2.findContours(frame_gray,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            print(area)
            areaMin = cv2.getTrackbarPos("Area","Parameters")
            if area > areaMin: # nho cau hinh lai
                cv2.drawContours(frame_countour,cnt,-1,(255,0,255),7) # 255,0,255 : mau tim
                peri = cv2.arcLength(cnt,True)
                approx = cv2.approxPolyDP(cnt,0.02*peri,True)
                x,y,w,h = cv2.boundingRect(approx)
                cv2.rectangle(frame_countour,(x,y),(x+w,y+h),(0,255,0),5) # 0,255,0 : mau xanh la cay
                cv2.putText(frame_countour,"Area: "+ str(int(area)),(x+w+40,y+65),cv2.FONT_HERSHEY_COMPLEX,0.7,
                            (0,255,0),2)
class PLC():
    def __init__(self) :
        # self.plc = c.Client()

        pass
        # self.plc.connect(ip,rack,slot)    
    # def connect(ip,rack,slot):
    #     global plc 
    #     plc = c.Client()
    #     plc.connect(ip,rack,slot)   
    #     flag = plc.get_connected()   
    #     if flag == True:
    #         print("PLC Connect Success...........")
    #     else: print("Connect Error")

    def WriteMemory(plc, byte, bit, datatype, value):
        result = plc.read_area(Areas.MK, 0, byte, datatype)
        if datatype == S7WLBit:
            set_bool(result, 0, bit, value)
        elif datatype == S7WLByte or datatype == S7WLWord:
            set_int(result, 0, value)
        elif datatype == S7WLReal:
            set_real(result, 0, value)
        elif datatype == S7WLDWord:
            set_dword(result, 0, value)
        plc.write_area(Areas.MK, 0, byte, result)


    def ReadMemory(plc, byte, bit, datatype):
        result = plc.read_area(Areas.MK, 0, byte, datatype)
        if datatype == S7WLBit:
            return get_bool(result, 0, bit)
        elif datatype == S7WLByte or datatype == S7WLWord:
            return get_int(result, 0)
        elif datatype == S7WLReal:
            return get_real(result, 0)
        elif datatype == S7WLDWord:
            return get_dword(result, 0)
        else:
            return None
class capture_img():
    def __init__(self) -> None:
        pass
    
 
    def capture(plc,byte, bit, datatype):
        global video
        sensor=PLC.ReadMemory(plc,byte,bit,datatype)
        ret,frame = video.read()
        print(f"sensor:{sensor}")
        if sensor == True:
            global count
            count += 1
            folder = "/home/pi/Mechatronics_Project/Mechatronics-Project/Image/Sample" + str(count)
            if not os.path.exists(folder):
                os.makedirs(folder)
            # global img
            # frame = img.copy()
            cv2.imwrite(folder +"/"+"sample No."+str(count) +".JPG", frame)
            print('capture success......................')
            # time.sleep(1)
           
############################ TRACKBAR #########################################

def empty(a):
    pass
cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters",640,240)
cv2.createTrackbar("Thershold1","Parameters",100,255,empty)
cv2.createTrackbar("Thershold2","Parameters",60,255,empty)
cv2.createTrackbar("Area","Parameters",5000,30000,empty)

##################################################################################



video = cv2.VideoCapture(0)
count =0

############### connect to PLC########################
ip = '192.168.0.1'
rack =0
slot =0
# PLC.connect(ip,rack,slot)
plc = c.Client()
plc.connect(ip,rack,slot)   
flag = plc.get_connected()   
if flag == True:
    print("PLC Connect Success...........")
else: print("Connect Error")
# while(True):
#     print(PLC.ReadMemory(plc,0,0,S7WLBit))
# #####################################################

############## set Frame image #####################
framewidth = 640
frameheight = 480
video.set(3,framewidth)
video.set(4,frameheight)
####################################################
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
    IMG_Processing.getcoutours(imgDil,imgContour)
    imgstack = IMG_Processing.stackImages(0.8,([img,gray,imgCany],
                                [imgDil,imgContour,opening]))


    cv2.imshow("Result Camera",imgstack)

    capture_img.capture(plc,0,0,S7WLBit)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release
cv2.destroyAllWindows()
# # while (True):
# #     print(PLC.ReadMemory(0,0,S7WLBit))


            