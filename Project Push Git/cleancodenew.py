import cv2
import numpy as np
import math
import glob
import os
import time
import PLCController
from PLCController import PLC
import DBconfig
# from DBconfig import firebase
from datetime import datetime
import qrcode
import snap7
from snap7.util import *
from snap7.types import *
import snap7.client as c
import pyrebase

config = {
    "apiKey": "AIzaSyCj8R0iJmoT-hlfETLGdTYxzk5VUQ9CLBw",
    "authDomain": "mechatronic-project-af507.firebaseapp.com",
    "databaseURL": "https://mechatronic-project-af507-default-rtdb.firebaseio.com",
    "projectId": "mechatronic-project-af507",
    "storageBucket": "mechatronic-project-af507.appspot.com",
    "messagingSenderId": "782997268535",
    "appId": "1:782997268535:web:0f36553a1637a1400977b2"
    

};


firebase = pyrebase.initialize_app(config)

storage = firebase.storage()
database = firebase.database()

Mass_Out = 0
flag_object = False
flag_defect = False
flag_PLC = True

MeetStandardIMGProcessing = False
# Lấy ngày và giờ hiện tại
current_datetime = datetime.now()

# Định dạng ngày tháng
formatted_date = current_datetime.strftime("%Y-%m-%d")
count = 0
list_Weights = []
def empty(a):
    pass
cv2.namedWindow("Tracking")
cv2.resizeWindow("Tracking",640,240)
cv2.createTrackbar("LH", "Tracking", 97, 255, empty) 
cv2.createTrackbar("LS", "Tracking", 0, 255, empty)
cv2.createTrackbar("LV", "Tracking", 0, 255, empty)
cv2.createTrackbar("UH", "Tracking", 106, 255, empty)
cv2.createTrackbar("US", "Tracking", 174, 255, empty)
cv2.createTrackbar("UV", "Tracking", 255, 255, empty)


class DetectObject:
    def ElipseContours(self, img, imgContour):
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            selected_contour = max(contours, key=lambda x: cv2.contourArea(x))
            # Config area to detect object value 
            areaMin = 1000 

            if area > areaMin:
                print("Area of object durian (pixel): ",area) 
                cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
                M = cv2.moments(cnt)
                cx= int(M["m10"]/M["m00"])
                cy= int(M["m01"]/M["m00"])  
                
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02* peri, True) 
                x, y, w, h = cv2.boundingRect(approx)
                ellipse = cv2.fitEllipse(selected_contour)
               
                center = ellipse[0]
                semi_majorAxis = (ellipse[1][0])/2
                semi_minorAxis = (ellipse[1][1])/2
                angle = ellipse[2]

                area_elipse = math.pi * semi_majorAxis * semi_minorAxis
                area_elipse = "{:.3f}".format(area_elipse)
                area_elipse = float(area_elipse)
                print("Area of the elipse classification (pixel):", area_elipse)

                result_sub = area_elipse - area
                result_percent = result_sub/area_elipse
                result_percent = "{:.3f}".format(result_percent)
                result_percent = float(result_percent)
                print("Area of substraction (pixel): ",result_percent)
                if (result_percent < 0.2): # 20% 
                    print("Durian meet standards")
                    meetStandard = True
                else:
                    print("Durian does not meet standards")
                    meetStandard = False
                cv2.ellipse(imgContour, ellipse, (0, 255, 0), 3)
                cv2.circle(imgContour,(cx,cy),7,(0,0,255),-1)
                cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 5)

                print(f"meetStandard:{meetStandard}")
                return meetStandard

    def getResultObject(self,image):
        global flag_object
        image = cv2.resize(image,(400,300))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Warming threshold needed apdative
        thresh, output_otsuthresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Erosion image to detect Object Elipse for Durian 
        kernel = np.ones((3,3),np.uint8)
        output_erosion = cv2.erode(output_otsuthresh, kernel,iterations=2)
        output_dilate = cv2.dilate(output_otsuthresh, kernel,iterations=4)
        boder =  output_dilate - output_erosion 
        # Detect Contour and measure the area durian object 
        resultObject = self.ElipseContours(boder,image)
        print(f"resultObject:{resultObject}")
        flag_object = True
        return resultObject



class DetectDefect:
    def __init__(self):
        pass
    def sharpen_image_laplacian(self, image):
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        sharpened_image = np.uint8(np.clip(image - 0.3*laplacian, 0, 255))
        return sharpened_image

    def RectangleContours(self, img, imgContour):
        list_area = []
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            list_area.append(area)
            selected_contour = max(contours, key=lambda x: cv2.contourArea(x))

            # Config area to detect defect of durian
            areaMin = 500 
            if area > areaMin:
                cv2.drawContours(imgContour, cnt, -1, (255, 0, 255),5)
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.009* peri, True)  
                x, y, w, h = cv2.boundingRect(approx)
                cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 5)


        S = sorted(list_area,key=None,reverse=True)
        print("S : ",S[0])
        if S[0]< areaMin : 
            Defect = False
        else : 
            Defect = True
        return Defect
    def getResultDefect(self,image):
        global flag_defect
        sharpened_image = self.sharpen_image_laplacian(image)
        rgb_img = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB)

        # Convert BGR to HSV 
        HSV_img = cv2.cvtColor(rgb_img,cv2.COLOR_BGR2HSV)

        # Set range for red color and  
        l_h = cv2.getTrackbarPos("LH", "Tracking")
        l_s = cv2.getTrackbarPos("LS", "Tracking")
        l_v = cv2.getTrackbarPos("LV", "Tracking")

        u_h = cv2.getTrackbarPos("UH", "Tracking")
        u_s = cv2.getTrackbarPos("US", "Tracking")
        u_v = cv2.getTrackbarPos("UV", "Tracking")

        l_b = np.array([l_h, l_s, l_v])
        u_b = np.array([u_h, u_s, u_v])

        mask = cv2.inRange(HSV_img, l_b, u_b)

        # Morphological and Dilate
        kernel = np.ones((5,5),np.uint8)
        mask_morpho = cv2.morphologyEx(mask, cv2.MORPH_OPEN,kernel)
        mask_dilate = cv2.dilate(mask_morpho, kernel,iterations=2)
        res = cv2.bitwise_and(image,image, mask=mask_dilate)

        # Detecting contours in image
        thresh, output_threshold = cv2.threshold(res,105, 255, 1, cv2.THRESH_BINARY)
        gray_image = cv2.cvtColor(output_threshold, cv2.COLOR_BGR2GRAY)
        bitwise_img = cv2.bitwise_not(gray_image)
        # Detecting contours in image
        
        resultDefect = self.RectangleContours(bitwise_img,image)
        print(f"resultDefect:{resultDefect}")
        flag_defect = True
        return resultDefect
    
#  Innovate class and cofig again
class PLCVal():
     
    def getWeightsSample(self):
        # global count
        global Mass_Out
        RL_chan = PLC.ReadMemory(3,1,S7WLBit)
        RL_le = PLC.ReadMemory(3,2,S7WLBit)
        if RL_chan == True and RL_le == False:
            Mass_Out = PLC.ReadMemory(50,0,S7WLWord)  ## MW50 , MW54
            list_Weights.append(Mass_Out)
            # count += 1
        elif RL_chan == False and RL_le == True:
            Mass_Out = PLC.ReadMemory(54,0,S7WLWord)
            list_Weights.append(Mass_Out)
            # count = 0
        return Mass_Out
    def getResult(self):
        global flag_PLC
        # RL_getLoadcellValue = PLC.ReadMemory(4,2,S7WLBit)


        ############################################ GET VALUE LOADCELLS #############################
        # if RL_getLoadcellValue == True:
        SampleWeight = self.getWeightsSample()
        flag_PLC = False
        return SampleWeight

def qrConfig():
    global count
    # Data to encode
    data = "https://haviet12.github.io/UI_Durian-s_Infor/?custom_param=Sample" + str(count)
    
    # Creating an instance of QRCode class
    qr = qrcode.QRCode(version = 1,
                    error_correction = qrcode.constants.ERROR_CORRECT_L,
                    box_size = 20,
                    border = 2)
    
    # Adding data to the instance 'qr'
    qr.add_data(data)
    
    qr.make(fit = True)
    img = qr.make_image(fill_color = 'black',
                        back_color = 'white')
    path_save_qr ="Image_QR/" + "QR_Sample" + str(count) +".png"
    img.save(path_save_qr)
    print("Successsssssssssssssss")

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

folder_IMG_RmBG = "Image_RMBG"
defect = DetectDefect()
object = DetectObject()
PLC_val = PLCVal()

count_img = 0
while True:
    RL_getLoadcellValue = PLC.ReadMemory(4,2,S7WLBit)

    sensor1 = PLC.ReadMemory(0,3,S7WLBit)
    if sensor1 == True:
        count_img =0
    # print(f"Sensor1 :{sensor1}")
    # if sensor1 == True:
    #     flag_PLC == True 
    #     print(' sjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjj')
    # ########################################### GET VALUE LOADCELLS #############################
    if RL_getLoadcellValue == True :
    #     SampleWeight = PLC_val.getResult()
        
    #     print(f"Mass_Out : {SampleWeight}")
    # else :pass 
        SampleWeight = PLC_val.getWeightsSample()
    
    # print(f"flag_PLC:{flag_PLC}")
    # print(f"RL_getLoadcellValue:{RL_getLoadcellValue}")
    ############################################ GET VALUE LOADCELLS #############################

    list_path_RMBG=[]
    # Define the pattern for image files (you can add more extensions if needed)
    image_files = os.listdir(folder_IMG_RmBG)
   
    # Get a list of all image files in the directory
    # Check if the list is empty
    if not image_files:
        print("FOLDER DON'T HAVE ANY IMAGE")
        pass

    else:
        for image_file in image_files :
            path = os.path.join(folder_IMG_RmBG,image_file)
            list_path_RMBG.append(path)
    # Use the max function with a lambda to find the file with the latest modification time
        newest_image = max(list_path_RMBG, key=os.path.getmtime)
        origin_img = newest_image.split('/')
        origin_img_path = "Image_Original/" +origin_img[1]
        if not newest_image:
            print("DON'T HAVE ANY NEW FILE")
            pass
        else:
            count_img  += 1
            if count_img  ==1 :

                path_file = "/home/pi/Mechatronics_Project/Mechatronics-Project/" + newest_image
                time.sleep(0.1)
                image_original = cv2.imread(path_file)

                if image_original is None:
                    print("Don't have img")
                    break

    ####################################### GET RESULT IMAGE PROCESSING #####################################

                imgToDetectObject = image_original.copy()
                imgToDetectDefect = image_original.copy()

                resultDefect=defect.getResultDefect(imgToDetectDefect)
                resultObject=object.getResultObject(imgToDetectObject)

                # SHOW THE IMAGE IN TERMINAL
                imgstack = stackImages(0.8,([image_original,image_original],[object.getResultObject(),defect.getResultDefect(image_original)]))
                cv2.imshow("The Image of the Project",imgstack)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    
                if resultObject == True and resultDefect == False:
                    print("########################### Meet Standard IMG Processing ##############")
                    MeetStandardIMGProcessing = True

                    
                else :
                    print("########################### Not Meet Standard IMG Processing ##############")


####################################### GET RESULT IMAGE PROCESSING #####################################



    if flag_object == True and flag_defect == True:
        print(f"Mass_Out : {SampleWeight}")
        path_original_img = "/home/pi/Mechatronics_Project/Mechatronics-Project/" + origin_img_path
      
        if MeetStandardIMGProcessing == True:
            if SampleWeight >1800  and SampleWeight<5000 :
                count += 1
                print("########################### Meet Standard Type 1 ##############")
                database.child("Sample"+str(count))
                data = {"Weight": SampleWeight, "Name": "Thai", "Type": 1, "Orgin":"Lam Dong", "Date_Export": formatted_date}
                database.set(data)
                print("PUSH DATA SUCCESSFUL")
                storage.child("Sample"+str(count)+".JPG").put(path_original_img)
                qrConfig()

                flag_object = False
                flag_defect = False
            elif (SampleWeight >1400  and SampleWeight <1800) or SampleWeight >5000 :
                count += 1
                print("########################### Meet Standard Type 2 ##############")
                database.child("Sample"+str(count))
                data = {"Weight": SampleWeight, "Name": "Thai", "Type": 2, "Orgin":"Lam Dong", "Date_Export": formatted_date}
                database.set(data)
                print("PUSH DATA SUCCESSFUL")
                storage.child("Sample"+str(count)+".JPG").put(path_original_img)
                qrConfig()
                flag_object = False
                flag_defect = False
        elif  MeetStandardIMGProcessing == False:
                print(" SAMPLE NOT MEET STANDARD")
                pass

        