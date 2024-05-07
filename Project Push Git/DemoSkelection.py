import cv2
import numpy as np
from math import sqrt
from collections import OrderedDict

def findangle(x1,y1,x2,y2,x3,y3):
    ria = np.arctan2(y2 - y1, x2 - x1) - np.arctan2(y3 - y1, x3 - x1)
    if ria > 0:
        if ria < 3:
            webangle = int(np.abs(ria * 180 / np.pi))
        elif ria > 3:
            webangle = int(np.abs(ria * 90 / np.pi))
    elif ria < 0:
        if ria < -3:
            webangle = int(np.abs(ria * 90 / np.pi))
            
        elif ria > -3:
            webangle = int(np.abs(ria * 180 / np.pi))
    return webangle



image = cv2.imread("Result Remove Background\sample No.1.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
output_adapthresh = cv2.adaptiveThreshold (gray,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,51, 0)  #51

# Mapping project in File ProjectFinish
# Erosion image to detect Object Elipse for Durian 
kernel = np.ones((3,3),np.uint8)
output_erosion = cv2.morphologyEx(output_adapthresh, cv2.MORPH_OPEN,kernel)
output_erosion = cv2.erode(output_adapthresh, kernel,iterations=2)
output_erosion = cv2.dilate(output_adapthresh, kernel,iterations=4)

contours, hierarchy = cv2.findContours(output_erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
selected_contour = max(contours, key=lambda x: cv2.contourArea(x))
# Draw Contour
approx = cv2.approxPolyDP(selected_contour, 0.0035 * cv2.arcLength(selected_contour, True), True)
for point in approx:
    cv2.drawContours(image, [point], 0, (0, 0, 255), 3)
convexHull = cv2.convexHull(selected_contour,returnPoints=False)
cv2.drawContours(image, cv2.convexHull(selected_contour), 0, (0, 255, 0), 3)
convexHull[::-1].sort(axis=0)
convexityDefects = cv2.convexityDefects(selected_contour, convexHull)
start2,distance=[],[]
for i in range(convexityDefects.shape[0]):
    s, e, f, d = convexityDefects[i, 0]
    start = tuple(selected_contour[s][0])
    end = tuple(selected_contour[e][0])
    far = tuple(selected_contour[f][0])
    start2.append(start)
    cv2.circle(image, start, 2, (255, 0, 0), 3)
    cv2.line(image,start,end , (0, 255, 0), 3)
    distance.append(d)
distance.sort(reverse=True)
for i in range(convexityDefects.shape[0]):
    s, e, f, d = convexityDefects[i, 0]
    if distance[0]==d:
       defect={"s":s,"e":e,"f":f,"d":d}


cv2.circle(image, selected_contour[defect.get("f")][0], 2, (255, 0, 0), 3)
cv2.circle(image, selected_contour[defect.get("s")][0], 2, (0, 0, 0), 3)
cv2.circle(image, selected_contour[defect.get("e")][0], 2, (0, 0, 255), 3)
x1, y1 = selected_contour[defect.get("f")][0]
x2, y2 = selected_contour[defect.get("e")][0]
x3, y3 = selected_contour[defect.get("s")][0]
cv2.line(image,(x1,y1),(x2,y2),(255,200,0),2)
cv2.line(image,(x1,y1),(x3,y3),(255,200,0),2)
cv2.putText(image, "Web  Angle : " + str((findangle(x1,y1,x2,y2,x3,y3))), (50, 200), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0,0,0),2,cv2.LINE_AA)
cv2.imshow("frame",image)
cv2.imshow("Output Erosion",output_erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()