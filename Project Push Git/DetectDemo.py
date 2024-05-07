import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=25,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
    
        #320 horizontal position, 250 vertical position, 80 horizontal size, 120 vertical size
        #0 angle, 0 startangle, 360 endangle
        #(0, 255, 0) color, 2 thickness
    
        startAngle = 0
        endAngle = 10
        for z in range(20):
            cv2.ellipse(frame, (320, 250), (80, 120), 0, startAngle, endAngle, (255, 255, 255), 2)
            startAngle = startAngle+20
            endAngle = endAngle+20
    
        #Centered Vertical Dashed Line
        xCord = 320
        yCord = 400
        for z in range(12):
            cv2.line(frame, (xCord, yCord), (xCord, yCord-20), (255, 255, 255), 2)
            yCord = yCord-25

        #Upper Horizontal Dashed Line
        xCord = 160
        yCord = 130
        for z in range(13):
            cv2.line(frame, (xCord, yCord), (xCord+20, yCord), (255, 255, 255), 2)
            xCord = xCord+25
    
        #Lower Horizontal Dashed Line
        xCord = 160
        yCord = 370
        for z in range(13):
            cv2.line(frame, (xCord, yCord), (xCord+20, yCord), (255, 255, 255), 2)
            xCord = xCord+25
    
        cv2.putText(frame, 'Head', (330, 120), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (255, 255, 255), 2) 
        cv2.putText(frame, 'Chin', (330, 390), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (255, 255, 255), 2) 

        # Display the resulting frame
        cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('c'):
        crop_img = frame[y-50: y+h+10, x: x+w] # Crop from x, y, w, h -> 100, 200, 300, 400, y-50 is to include head in the picture too y+h+10 is to include chin.
        cv2.imwrite("media/faces/face.jpg", crop_img)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

# mask = np.zeros_like(frame)
#     rows, cols,_ = mask.shape
#     mask=cv2.ellipse(mask, (320, 250), (80, 120), 0, 0, 360, (255,255,255), -1)
#     result = np.bitwise_and(frame,mask)
#     bgr = result[:,:,:3] # Channels 0..2
#     gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
#     # Some sort of processing...
#     bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
#     alpha = result[:,:,3] # Channel 3
#     result = np.dstack([bgr, alpha])
#     cv2.imwrite("media/faces/face.png", result)