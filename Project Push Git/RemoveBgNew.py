# # from typing import Tuple, List
# # import cv2 
# # import numpy as np
# # from skimage import measure
# # from matplotlib import pyplot as plt





# # img = cv2.imread('Image/sample No.14.JPG')
# # def sharpen_image_laplacian(image):
# #     laplacian = cv2.Laplacian(image, cv2.CV_64F)
# #     sharpened_image = np.uint8(np.clip(image - 0.3*laplacian, 0, 255))
# #     return sharpened_image  # Return the sharpened image

# # def bgremove3(myimage):
# #     # BG Remover 3
# #     myimage_hsv = cv2.cvtColor(myimage, cv2.COLOR_BGR2HSV)
     
# #     #Take S and remove any value that is less than half
# #     s = myimage_hsv[:,:,1]
# #     s = np.where(s < 25, 0, 1) # Any value below 25 will be excluded to detect and config
 
# #     # We increase the brightness of the image and then mod by 255
# #     v = (myimage_hsv[:,:,2] + 115) % 255  # 127
# #     v = np.where(v > 115, 1, 0)  # Any value above 127 will be part of our mask
 
# #     # Combine our two masks based on S and V into a single "Foreground"
# #     foreground = np.where(s+v > 0, 1, 0).astype(np.uint8)  #Casting back into 8bit integer

# #     background = np.where(foreground==0,255,0).astype(np.uint8) # Invert foreground to get background in uint8
# #     background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)  # Convert background back into BGR space
# #     foreground=cv2.bitwise_and(myimage,myimage,mask=foreground) # Apply our foreground map to original image
# #     finalimage = background+foreground # Combine foreground and background
# #     return finalimage

# # from rembg import remove
# # from PIL import Image
# # import os

# # for i in os.listdir(r'D:\DATN\Detect Object and Failure Object\Project Push Git\Image'):
# #     j = i.rsplit('.', maxsplit=1)[0]
# #     input_path = r'D:\DATN\Detect Object and Failure Object\Project Push Git\Image\\' + i
# #     output_path = r'D:\DATN\Detect Object and Failure Object\Project Push Git\Result Remove Background\\' + j + ".png"
# #     input = Image.open(input_path)
# #     output = remove(input)
# #     output.save(output_path)

# # import cv2
# # import numpy as np
# # import cvzone
# # from cvzone.SelfiSegmentationModule import SelfiSegmentation
# # import os

# # # Initialize the SelfiSegmentation module
# # segmentor = SelfiSegmentation()


# # # Set the directory containing images and the directory to save the processed images
# # input_image_dir = "image"
# # output_image_dir = "Result Remove Background"

# # # Create the output directory if it doesn't exist
# # if not os.path.exists(output_image_dir):
# #     os.makedirs(output_image_dir)

# # # List all image files in the directory
# # image_files = [os.path.join(input_image_dir, filename) for filename in os.listdir(input_image_dir) if filename.endswith(('.JPG', '.png', '.jpeg'))]

# # # Ensure there are images in the directory
# # if not image_files:
# #     print("No images found in the directory.")
# # else:
# #     # Process each image in the directory
# #     for img_path in image_files:
# #         # Read the image
# #         img = cv2.imread(img_path)
# #         # Perform background removal
# #         img_out = segmentor.removeBG(img,cutThreshold=0.85)  # Adjust threshold as needed
# #         # Get the filename (without extension) from the input image path
# #         filename = os.path.splitext(os.path.basename(img_path))[0]

# #         # Save the processed image to the output directory
# #         output_path = os.path.join(output_image_dir, f"{filename}_processed.jpg")
# #         cv2.imwrite(output_path, img_out)

        
       
# # # def sharpen_image_laplacian(image):
# # #     laplacian = cv2.Laplacian(image, cv2.CV_64F)
# # #     sharpened_image = np.uint8(np.clip(image - 0.3*laplacian, 0, 255))
# # #     return sharpened_image  # Return the sharpened image


# # while True:
# #     #img = img_out.copy()
# #     img = cv2.imread(r'Result Remove Background\sample No.1_processed.jpg')
# #     # sharpened_image = sharpen_image_laplacian(img)
# #     # img = sharpened_image.copy()
# #     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# #     thresh, output_otsuthresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# #     output_adapthresh = cv2.adaptiveThreshold (gray,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,51, 0)  #51
    
    
# #     # Display the original and processed images
# #     cv2.imshow("Original Image", img)
# #     cv2.imshow("Background Removed Image", img_out)
    
# #     cv2.imshow("Gray image",gray)
# #     cv2.imshow("Thresh image",output_adapthresh)
# #  # Wait for a key press to proceed to the next image or exit
# #     key = cv2.waitKey(0)

# #     # If 'q' is pressed, exit
# #     if key == ord('q'):
# #         break
# # # Close all windows
# # cv2.destroyAllWindows()

# #################################### Using detect defect #############################################
# # Import the library OpenCV 
# # import cv2 
# # while True:

# #     # Import the image 
# #     file_name = "Image/Sample No.1.JPG"

# #     # Read the image 
# #     src = cv2.imread(file_name, 1) 

# #     # Convert image to image gray 
# #     tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) 

# #     # Applying thresholding technique 
# #     _, alpha = cv2.threshold(tmp,125, 255, cv2.THRESH_BINARY) 

# #     # Using cv2.split() to split channels 
# #     # of coloured image 
# #     b, g, r = cv2.split(src) 

# #     # Making list of Red, Green, Blue 
# #     # Channels and alpha 
# #     rgba = [b, g, r, alpha] 

# #     # Using cv2.merge() to merge rgba 
# #     # into a coloured/multi-channeled image 
# #     dst = cv2.merge(rgba, 4) 

# #     # Writing and saving to a new image 
# #     cv2.imshow("gfg_white.png", dst) 
# #     cv2.imshow("Threshold",alpha)
# #     key = cv2.waitKey(0)
# #     # If 'q' is pressed, exit
# #     if key == ord('q'):
# #             break

# # # # Close all windows
# # cv2.destroyAllWindows()


# import cv2
# import numpy as np


# def sharpen_image_laplacian(image):
#     laplacian = cv2.Laplacian(image, cv2.CV_64F)
#     sharpened_image = np.uint8(np.clip(image - 0.3*laplacian, 0, 255))
#     return sharpened_image  # Return the sharpened image

# def bgremove3(myimage):
#     # BG Remover 3
#     myimage_hsv = cv2.cvtColor(myimage, cv2.COLOR_BGR2HSV)
     
#     #Take S and remove any value that is less than half
#     s = myimage_hsv[:,:,1]
#     s = np.where(s < 25, 0, 1) # Any value below 25 will be excluded to detect and config
 
#     # We increase the brightness of the image and then mod by 255
#     v = (myimage_hsv[:,:,2] + 115) % 255  # 127
#     v = np.where(v > 115, 1, 0)  # Any value above 127 will be part of our mask
 
#     # Combine our two masks based on S and V into a single "Foreground"
#     foreground = np.where(s+v > 0, 1, 0).astype(np.uint8)  #Casting back into 8bit integer

#     background = np.where(foreground==0,255,0).astype(np.uint8) # Invert foreground to get background in uint8
#     background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)  # Convert background back into BGR space
#     foreground=cv2.bitwise_and(myimage,myimage,mask=foreground) # Apply our foreground map to original image
#     image_rmbg = background+foreground # Combine foreground and background
#     return image_rmbg


# while True:
#     img = cv2.imread('Image/sample No.14.JPG')
#     img = cv2.resize(img,(400,300))
#     sharpened_image = sharpen_image_laplacian(img)
#     image_rmbg = bgremove3(sharpened_image)
    
#     gray = cv2.cvtColor(image_rmbg, cv2.COLOR_BGR2GRAY)
#     thresh, output_otsuthresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#     output_adapthresh = cv2.adaptiveThreshold (gray,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,51, 0)

#     cv2.imshow("Remove background",image_rmbg)
#     cv2.imshow("Thresh image",output_adapthresh)

#     key = cv2.waitKey(1)
#     if key == ord("q"):
#         break
# cv2.destroyAllWindows()

import cv2
import numpy as np
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

# Initialize the SelfiSegmentation module
segmentor = SelfiSegmentation()


# Set the directory containing images and the directory to save the processed images
input_image_dir = "Image1"
output_image_dir = "Result_Remove_Background1"

# Create the output directory if it doesn't exist
if not os.path.exists(output_image_dir):
    os.makedirs(output_image_dir)

# List all image files in the directory
image_files = [os.path.join(input_image_dir, filename) for filename in os.listdir(input_image_dir) if filename.endswith(('.JPG', '.png', '.jpeg','.jpg'))]
def bgremove3(myimage):
    # BG Remover 3
    myimage_hsv = cv2.cvtColor(myimage, cv2.COLOR_BGR2HSV)
     
    #Take S and remove any value that is less than half
    s = myimage_hsv[:,:,1]
    s = np.where(s <25, 0, 1) # Any value below 25 will be excluded to detect and config
 
    # We increase the brightness of the image and then mod by 255
    v = (myimage_hsv[:,:,2] + 127) % 255  # 127
    v = np.where(v > 127, 1, 0)  # Any value above 127 will be part of our mask
 
    # Combine our two masks based on S and V into a single "Foreground"
    foreground = np.where(s+v > 0, 1, 0).astype(np.uint8)  #Casting back into 8bit integer

    background = np.where(foreground==0,255,0).astype(np.uint8) # Invert foreground to get background in uint8
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)  # Convert background back into BGR space
    foreground=cv2.bitwise_and(myimage,myimage,mask=foreground) # Apply our foreground map to original image
    finalimage = background+foreground # Combine foreground and background
    return finalimage

# Ensure there are images in the directory
if not image_files:
    print("No images found in the directory.")
else:
    # Process each image in the directory
    for img_path in image_files:
        # Read the image
        img = cv2.imread(img_path)
        finalimage = bgremove3(img)
        # Perform background removal
        img_out = segmentor.removeBG(finalimage,cutThreshold=0.75)  # Adjust threshold as needed
        img_out = segmentor.removeBG(img_out,cutThreshold=0.94)  # Adjust threshold as needed

        # img_out = segmentor.removeBG(img_out,cutThreshold=0.65)
        # Get the filename (without extension) from the input image path
        filename = os.path.splitext(os.path.basename(img_path))[0]

        # Save the processed image to the output directory
        output_path = os.path.join(output_image_dir, f"{filename}_processed.jpg")
        cv2.imwrite(output_path, img_out)

        
       
# def sharpen_image_laplacian(image):
#     laplacian = cv2.Laplacian(image, cv2.CV_64F)
#     sharpened_image = np.uint8(np.clip(image - 0.3*laplacian, 0, 255))
#     return sharpened_image  # Return the sharpened image

def getcoutours(img, imgContour):
    contours, hierachy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print("Area of Contour",area)
        selected_contour = max(contours, key=lambda x: cv2.contourArea(x))
        areaMin = 1000 # Config area

        if area > areaMin:  
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
            # M = cv2.moments(cnt)
            # cx= int(M["m10"]/M["m00"])
            # cy= int(M["m01"]/M["m00"])  
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            ellipse = cv2.fitEllipse(selected_contour)
            cv2.ellipse(img, ellipse, (0, 255, 0), 3)
            # cv2.circle(img,(cx,cy),7,(0,0,255),-1)
            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 5) 
while True:
    #img = img_out.copy()
    # img = cv2.imread(r'')
    # sharpened_image = sharpen_image_laplacian(img)
    # img = sharpened_image.copy()
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # thresh, output_otsuthresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # # cv2.imshow("output_otsuthresh", output_otsuthresh)
    # # output_adapthresh = cv2.adaptiveThreshold (gray,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,51, 0)  #51
    # # Erosion image to detect Object Elipse for Durian 
    # kernel = np.ones((3,3),np.uint8)
    # # output_erosion = cv2.erode(output_adapthresh, kernel,iterations=1)
    # output_erosion = cv2.erode(output_otsuthresh, kernel,iterations=1)
    # # output_dilate = cv2.dilate(output_erosion, kernel,iterations=1)
    # output_dilate = cv2.dilate(output_otsuthresh, kernel,iterations=1)

    # sub = 255 - output_dilate
    # border = output_dilate - output_erosion

 # Wait for a key press to proceed to the next image or exit
    key = cv2.waitKey(0)

    # If 'q' is pressed, exit
    if key == ord('q'):
        break
# Close all windows
cv2.destroyAllWindows()
