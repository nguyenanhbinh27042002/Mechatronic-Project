import cv2
import numpy as np
def crop_image(image_path, target_height_cm, target_width_cm,resolution):
    # Read the image
    image = cv2.imread(image_path)
    # Print the pixel values
    print("Pixel values of the original image:")
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            pixel_value = image[row, col]
            
    print("Pixel at (row={}, col={}): {}".format(row, col, pixel_value))

    #Convert centimeters to pixels
    target_height_px = int(target_height_cm * resolution)
    target_width_px = int(target_width_cm * resolution)
    
    # Get the original image dimensions
    original_height, original_width = image.shape[:2]
    
    # Calculate the center coordinates
    center_x = original_height // 2
    center_y = original_height // 2
    # Calculate the coordinates for cropping
    start_y = center_y - (target_height_px // 2 )
    end_y = start_y + target_height_px

    start_x = center_x - (target_width_px // 2)
    end_x = start_x + target_width_px

    # Crop the image
    cropped_image = image[start_y:end_y, start_x:end_x]
    return cropped_image

# Path to the image you want to crop
image_path = 'D:\DATN\Detect Object and Failure Object\Test1(2).png'

# Specify the target height and width
target_height_cm = 35
target_width_cm = 25
resolution = 100

# Crop the image
cropped_image =  crop_image(image_path,target_height_cm,target_width_cm,resolution)

# Display the cropped image
cv2.imshow('Cropped Image', cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()