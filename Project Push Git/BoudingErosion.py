import cv2
import numpy as np
from scipy.spatial import distance

while True:

    image = cv2.imread("testhu.jpg")
    image = cv2.resize(image,(400,300))
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # thresh, output_binthresh = cv2.threshold(image_gray, 45, 255, cv2.THRESH_BINARY)    
    # cv2.imshow("Binary Threshold (fixed)", output_binthresh)

    # thresh, output_otsuthresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv2.imshow("Binary Threshold (otsu)", output_otsuthresh)

    output_adapthresh = cv2.adaptiveThreshold (image_gray,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,51, 0)  #51
    cv2.imshow("Adaptive Thresholding", output_adapthresh)

    kernel = np.ones((3,3),np.uint8)
    # output_erosion = cv2.erode(output_adapthresh, kernel,iterations=2)
    output_erosion = cv2.morphologyEx(output_adapthresh, cv2.MORPH_OPEN,kernel)
    output_erosion = cv2.erode(output_adapthresh, kernel,iterations=2)
    output_erosion = cv2.dilate(output_adapthresh, kernel,iterations=3)
    cv2.imshow("Morphological Erosion", output_erosion)

    contours, _ = cv2.findContours(output_erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_contour = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(output_contour, contours, -1, (0, 0, 255), 2)
    cv2.imshow("Contours", output_contour)

    contours_sorted = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    image_result = image.copy()

    pixelsPerMm = 5.509

    rices = []

    for c in contours_sorted:

        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
       
        (top_left, top_right, bottom_right, bottom_left) = box

        distance_a = distance.euclidean((top_left[0], top_left[1]), (top_right[0], top_right[1]))
        distance_b = distance.euclidean((top_left[0], top_left[1]), (bottom_left[0], bottom_left[1]))

        dimension_a = distance_a / pixelsPerMm
        dimension_b = distance_b / pixelsPerMm

        length = max(dimension_a, dimension_b)        
        width = min(dimension_a, dimension_b)        

        # if length < 1.4 or length > 12.3:
        #     continue
        # if width < 1.4 or width > 3.5:
        #     continue
        if length > 12.3:
            continue
        if width > 4.0:
             continue

        rice = [box.astype("int")]

        # Check class
        # if length <= 1.6:
        #      rices.append({'class': 'class1', 'box': rice, 'color': (255, 0, 255), 'width': width, 'length': length, 'top_right': top_right})
        # elif length <= 1.8:
        #      rices.append({'class': 'class2', 'box': rice, 'color': (255, 0, 0), 'width': width, 'length': length, 'top_right': top_right})
        # elif length <= 3.8:
        #      rices.append({'class': 'class3', 'box': rice, 'color': (0, 255, 0), 'width': width, 'length': length, 'top_right': top_right})
        # elif length <= 5.6:
        #     rices.append({'class': 'class4', 'box': rice, 'color': (51, 255, 255), 'width': width, 'length': length, 'top_right': top_right})
        if length <= 12:
            rices.append({'class': 'class5', 'box': rice, 'color': (0, 0, 255), 'width': width, 'length': length, 'top_right': top_right})
        else:
            continue       
        
        
    for rice in rices:
        cv2.drawContours(image_result, rice['box'], -1, rice['color'], 1)
        # cv2.putText(image_result, "l={:.1f}mm".format(rice['length']), (rice['top_right'][0]-10, rice['top_right'][1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        # cv2.putText(image_result, "w={:.1f}mm".format(rice['width']), (rice['top_right'][0]-10, rice['top_right'][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)


    cv2.imshow("Result", image_result)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
cv2.destroyAllWindows()