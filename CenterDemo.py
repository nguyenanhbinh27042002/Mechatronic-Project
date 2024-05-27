import cv2
import numpy as np
import imutils
from imutils import paths
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import argparse
import os

def process_images():
    # Function to find markers in the image
    def find_markers(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 35, 125)
        cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        return cv2.minAreaRect(c)

    # Initialize distance from camera to the object, which in this case is 24cm
    Know_distance = 24.0
    # Initialize the known object width, which in this case, the piece of is 12cm
    known_width = 11.0

    # Define and distance calculation
    def distance_to_camera(knowWidth, focal_length, perWidth):
        # Compute and return the distance from the marker to the camera
        return (knowWidth * focal_length) / perWidth

    # Size measurement
    def midpoint(ptA, ptB):
        return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

    # # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", required=True, help="path to input image")
    # ap.add_argument("-w", "--width", type=float, required=True, help="width of the left-most object in the image (in millimeter)")
    args = vars(ap.parse_args())

    # Load the input image
    # image = cv2.imread(args["image"])
    image = cv2.imread("Durian_Test.jpg")
    # Find markers in the image
    markers = find_markers(image)
    focal_length = (markers[1][0] * Know_distance) / known_width

    # Image processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = None

    # Iterate through each image in the folder
    for imagePath in sorted(paths.list_images("images")):
        image = cv2.imread(imagePath)
        markers = find_markers(image)
        millimeter = distance_to_camera(known_width, focal_length, markers[1][0])

        # Draw a bounding box around the image
        box = cv2.boxPoints(markers) if imutils.is_cv2() else cv2.boxPoints(markers)
        box = np.int0(box)
        cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
        cv2.putText(image, "%.2fft" % (millimeter / 12),
                    (image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)

        cv2.imshow("image", image)
        cv2.waitKey(0)

    # Further processing and measurement
    for c in cnts:
        if cv2.contourArea(c) < 100:
            continue

        orig = image.copy()
        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

        # Midpoint calculation
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # Draw midpoints
        cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        # Draw lines between midpoints
        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

        # Compute the Euclidean distance between the midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        # If pixelsPerMetric has not been initialized, compute it as the ratio of pixels to the supplied metrics
        if pixelsPerMetric is None:
            pixelsPerMetric = dB / args["width"]

        # Compute the size of the object
        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric

        # Draw the object sizes on the image
        cv2.putText(orig, "{:.1f}in".format(dimA),
                    (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # Show the input image
        cv2.imshow("Image", orig)
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

# Add a button to execute the function in VSCode
if __name__ == "__main__":
    process_images()
