import cv2
import numpy as np

# def find_angle(x1, y1, x2, y2, x3, y3):
#     ria = np.arctan2(y2 - y1, x2 - x1) - np.arctan2(y3 - y1, x3 - x1)
#     if ria > 0:
#         if ria < 3:
#             web_angle = int(np.abs(ria * 180 / np.pi))
#         elif ria > 3:
#             web_angle = int(np.abs(ria * 90 / np.pi))
#     elif ria < 0:
#         if ria < -3:
#             web_angle = int(np.abs(ria * 90 / np.pi))
#         elif ria > -3:
#             web_angle = int(np.abs(ria * 180 / np.pi))
#     return web_angle

def detect_objects(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    output_adapthresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 0)

    # Erosion image to detect object ellipse
    kernel = np.ones((3, 3), np.uint8)
    output_erosion = cv2.morphologyEx(output_adapthresh, cv2.MORPH_OPEN, kernel)
    output_erosion = cv2.erode(output_adapthresh, kernel, iterations=2)
    output_erosion = cv2.dilate(output_adapthresh, kernel, iterations=4)

    contours, hierarchy = cv2.findContours(output_erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    selected_contour = max(contours, key=lambda x: cv2.contourArea(x))

    # Fit ellipse
    ellipse = cv2.fitEllipse(selected_contour)
    cv2.ellipse(image, ellipse, (0, 255, 0), 3)

    # Find and draw angle
    (x, y), (MA, ma), angle = ellipse
    angle = int(angle)
    # cv2.putText(image, "Angle: " + str(angle), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Detected Objects", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "Image/sample No.17.JPG"
    detect_objects(image_path)
