from collections import deque
import numpy as np
import argparse
import imutils
import cv2



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")

# Set default=0 to disable tail (deque)
ap.add_argument("-b", "--buffer", type=int, default=10,
                help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries for the "green" "red" and "blue"
# Change the Values as per Requirement
# objects in the HSV model

redLower = (0, 100, 100)
redUpper = (10, 255, 255)

redLower2 = (160, 100, 100)
redUpper2 = (179, 255, 255)

# Kd's
greenLower = (29, 100, 100)
greenUpper = (50, 255, 255)


blueLower = (100, 100, 100)
blueUpper = (140, 255, 255)

# greenLower = (29, 86, 6)
# greenUpper = (64, 255, 255)

# Load lower and upper boundaries into colorBoundaries array we later iterate it to find Objects of different colors
colorBoundaries = [[redLower, redUpper], [redLower2, redUpper2], [greenLower, greenUpper], [blueLower, blueUpper]]
print(colorBoundaries)

ptsB = deque(maxlen=args["buffer"])
ptsG = deque(maxlen=args["buffer"])
ptsR = deque(maxlen=args["buffer"])
# If video path is not in arguments then start webcam

if not args.get("video", False):
    camera = cv2.VideoCapture(0)

else:
    camera = cv2.VideoCapture(args["video"])

# Main loop to captures frames and Process it. Will end if "q" is pressed
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()

    # if we passed a video and we did not grab a frame,
    # It's end of video so we break (quit)
    if args.get("video") and not grabbed:
        break
    # Resizing frame, Apply Gaussian Blur to remove noise, Convert it to the HSV

    frame = imutils.resize(frame, width=600)
    #blurred = cv2.GaussianBlur(frame, (11, 11), 3)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Set Centers of red blue and green objects captures to None
    centerR = None
    centerG = None
    centerB = None

    # For a Single Frame check for Red ,green or blue color objects and also maintain its tail (deque)
    for lowers, uppers in colorBoundaries:

        mask = cv2.inRange(hsv, lowers, uppers)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # Find Countours
        countours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        # Go in if atleast one Countour was found
        if len(countours) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(countours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)

            # only proceed if the radius meets a minimum size
            if radius > 10:
                M = cv2.moments(c)

                if lowers[0] == 0 or lowers[0] == 160:
                    r = 255
                    g = 0
                    b = 0
                    centerR = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                if lowers[0] == 100:
                    r = 000
                    g = 0
                    b = 255
                    centerB = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                if lowers[0] == 29:
                    r = 000
                    g = 255
                    b = 0
                    centerG = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                # Draw Circle
                cv2.circle(frame, (int(x), int(y)), int(radius),
                           (b, g, r), 2)

                # Update points in deque
                # Also Plot Centroid
                if lowers[0] == 0 or lowers[0] == 160:
                    cv2.circle(frame, centerR, 5, (0, 0, 255), -1)
                    ptsR.appendleft(centerR)
                    # loop and plot all points
                    for i in range(1, len(ptsR)):
                        # if either of the tracked points are None, ignore
                        # them
                        if ptsR[i - 1] is None or ptsR[i] is None:
                            continue
                       
                        cv2.line(frame, ptsR[i - 1], ptsR[i], (0, 0, 255), 3)

                if lowers[0] == 100:
                    # Print("making blue line")
                    cv2.circle(frame, centerB, 5, (0, 0, 255), -1)
                    ptsB.appendleft(centerB)
                    # loop and plot all points
                    for i in range(1, len(ptsB)):
                        # if either of the tracked points are None, ignore
                        # them
                        if ptsB[i - 1] is None or ptsB[i] is None:
                            continue
              
                        cv2.line(frame, ptsB[i - 1], ptsB[i], (255, 0, 0), 3)

                if lowers[0] == 29:
                    cv2.circle(frame, centerG, 5, (0, 0, 255), -1)
                    ptsG.appendleft(centerG)
                    # loop and plot all points
                    for i in range(1, len(ptsG)):
                        # if either of the tracked points are None, ignore
                        # them
                        if ptsG[i - 1] is None or ptsG[i] is None:
                            continue
                        #draw tail
                        cv2.line(frame, ptsG[i - 1], ptsG[i], (0, 255, 0), 3)

    # Show the frame after processing
    cv2.imshow("OUTPUT FRAME", frame)
    key = cv2.waitKey(1) & 0xFF

    # If 'q' key is pressed, Exit Loop
    if key == ord("q"):
        break


# Release camera and close windows
camera.release()
cv2.destroyAllWindows()
