from keras.models import load_model
import cv2 as cv
import imutils

model = load_model('brain_tumor_detection_model.h5')


def predictTumor(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)

    # Ngưỡng hình ảnh, sau đó thực hiện một loạt các bước xói mòn+
    # giãn nở để loại bỏ bất kỳ vùng nhiễu nhỏ nào
    thresh = cv.threshold(gray, 45, 255, cv.THRESH_BINARY)[1]
    thresh = cv.erode(thresh, None, iterations=2)
    thresh = cv.dilate(thresh, None, iterations=2)

    # Tìm đường viền trong ngưỡng ảnh sua đó lấy hình ảnh lớn nhất
    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv.contourArea)

    #Tìm điểm cực trị (extreme point)
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # crop ảnh mới ra khỏi anh gốc bằng 4 extreme point(4 điểm cực trị) (left, right, top, bottom)
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

    image = cv.resize(new_image, dsize=(32, 32), interpolation=cv.INTER_CUBIC)
    image = image / 255.

    image = image.reshape((1, 32, 32, 3))

    res = model.predict(image)

    return res