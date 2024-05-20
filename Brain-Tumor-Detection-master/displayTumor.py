import numpy as np
import cv2 as cv


class DisplayTumor:
    curImg = 0
    Img = 0

    def readImage(self, img):
        self.Img = np.array(img)
        self.curImg = np.array(img)
        #Định nghĩa một biến màu mới, Chuyển đởi từ ảnh thường sang mức xám
        gray = cv.cvtColor(np.array(img), cv.COLOR_BGR2GRAY)
        #Ngưỡng của Otsu(phương pháp nhị phân hóa của Otsu)
        self.ret, self.thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    def getImage(self):
        return self.curImg

    # noise removal(loại bỏ nhiễu)
    def removeNoise(self):
        #Tạo mảng có kích thước 3x3 với giá trị 1
        self.kernel = np.ones((3, 3), np.uint8)
        #loại bỏ nhiễu bên ngoài khối u cần xác định(dùng Open), iterations(lặp đi lặp lại)
        opening = cv.morphologyEx(self.thresh, cv.MORPH_OPEN, self.kernel, iterations=2)
        self.curImg = opening

    def displayTumor(self):
        #vùng sure background
        sure_bg = cv.dilate(self.curImg, self.kernel, iterations=3)

        # Tìm vùng sure foreground
        dist_transform = cv.distanceTransform(self.curImg, cv.DIST_L2, 5)
        ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # tìm vùng chưa xác định
        sure_fg = np.uint8(sure_fg)
        #vùng backround - vùng foreground còn lại là vùng không xác định
        unknown = cv.subtract(sure_bg, sure_fg)

        # ghi nhãn điểm đánh dấu
        ret, markers = cv.connectedComponents(sure_fg)

        # thêm 1 vào tất cả các nhãn để sure background không phải là 0 mà là 1
        markers = markers + 1

        # giờ đánh dấu vùng chưa xác định = 0
        markers[unknown == 255] = 0
        markers = cv.watershed(self.Img, markers)
        self.Img[markers == -1] = [255, 0, 0]

        tumorImage = cv.cvtColor(self.Img, cv.COLOR_HSV2BGR)
        self.curImg = tumorImage