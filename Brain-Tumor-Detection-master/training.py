import os
import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt
import tensorflow as tf

#Chế độ kiểm tra được đặt thành false để tránh kết quả kiểm tra không mong muốn từ các ô
test = False

YES_DIR_PATH = 'brain_tumor_dataset/yes/'
NO_DIR_PATH = 'brain_tumor_dataset/no/'
# YES_DIR_PATH = 'brain_tumor_dataset/yes/'
# NO_DIR_PATH = 'brain_tumor_dataset/no/'
yes_imgs_name = os.listdir(YES_DIR_PATH)
no_imgs_name = os.listdir(NO_DIR_PATH)

#crop phần não ra khỏi hình ảnh
def crop_image(img):
   # Thay đổi kích thước hình ảnh thành 256x256 pixel
    resized_img = cv2.resize(
        img,
        dsize=(256, 256),
        interpolation=cv2.INTER_CUBIC
    )
    # Chuyển đổi hình ảnh thành thang độ xám
    gray = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)

    # Áp dụng hiệu ứng làm mờ Gaussian cho hình ảnh
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Ngưỡng hình ảnh bằng Ngưỡng nhị phân
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    # thực hiện một loạt xói mòn & giãn nở để loại bỏ bất kỳ vùng nhiễu nhỏ nào
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # tìm đường viền trong hình ảnh ngưỡng, sau đó lấy hình ảnh lớn nhất
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # tìm điểm cực trị
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # crop
    ADD_PIXELS = 0
    cropped_img = resized_img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS,
                              extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
    return cropped_img

# Sử dụng chức năng crop_image để cắt ảnh
yes_imgs_cropped = [crop_image(cv2.imread(YES_DIR_PATH + img_file)) for img_file in yes_imgs_name]
no_imgs_cropped = [crop_image(cv2.imread(NO_DIR_PATH + img_file)) for img_file in no_imgs_name]

#1.2 thay đổi kích thước hình ảnh
orig_imgs = yes_imgs_cropped + no_imgs_cropped
resized_imgs = [cv2.resize(img, dsize=(32, 32), interpolation=cv2.INTER_CUBIC) for img in orig_imgs]
X = np.squeeze(resized_imgs)
if (test):
    print(type(X))
    print(X.shape)
    print(X)
    print(resized_imgs)
# chuẩn hóa dữ liệu
X = X.astype('float32')
X /= 255

if (test):
    print(X)

labels_yes = np.full(len(yes_imgs_name), 1)
labels_no = np.full(len(no_imgs_name), 0)

img_labels = np.concatenate([labels_yes, labels_no])
if (test):
    print(img_labels.size, img_labels)
    
# Tách tập dữ liệu thành Tập huấn luyện (tức là `x_train`) và Tập kiểm tra/Tập xác thực (tức là `x_valid`)
# Chúng em cũng sẽ giữ các hình ảnh gốc của bộ xác thực trong `x_orig_valid` cho mục đích trực quan hóa
yes_imgs = X[:155]
no_imgs = X[155:]
yes_orig_imgs = orig_imgs[:155]
no_orig_imgs = orig_imgs[155:]

x_yes_train = yes_imgs[:124]
x_yes_valid = yes_imgs[124:]
x_yes_orig_valid = yes_orig_imgs[124:]

x_no_train = no_imgs[:78]
x_no_valid = no_imgs[78:]
x_no_orig_valid = no_orig_imgs[78:]

x_train = np.concatenate([x_yes_train, x_no_train])
x_valid = np.concatenate([x_yes_valid, x_no_valid])
x_orig_valid = np.concatenate([x_yes_orig_valid, x_no_orig_valid])

# Tách nhãn tập dữ liệu cho Tập huấn luyện (tức là `y_train`) và Tập kiểm tra/Tập xác thực (tức là `y_valid`)
yes_labels = img_labels[:155]
no_labels = img_labels[155:]

y_yes_train = yes_labels[:124]
y_yes_valid = yes_labels[124:]

y_no_train = no_labels[:78]
y_no_valid = no_labels[78:]

y_train = np.concatenate([y_yes_train, y_no_train])
y_valid = np.concatenate([y_yes_valid, y_no_valid])

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=9,
          padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.45))

model.add(tf.keras.layers.Conv2D(
    filters=16, kernel_size=9, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Conv2D(
    filters=36, kernel_size=9, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dropout(0.15))


model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# tóm tắt mô hình
# model.summary() # Bỏ ghi chú để xem tóm tắt mô hình. Tóm tắt mô hình đã được hiển thị ô bên dưới trong biểu diễn SVG

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['acc'])
model.fit(x_train,
          y_train,
          batch_size=128,
          epochs=200,
          validation_data=(x_valid, y_valid),)

# Lưu mô hình được đào tạo
model.save('brain_tumor_detection_model.h5')

