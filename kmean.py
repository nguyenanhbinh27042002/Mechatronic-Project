import numpy as np
import cv2 as cv

img = cv.imread('images_mui.jpg')
img_resize = cv.resize(img,(500,500))
Z = img_resize.reshape((-1,3))
 
# convert to np.float32
Z = np.float32(Z)
 
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1.0)
K =  35 # dieu chinh do phan giai
ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
 
# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img_resize.shape)) 



cv.imshow('res2',res2)

cv.waitKey(0)
cv.destroyAllWindows()