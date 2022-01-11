# Software
test

    
    ![image](https://github.com/CTingze/Software/blob/main/OIP.jfif)

![image](https://github.com/CTingze/Software/blob/main/tenor.gif)

``` python
import cv2
import numpy as np
net = cv2.dnn.readNetFromDarknet("yolov3.cfg","yolov3_1100.weights")

classes = [line.strip() for line in open("cfg_mask/obj.names")]
colors = [(0,0,255),(255,0,0),(0,255,0)]

## 讀取圖片

from PIL import Image
Image.open('/Users/davidchiu/Desktop/test.jpg')
	
img = cv2.imread("/Users/davidchiu/Desktop/test.jpg")
img.shape
`‵`
