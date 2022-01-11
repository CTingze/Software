<h1> 系統介紹</h1>

環境建置
<br>
``` python
import cv2
import numpy as np
net = cv2.dnn.readNetFromDarknet("yolov3.cfg","yolov3_1100.weights")
`‵`
<br>
使用OpenCV 讀取YOLOv3模型
`‵`
classes = [line.strip() for line in open("cfg_mask/obj.names")]
colors = [(0,0,255),(255,0,0),(0,255,0)]
`‵`

`‵`
img = cv2.imread("/Users/davidchiu/Desktop/test.jpg")
img.shape
`‵`

`‵`
img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape 
blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)
`‵`

`‵`

`‵`

`‵`

`‵`

`‵`

`‵`

`‵`

`‵`


`‵`

`‵`

`‵`

`‵`

`‵`

`‵`

`‵`

`‵`

`‵`

`‵`


