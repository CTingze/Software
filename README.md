<h1> 系統介紹</h1>

環境建置
<br>
使用OpenCV 讀取YOLOv3模型
``` python
import cv2
import numpy as np
net = cv2.dnn.readNetFromDarknet("yolov3.cfg","yolov3_1100.weights")
```
<br>
讀取相關參數
`‵`
classes = [line.strip() for line in open("cfg_mask/obj.names")]
colors = [(0,0,255),(255,0,0),(0,255,0)]
`‵`
讀取圖片
`‵`
img = cv2.imread("/Users/davidchiu/Desktop/test.jpg")
img.shape
`‵`

利用YOLOv3 模型辨識圖片
320 x 320 (high speed, less accuracy)
416 x 416 (moderate speed, moderate accuracy)
608 x 608 (less speed, high accuracy)
`‵`
img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape 
blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)
`‵`
擷取偵測物件位置
`‵`
%%html
<img src='https://miro.medium.com/max/1200/0*3A8U0Hm5IKmRa6hu.png' width="500px" />
`‵`
框住偵測物件區域
`‵`
font = cv2.FONT_HERSHEY_PLAIN

for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[class_ids[i]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 5), font, 2, color, 3)

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


