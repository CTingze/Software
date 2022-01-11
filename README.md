<h1> 人臉有無口罩偵測系統</h1>

## 系統介紹
<p>
　　本專題主要為在嚴峻的疫情下偵測民眾是否有戴好口罩，以防止感染擴散。
    當民眾出入公共場所經過人臉口罩偵測系統時，監控螢幕上顯示民眾，並即時擷取臉部圖像，再將圖像利用模型分析是否有口罩，若無口罩偵測系統就會立即在監控螢幕上，用紅框框上口罩沒戴確實的民眾，並發出警示聲提醒民眾或工作人員，以確保每一位民眾都有把口罩戴好且戴正確。
</p>

* 撰寫語言：python
* 執行平台：Google Colaboratory
* 網路架構：Yolov3
* 資料來源：Kaggle Face mask detection

## 環境建置

## 使用OpenCV 讀取YOLOv3模型

```
import cv2
import numpy as np
net = cv2.dnn.readNetFromDarknet("yolov3.cfg","yolov3_1100.weights")
```

## 讀取相關參數

```
classes = [line.strip() for line in open("cfg_mask/obj.names")]
colors = [(0,0,255),(255,0,0),(0,255,0)]
```

## 讀取圖片
```
img = cv2.imread("/Users/davidchiu/Desktop/test.jpg")
img.shape
```
## 利用YOLOv3 模型辨識圖片

* 320 x 320 (high speed, less accuracy)
* 416 x 416 (moderate speed, moderate accuracy)
* 608 x 608 (less speed, high accuracy)

```
img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape 
blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)
```
## 擷取偵測物件位置
```
class_ids = []
confidences = []
boxes = []
    
for out in outs:
    for detection in out:
        tx, ty, tw, th, confidence = detection[0:5]
        scores = detection[5:]
        class_id = np.argmax(scores)  
        if confidence > 0.3:   
            center_x = int(tx * width)
            center_y = int(ty * height)
            w = int(tw * width)
            h = int(th * height)
            
            # 取得箱子方框座標
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
            
```


## 框住偵測物件區域
```
font = cv2.FONT_HERSHEY_PLAIN

for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[class_ids[i]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 5), font, 2, color, 3)

```
## 將辨識過程包裝成函數
```
def yolo_detect(frame):
    # forward propogation
    img = cv2.resize(frame, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape 
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # get detection boxes
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            tx, ty, tw, th, confidence = detection[0:5]
            scores = detection[5:]
            class_id = np.argmax(scores)  
            if confidence > 0.3:   
                center_x = int(tx * width)
                center_y = int(ty * height)
                w = int(tw * width)
                h = int(th * height)

                # 取得箱子方框座標
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    # draw boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y -5), font, 3, color, 3)
    return img
```

## 測試函數功能
```
img = cv2.imread("/Users/davidchiu/Desktop/test.jpg")
im = yolo_detect(img)
img_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
```

## 使用攝像頭即時偵測物件
```
import cv2
import imutils
import time

VIDEO_IN = cv2.VideoCapture(0)

while True:
    hasFrame, frame = VIDEO_IN.read()
    
    img = yolo_detect(frame)
    cv2.imshow("Frame", imutils.resize(img, width=850))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
VIDEO_IN.release()
cv2.destroyAllWindows()
```

## 參考文獻
<p>
主題:雜亂大全24-YOLOv3(v1-v4統整資料)簡介(上) 作者:liao86221@gmail.com
https://liaozihzrong.github.io/2020/09/18/allinone24/ 
<br>
主題:學校由我們「罩」—口罩辨識系統建置 作者:張菖芠、高振皓、蔡尚宏
https://www.shs.edu.tw/works/essay/2021/03/2021031510532408.pdf 
<br>
主題:深度學習-物件偵測:You Only Look Once (YOLO) 作者:Tommy Huang
https://chih-sheng-huang821.medium.com/%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92-%E7%89%A9%E4%BB%B6%E5%81%B5%E6%B8%AC-you-only-look-once-yolo-4fb9cf49453c
<br>
主題:深度學習-什麼是one stage，什麼是two stage 物件偵測 作者:Tommy Huang
https://chih-sheng-huang821.medium.com/%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92-%E4%BB%80%E9%BA%BC%E6%98%AFone-stage-%E4%BB%80%E9%BA%BC%E6%98%AFtwo-stage-%E7%89%A9%E4%BB%B6%E5%81%B5%E6%B8%AC-fc3ce505390f
<br>
主題:目標檢測|YOLO原理與實現 作者:小小将
https://www.itread01.com/content/1545440522.html
<br>
主題:計算機視覺 作者:維基百科
https://zh.wikipedia.org/zh-hant/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89
<br>
主題:基礎目標檢測演算法介紹（一）：CNN、RCNN、Fast RCNN和Faster RCNN 作者:PULKIT SHARMA
https://www.itread01.com/fchkfhk.html
<br>
主題:論文筆記：目標檢測演算法（R-CNN，Fast R-CNN，Faster R-CNN，YOLOv1-v3） 作者:不詳
https://www.itread01.com/cqcly.html
<br>
主題:Top 8 Algorithms For Object Detection 作者:AMBIKA CHOUDHURY
https://analyticsindiamag.com/top-8-algorithms-for-object-detection/
<br>
主題:Top 6 Object Detection Algorithms 作者:Aditya Singh
https://medium.com/augmented-startups/top-6-object-detection-algorithms-b8e5c41b952f
<br>
主題:Comparative analysis of deep learning image detection algorithms 作者:Shrey Srivastava, Amit Vishvas Divekar, Chandu Anilkumar, Ishika Naik, Ved Kulkarni & V. Pattabiraman
https://journalofbigdata.springeropen.com/articles/10.1186/s40537-021-00434-w
<br>
主題:You Only Look Once: Unified, Real-Time Object Detection 作者:Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi
https://arxiv.org/abs/1506.02640
<br>
主題:EfficientDet高效率的物件偵測模型 作者:CH.Tseng
https://chtseng.wordpress.com/2020/05/23/efficientdet%E9%AB%98%E6%95%88%E7%8E%87%E7%9A%84%E7%89%A9%E4%BB%B6%E5%81%B5%E6%B8%AC%E6%A8%A1%E5%9E%8B/
<br>
主題:目标检测：Faster-RCNN与YOLO V3模型的对比分析 作者:普通攻击往后拉
https://blog.csdn.net/weixin_43483381/article/details/107944903
<br>
主題:Architecture of YOLOv3 作者:Akshay Atam
https://iq.opengenus.org/architecture-of-yolov3/
<br>
主題:FPN（feature pyramid networks）算法讲解 作者:AI之路
https://blog.csdn.net/u014380165/article/details/72890275
<br>
</p>
