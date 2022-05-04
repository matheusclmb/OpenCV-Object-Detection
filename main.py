import cv2
from matplotlib import pyplot as plt

config_file = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
frozen_model = "frozen_inference_graph.pb"
model = cv2.dnn_DetectionModel(frozen_model,config_file)

classLabels = []
file_name = "coco.names"
with open(file_name,"rt") as fpt:
    classLabels = fpt.read().rstrip("\n").split("\n")

model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()

    font_scale = 2
    font = cv2.FONT_HERSHEY_PLAIN

    classIndex, confidence, bbox = model.detect(frame,confThreshold=0.55)

    if (len(classIndex)!=0):
        for classInd,conf,boxes in zip(classIndex.flatten(), confidence.flatten(), bbox):
            if (classInd<=80):
                cv2.rectangle(frame,boxes,(255,0,0), 2)
                cv2.putText(frame, classLabels[classInd-1],(boxes[0]+10,boxes[1]+40), font, fontScale=font_scale, color=(0,255,0),thickness=3)

        cv2.imshow("Object Detection", frame)

        if cv2.waitKey(2) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()