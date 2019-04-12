import cv2
import numpy as np


video_capture = cv2.VideoCapture("rtsp://admin:admin1234@192.168.1.208:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif")
# capture = cv2.VideoCapture('rtsp://username:password@192.168.1.64/1')
while True:
    ret, frame = video_capture.read()

    if((cv2.waitKey(1) & 0xFF == ord('q')) or ret ==False):
        break
    cv2.imshow("frame",frame)

video_capture.release()
cv2.destroyAllWindows()