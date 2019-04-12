import face_recognition
import time
import numpy as np
import cv2
import os
from imutils import face_utils,rotate

def SaveFeature(src,dst):
    list_address_images=os.listdir(src)
    for address_image in list_address_images:
        add_img = src + "\\" +address_image
        add_feature=dst + "\\" +address_image.split(".")[0]+".npy" 
        print(add_feature)
        im=cv2.imread(add_img)
        img=im[:, :, ::-1]
        img_face_locations = face_recognition.face_locations(img)
        img_face_encodings = face_recognition.face_encodings(img, img_face_locations)[0]
        np.save(add_feature, img_face_encodings)
# SaveFeature(r"D:\PROJECT\Two_Face_Comparer\Images",r"D:\PROJECT\Two_Face_Comparer\Features")
# exit()
def LoadFeature(src):
    name_feature=[]
    list_address_feature=os.listdir(src)
    for address_feature in list_address_feature:
       name = address_feature.split(".")[0]
       features = np.load(src+"\\"+address_feature)
       name_feature.append((name,features))
    return name_feature

def Recognition_image_face():
    im=cv2.imread("D:\\PROJECT\\Two_Face_Comparer\\v6.png")
    img=im[:, :, ::-1]
    img_face_locations = face_recognition.face_locations(img)
    img_face_encodings = face_recognition.face_encodings(img, img_face_locations)
    DB=LoadFeature("D:\PROJECT\Two_Face_Comparer\Features")
    for i in range(len(img_face_encodings)):
        minDistance=10
        nameID=""
        for face in DB:
            distance=np.linalg.norm(img_face_encodings[i] - face[1])
            if distance < minDistance:
                minDistance=distance
                nameID=face[0]
        cv2.rectangle(im,(img_face_locations[i][1],img_face_locations[i][0]),(img_face_locations[i][3],img_face_locations[i][2]),(0,255,0),3)
        cv2.putText(im, "Face : " + nameID, (img_face_locations[i][3],img_face_locations[i][0]-30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        cv2.putText(im, "Dist : " + str(round(minDistance, 4)), (img_face_locations[i][3],img_face_locations[i][0]-10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        print(nameID, minDistance)


    cv2.imshow("img ",im)
    cv2.waitKey()
# Recognition_image_face()

def Recognition_video_face(src):
    out = cv2.VideoWriter('D:\\PROJECT\\Two_Face_Comparer\\testGVC.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (1920,1080))
    video_capture = cv2.VideoCapture(src)
    k=1
    while True:
        ret, frame = video_capture.read()
        k+=1
        if k%7 !=0:
            continue
        # print("frame : ",k)
        if((cv2.waitKey(1) & 0xFF == ord('q')) or ret ==False):
            break
        # frame =rotate(frame, -90)
        frame = cv2.resize(frame, (0,0), fx=0.6, fy=0.6) 
        img=frame[:, :, ::-1]
        img_face_locations = face_recognition.face_locations(img)
        img_face_encodings = face_recognition.face_encodings(img, img_face_locations)
        DB=LoadFeature("D:\PROJECT\Two_Face_Comparer\Features")
        for i in range(len(img_face_encodings)):
            minDistance=10
            nameID=""
            for face in DB:
                distance=np.linalg.norm(img_face_encodings[i] - face[1])
                if distance < minDistance:
                    minDistance=distance
                    nameID=face[0]
            if minDistance < 0.35:
                cv2.rectangle(frame,(img_face_locations[i][1],img_face_locations[i][0]),(img_face_locations[i][3],img_face_locations[i][2]),(0,255,0),3)
                cv2.putText(frame, "Face : " + nameID, (img_face_locations[i][3],img_face_locations[i][0]-30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                cv2.putText(frame, "Dist : " + str(round(minDistance, 4)), (img_face_locations[i][3],img_face_locations[i][0]-10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame,(img_face_locations[i][1],img_face_locations[i][0]),(img_face_locations[i][3],img_face_locations[i][2]),(0,0,255),3)
                cv2.putText(frame, "Face not in Database " , (img_face_locations[i][3],img_face_locations[i][0]-30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0, 255), 2)
            # print(nameID, minDistance)
        # out.write(frame)
        cv2.putText(frame, "FACE RECOGNITION SYSTEM IN GVC COMPANY " , (10,50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
        cv2.imshow("frame ",frame)


    video_capture.release()
    cv2.destroyAllWindows()
Recognition_video_face("rtsp://admin:admin1234@192.168.1.208:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif")