from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import cv2, os, urllib.request, pickle
import numpy as np
from datetime import datetime
import os.path
import pyrebase

protoPath = "face_detection_model/deploy.prototxt"
modelPath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
embedder = cv2.dnn.readNetFromTorch('openface_nn4.small2.v1.t7')
recognizer = "output/recognizer.pickle"
le = "output/le.pickle"
try:
    recognizer = pickle.loads(open(recognizer, "rb").read())
    le = pickle.loads(open(le, "rb").read())
except:
    pass

today = datetime.today()
attendance_csv = 'Attendance' + today.strftime(
    "%m_%d_%y") + '.csv'

config = {
    "apiKey": "AIzaSyBY3X-PB5Iziv3YE-S62jcthPgZbrDpHnc",
    "authDomain": "facemark-a0bef.firebaseapp.com",
    "databaseURL": "https://facemark-a0bef-default-rtdb.firebaseio.com/",
    "projectId": "facemark-a0bef",
    "storageBucket": "facemark-a0bef.appspot.com",
    "messagingSenderId": "1086140254646",
    "appId": "1:1086140254646:web:edfa35b1d2a6d5217d1917",
    "measurementId": "G-VD5E8YXKDJ"
}

firebase = pyrebase.initialize_app(config)
authe = firebase.auth()
database = firebase.database()


class FaceDetect(object):
    def __init__(self):
        self.vs = VideoStream(src=0).start()
        self.fps = FPS().start()

    def __del__(self):
        self.fps.stop()
        cv2.destroyAllWindows()

    def markAttendance(self, name):
        if os.path.isfile(attendance_csv):
            access = 'r+'
        else:
            access = 'w+'
        with open(attendance_csv, access) as f:
            myDataList = f.readlines()
            nameList = []
            for line in myDataList:
                entry = line.split(',')
                nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%m/%d/%y %H:%M:%S')
                f.writelines(f'\n{name},{dtString}')
                student_data = database.child('data').child('dataset').get().val()
                id = 0
                for i in range(len(student_data)):
                    if (student_data[i] != None and student_data[i].get("name") == name):
                        type = student_data[i].get("type")
                        id = student_data[i].get("id")
                data = database.child('data').child('attendance').get().val()
                newData = {"id": id, "name": name, "timestamp": dtString,
                           "type": type}
                database.child('data').child('attendance').child(len(data)).set(newData)

    def get_frame(self):
        frame = self.vs.read()
        frame = cv2.flip(frame, 1)
        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        detector.setInput(imageBlob)
        detections = detector.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.8:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]
                if fW < 20 or fH < 20:
                    continue
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                 (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()
                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                
                # text = "{}: {:.2f}%".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                if (proba * 100 > 70):
                    name = le.classes_[j]
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                                    (0, 0, 255), 2)
                    cv2.putText(frame, name, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    self.markAttendance(name)
                else:
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                                    (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown", (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        self.fps.update()
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
