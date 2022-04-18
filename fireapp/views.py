import urllib
from urllib.request import urlopen
from django.shortcuts import render
import pyrebase
from json import dumps
import certifi
from django.http.response import StreamingHttpResponse
from .camera import FaceDetect
import numpy as np
import imutils
import cv2
import os
from django.contrib import messages
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

STUDENT_NAME = "None"

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


def index(request):
    return render(request, 'index.html')


def recognition(request):
    return render(request, 'recognition.html')


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def facecam_feed(request):
    return StreamingHttpResponse(gen(FaceDetect()),
                                 content_type='multipart/x-mixed-replace; boundary=frame')


def attendance_list(request):
    data = database.child('data').child('attendance').get().val()
    js_data = dumps(data)
    context = {
        'data': js_data
    }
    if request.method == 'POST':
        report_filter = request.POST["report_month"]
    return render(request, 'attendance_list.html', context)


def loginUser(request):
    username = request.POST.get('username')
    password = request.POST.get('password')
    data = database.child('data').child('login').get().val()
    js_data = dumps(data)
    for x in data:
        if username != x.__getitem__('name') and password != x.__getitem__('password'):
            continue

        elif username == x.__getitem__('name') and password == x.__getitem__('password'):
            active_user = username
            context = {
                'data': js_data,
                'active_user': active_user
            }
            return render(request, 'index.html', context)

    return render(request, 'loginUser.html')


def logoutUser(request):
    return render(request, 'logout.html')


def onClickHome(request):
    return render(request, 'home.html')


def onClickEmbeddings(request):
    data = database.child('data').child('dataset').get().val()
    js_data = dumps(data)
    context = {
        'data': js_data,
        'user': STUDENT_NAME,
        'complete':'false'
    }

    return render(request, 'embeddings.html',context)


def callExtractEmbedding(request):
    STUDENT_NAME = request.POST["username"]
    data = database.child('data').child('dataset').get().val()
    js_data = dumps(data)
    url_list = ""
    for x in data:
        if (x is not None) and (x.__getitem__('name') == STUDENT_NAME):
            url_list = x.__getitem__('url')
            continue
 
    dataset = 'dataset'
    embeddings = 'output/embeddings.pickle'
    detector = 'face_detection_model'
    embedding_model = 'openface_nn4.small2.v1.t7'
    confidence_val = 0.9

    messages.success(request, "[INFO] loading face detector...")
    protoPath = os.path.sep.join([detector, "deploy.prototxt"])

    modelPath = os.path.sep.join([detector,
                                  "res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    messages.success(request, "[INFO] loading face recognizer...")
    embedder = cv2.dnn.readNetFromTorch(embedding_model)
    if os.path.getsize(embeddings) > 0:
        with open(embeddings, "rb") as file:
            unpickler = pickle.Unpickler(file)
            content = unpickler.load()
    try:
        knownEmbeddings = content.get("embeddings")
        knownNames = content.get("names")
    except:
        knownEmbeddings = []
        knownNames = []
    total = 0
    for item in range(len(url_list)):
        print("[INFO] processing image {}/{}".format(item + 1, len(url_list)))
        messages.success(request, "[INFO] processing image {}/{}".format(item + 1, len(url_list)))
        name = STUDENT_NAME
        requ = urllib.request.Request(url_list[item], headers = {'User-Agent': 'Mozilla/5.0'})
        req = urllib.request.urlopen(requ,cafile=certifi.where())
        print(url_list[item])
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        image = cv2.imdecode(arr, -1)  # 'Load it as it is'
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)
        detector.setInput(imageBlob)
        detections = detector.forward()
        if len(detections) > 0:
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_val:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]
                if fW < 20 or fH < 20:
                    continue
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                 (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()
                knownNames.append(name)
                knownEmbeddings.append(vec.flatten())
                total += 1

    messages.success(request, "[INFO] serializing {} encodings...".format(total))
    data = {"embeddings": knownEmbeddings, "names": knownNames}
    f = open(embeddings, "wb")
    f.write(pickle.dumps(data))
    messages.success(request, "Embeddings Complete")
    f.close()


def embedding(request):
    data = database.child('data').child('dataset').get().val()
    js_data = dumps(data)

    context = {
        'data': js_data,
        'user': STUDENT_NAME,
        'complete':'true'
    }
    if request.method == 'POST':
        if 'create_btn_embedding' in request.POST:
            callExtractEmbedding(request)
        elif 'create_btn_training' in request.POST:
            callTraining(request)
    return render(request, 'embeddings.html', context)


def callTraining(request):
    embeddings_val = 'output/embeddings.pickle'
    recognizer_val = 'output/recognizer.pickle'
    le_val = 'output/le.pickle'
    messages.success(request, "[INFO] loading face embeddings...")
    data = pickle.loads(open(embeddings_val, "rb").read())
    messages.success(request, "[INFO] encoding labels...")
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])
    messages.success(request, "[INFO] training model...")
    recognizer = SVC(kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)
    f = open(recognizer_val, "wb")
    f.write(pickle.dumps(recognizer))
    f.close()
    f = open(le_val, "wb")
    f.write(pickle.dumps(le))
    messages.success(request, "[INFO] training model complete...")
    f.close()
    STUDENT_NAME = None

def trainModel(request):
    context = {
        'student_name': STUDENT_NAME
    }
    if request.method == 'POST':
        callTraining(request)
    return render(request, 'train.html', context)


def camera(request):
    if request.method == 'POST':
        global STUDENT_NAME
        studentName = request.POST["student_name_val"]
        studentId = request.POST["student_id_val"]
        user_type = request.POST["user_type"]
        url = request.POST["url"]
        STUDENT_NAME = studentName
        def mysplit(s, delim=None):
            return [x for x in s.split(delim) if x]
        urls = mysplit(url, ",")
        myImages = list(dict.fromkeys(urls))
        data = database.child('data').child('dataset').get().val()
        newData = {"id": studentId,
                   "name": STUDENT_NAME,
                   "type": user_type,
                   "url": myImages
                   }
        if STUDENT_NAME is None:
            print("name empty, not saved to database.")
        else:
            database.child('data').child('dataset').child(len(data)).set(newData)
            return render(request, 'camera.html', {'student_name': STUDENT_NAME})

    return render(request, 'camera.html')
