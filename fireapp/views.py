from sys import path
from urllib.request import urlopen

from django.shortcuts import render

import pyrebase
from json import dumps
from django.core.files import File
from django.core.files.base import ContentFile
from django.core.files.temp import NamedTemporaryFile
from urllib.request import urlopen
from PIL import Image

import os

# Import these methods
from django.core.files import File
from django.core.files.base import ContentFile
from PIL import Image
import urllib3
from django.core.files.temp import NamedTemporaryFile

from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os
from django.contrib import messages
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

# Remember the code we copied from Firebase.
# This can be copied by clicking on the settings icon > project settings, then scroll down in your firebase dashboard
# from facemark.fireapp.models import user_details

STUDENT_NAME = "none"

# class views:
#     def __init__():
#         global student_name
        


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

# here we are doing firebase authentication
firebase = pyrebase.initialize_app(config)
authe = firebase.auth()
database = firebase.database()


def camera(request):
    # form = StudentNameForm()
    # context = {
    #     'form': form,
    #     'student_name':student_name
    # }
    if request.method == 'POST':
        # User_name = request.POST["Username"]
        # User_phone = request.POST["Userphone"]
        # User_address = request.POST["Useraddress"]
        # pic = request.FILES["photo"]
        # User_info= UserDetails(User_name=User_name, User_phone=User_phone, User_address=User_address, User_pic= pic)
        # User_info.save()    
        # path = request.POST["src"]
        # image = NamedTemporaryFile()
        # image.write(urlopen(path).read())
        # image.flush()
        # image = File(image)
        # name = str(image.name).split('\\')[-1]
        # name += '.jpg'
        # image.name = name
        # obj = Image.objects.create(image=image)
        # obj.save()
        global STUDENT_NAME
        studentName = request.POST["student_name_val"]
        STUDENT_NAME = studentName
        print(STUDENT_NAME)

    return render(request, 'camera.html',{'student_name':STUDENT_NAME })


def add_record(request):
    if request.method == 'POST':
        data = database.child('data').child('attendance').get().val()
        newData = {"id": 1, "name": request.POST.get('your_name'), "timestamp": request.POST.get('timestamp'),
                   "type": request.POST.get('type')}
        print(request.POST.get('your_name'))
        database.child('data').child('attendance').child(len(data)).set(newData)

    return render(request, 'insert_attendance.html')


def index(request):
    return render(request, 'index.html')


def attendance_list(request):
    # accessing our firebase data and storing it in a variable
    data = database.child('data').child('attendance').get() .val()
    js_data = dumps(data)
    # print(data)

    context = {
        'data': js_data
    }

    if request.method == 'POST':
        report_filter = request.POST["report_month"]
        print(report_filter)

    return render(request, 'attendance_list.html', context)


def loginUser(request):
    name = database.child('student').child('name').get().val()
    password = database.child('student').child('password').get().val()

    context = {
        'name': name,
        'password': password,
    }

    return render(request, 'login.html', context)


def onCLickDatasets(request):
    name = database.child('student').child('name').get().val()
    password = database.child('student').child('password').get().val()

    context = {
        'name': name,
        'password': password,
    }

    return render(request, 'datasets.html', context)


def onClickEmbeddings(request):
    name = database.child('student').child('name').get().val()
    password = database.child('student').child('password').get().val()

    context = {
        'name': name,
        'password': password
    }

    return render(request, 'embeddings.html', context)


def onClickTrain(request):
    # print("hi")
    name = database.child('student').child('name').get().val()
    password = database.child('student').child('password').get().val()

    context = {
        'name': name,
        'password': password,
    }

    return render(request, 'train.html', context)

def callExtractEmbedding(request):
    print(STUDENT_NAME)
    dataset ='dataset/'+STUDENT_NAME
    embeddings = 'output/embeddings.pickle'
    detector = 'face_detection_model'
    embedding_model = 'openface_nn4.small2.v1.t7'
    confidence_val = 0.5

    # load our serialized face detector from disk
    messages.success(request, "[INFO] loading face detector...")
    print("[INFO] loading face detector...")
    protoPath = os.path.sep.join([detector, "deploy.prototxt"])
    print(protoPath)

    modelPath = os.path.sep.join([detector,
                                "res10_300x300_ssd_iter_140000.caffemodel"])
    print(modelPath)
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    print(detector)

    # load our serialized face embedding model from disk
    print("[INFO] loading face recognizer...")
    messages.success(request, "[INFO] loading face recognizer...")
    embedder = cv2.dnn.readNetFromTorch(embedding_model)

    # grab the paths to the input images in our dataset
    print("[INFO] quantifying faces...")
    imagePaths = list(paths.list_images(dataset))

    # initialize our lists of extracted facial embeddings and
    # corresponding people names
    knownEmbeddings = []
    knownNames = []

    # initialize the total number of faces processed
    total = 0

    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        print("[INFO] processing image {}/{}".format(i + 1,
                                                    len(imagePaths)))
        messages.success(request, "[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]

        # load the image, resize it to have a width of 600 pixels (while
        # maintaining the aspect ratio), and then grab the image
        # dimensions
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize
        # faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()

        # ensure at least one face was found
        if len(detections) > 0:
            # we're making the assumption that each image has only ONE
            # face, so find the bounding box with the largest probability
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]

            # ensure that the detection with the largest probability also
            # means our minimum probability test (thus helping filter out
            # weak detections)
            if confidence > confidence_val:
                # compute the (x, y)-coordinates of the bounding box for
                # the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI and grab the ROI dimensions
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                # construct a blob for the face ROI, then pass the blob
                # through our face embedding model to obtain the 128-d
                # quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # add the name of the person + corresponding face
                # embedding to their respective lists
                knownNames.append(name)
                knownEmbeddings.append(vec.flatten())
                total += 1

        # dump the facial embeddings + names to disk
    print("[INFO] serializing {} encodings...".format(total))
    messages.success(request, "[INFO] serializing {} encodings...".format(total))
    # read embeddings and store in variable/obj
    data = {"embeddings": knownEmbeddings, "names": knownNames}
    f = open(embeddings, "ab")
    f.write(pickle.dumps(data))
    messages.success(request, "Embeddings Complete")
    # old variable/obj + new data append pickle file for new images
    f.close()

def embedding(request):
    print("extracting function",STUDENT_NAME)

    context = {
        'student_name':STUDENT_NAME
    }

    if request.method == 'POST':
        callExtractEmbedding(request)
    #  os.system('python3 extract_embeddings.py')
    # os.system('python extract_embeddings.py --dataset dataset --embeddings output/embeddings.pickle --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7')

    return render(request, 'embeddings.html',context)

def callTraining(request):
    embeddings_val = 'output/embeddings.pickle'
    recognizer_val = 'output/recognizer.pickle'
    le_val = 'output/le.pickle'

    # load the face embeddings
    print("[INFO] loading face embeddings...")
    messages.success(request, "[INFO] loading face embeddings...")
    data = pickle.loads(open(embeddings_val, "rb").read())

    # encode the labels
    print("[INFO] encoding labels...")
    messages.success(request, "[INFO] encoding labels...")
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])

    # train the model used to accept the 128-d embeddings of the face and
    # then produce the actual face recognition
    print("[INFO] training model...")
    messages.success(request, "[INFO] training model...")
    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)

    # write the actual face recognition model to disk
    f = open(recognizer_val, "ab")
    f.write(pickle.dumps(recognizer))
    f.close()

    # write the label encoder to disk
    f = open(le_val, "ab")
    f.write(pickle.dumps(le))
    messages.success(request,"[INFO] training model complete...")
    # append instead of write
    # file1 = open("myfile.txt", "ab")
    # a append, b binary

    f.close()

def trainModel(request):
    print("train model",STUDENT_NAME)

    context = {
        'student_name':STUDENT_NAME
    }

    if request.method == 'POST':
        callTraining(request)

    # os.system('python3 train_model.py')
    # os.system('python train_model.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle')
    return render(request, 'train.html',context)


def image_upload(request):
    context = dict()
    if request.method == 'POST':
        username = "Nauka"
        # username = request.POST["username"]
        image_path = request.POST[
            "src"]  # src is the name of input attribute in your html file, this src value is set in javascript code
        image = NamedTemporaryFile()
        # image.write("fireapp/images")
        print(path[0])
        # https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.amazon.com%2FLouis-Garden-Artificial-Silk-Flowers%2Fdp%2FB00YY0B2DG&psig=AOvVaw3zMAtPyVzr_elwIX9c1MDR&ust=1648100787686000&source=images&cd=vfe&ved=0CAsQjRxqFwoTCJCOxYLE2_YCFQAAAAAdAAAAABAE
        # image.write(urlopen(path[0]).read())
        image.write(urlopen(
            "https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.amazon.com%2FLouis-Garden-Artificial-Silk-Flowers%2Fdp%2FB00YY0B2DG&psig=AOvVaw3zMAtPyVzr_elwIX9c1MDR&ust=1648100787686000&source=images&cd=vfe&ved=0CAsQjRxqFwoTCJCOxYLE2_YCFQAAAAAdAAAAABAE").read())
        image.flush()
        image = File(image)
        print(image)
        name = str(image.name).split('\\')[-1]
        name += '.jpg'  # store image in jpeg format
        image.name = name
        if image is not None:
            print("if condition")
            obj = Image.objects.create(title=username,
                                       file="test_rf1.png")  # create a object of Image type defined in your model
            obj.save()
            context["path"] = obj.image.url  # url to image stored in my server/local device
            context["username"] = obj.username
        else:
            print("else executed")
        #     return redirect('/')
        # return redirect('any_url')
    return render(request, 'temp.html',
                  context=context)  # context is like respose data we are sending back to user, that will be rendered with specified 'html file'.
