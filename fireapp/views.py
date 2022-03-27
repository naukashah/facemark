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

# Remember the code we copied from Firebase.
# This can be copied by clicking on the settings icon > project settings, then scroll down in your firebase dashboard
# from facemark.fireapp.models import user_details

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
    if request.method == 'POST':
        # User_name = request.POST["Username"]
        # User_phone = request.POST["Userphone"]
        # User_address = request.POST["Useraddress"]
        # pic = request.FILES["photo"]
        # User_info= UserDetails(User_name=User_name, User_phone=User_phone, User_address=User_address, User_pic= pic)
        # User_info.save()    
        path = request.POST["src"]
        image = NamedTemporaryFile()
        image.write(urlopen(path).read())
        image.flush()
        image = File(image)
        name = str(image.name).split('\\')[-1]
        name += '.jpg'
        image.name = name
        obj = Image.objects.create(image=image)
        obj.save()

    return render(request, 'camera.html')


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
    data = database.child('data').child('attendance').get().val()
    js_data = dumps(data)
    # print(data)

    context = {
        'data': js_data
    }
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
        'password': password,
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


def embedding(request):
    print("extracting function")

    os.system('python3 extract_embeddings.py --dataset dataset --embeddings output/embeddings.pickle --detector '
              'face_detection_model --embedding-model openface_nn4.small2.v1.t7')

    return render(request, 'embeddings.html')


def trainModel(request):
    print("train model")
    os.system(
        'python3 train_model.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle')
    return render(request, 'train.html')


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
