from django.shortcuts import render

import pyrebase
from json import dumps
from django.core.files import File
from django.core.files.base import ContentFile
from django.core.files.temp import NamedTemporaryFile
from urllib.request import urlopen
from PIL import Image

# Remember the code we copied from Firebase.
# This can be copied by clicking on the settings icon > project settings, then scroll down in your firebase dashboard
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

    
    if request.method== 'POST':        
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
        newData = { "id" : 1,"name" : request.POST.get('your_name'),"timestamp" : request.POST.get('timestamp'), "type" : request.POST.get('type') }
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
        'data' : js_data
    }

    return render(request, 'attendance_list.html', context)