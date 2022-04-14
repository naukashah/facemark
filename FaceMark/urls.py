"""FaceMark URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.conf.urls.static import static
from django.urls import path, include
from django.views.generic import TemplateView
from fireapp import views
from django.conf import settings
from django.conf.urls.static import static

from FaceMark import settings

urlpatterns = [
                path('admin/', admin.site.urls),
                path('', views.index, name='index'),
                
                # Menu clicks
                path('embeddings.html', views.onClickEmbeddings, name='embeddings'),
                path('train.html', views.onClickTrain, name='train'),
                path('login.html', views.loginUser, name='login'),
                path('camera', views.camera, name='camera'),
                path('logout/', views.logoutUser, name='logout'),
                path('recognition.html', views.facecam_feed, name='facecam_feed'),

                # Function calls
                path('embedding', views.embedding, name='embedding'),
                path('trainModel', views.trainModel, name='trainModel'),
                path('attendance_list', views.attendance_list, name='attendance_list'),
                # path('imageUploader', views.imageUploader),
                path('recognition', views.recognition, name='recognition'),

                path('temp/', include("django.contrib.auth.urls")),  # new
                path('postsignIn/', views.loginUser),


                path('home', views.onClickHome, name='home'),

                  
              ] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT, ) + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)